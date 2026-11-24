"""Run the same CPU CLI on SSH targets; collect raw evidence for local reporting."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

from . import ROOT, core, machines, prepare, profiles, provenance


def connection(path: Path, raw: dict) -> list[str]:
    machine = raw["machine"]
    args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    identity = machine.get("identity_file")
    if identity:
        args += ["-i", str(Path(identity).expanduser().absolute())]
    if machine["kind"] == "aws":
        receipt = core.read_json(machines.receipt_path(path, raw))
        if receipt["state"] != "ready":
            raise ValueError("Run machine create and resolve its launch receipt first")
        request = machines.aws_plan(raw)
        if request != receipt["request"]:
            raise ValueError("AWS profile differs from the recorded instance request")
        if request["connect"] == "ssm":
            if not shutil.which("aws"):
                raise ValueError(
                    "SSM connections require AWS CLI and the Session Manager plugin on the controller"
                )
            proxy = shlex.join(
                [
                    "aws",
                    "--region",
                    receipt["region"],
                    "ssm",
                    "start-session",
                    "--target",
                    receipt["instance_id"],
                    "--document-name",
                    "AWS-StartSSHSession",
                    "--parameters",
                    "portNumber=22",
                ]
            )
            args += ["-o", "ProxyCommand=" + proxy]
            host = receipt["instance_id"]
        else:
            host = receipt["private_ip"]
        host = machine.get("user", "ubuntu") + "@" + host
        args += [
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "UserKnownHostsFile="
            + str(machines.receipt_path(path, raw))
            + ".known_hosts",
        ]
    else:
        host = machine.get("host", "")
    if (
        not host
        or not re.fullmatch(r"[A-Za-z0-9_.@:\[\]-]+", host)
        or host.startswith("-")
    ):
        raise ValueError("Set machine.host to an SSH host or alias")
    return args + [host]


def call(ssh, argv, *, data=None, capture=True):
    """OpenSSH passes one shell command; quote every argument exactly once."""
    return subprocess.run(
        [*ssh, shlex.join(list(map(str, argv)))],
        input=data,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=True,
    )


def bootstrap_aws(path: Path, raw: dict):
    """Finish bootstrap/reboot only on the instance owned by this profile."""
    receipt_path = machines.receipt_path(path, raw)
    receipt = core.read_json(receipt_path)
    if receipt.get("bootstrap_complete"):
        return
    ssh = connection(path, raw)
    deadline = time.monotonic() + 1800

    def available(previous=None):
        while time.monotonic() < deadline:
            try:
                boot = (
                    call(ssh, ["cat", "/proc/sys/kernel/random/boot_id"])
                    .stdout.decode()
                    .strip()
                )
                if previous is None or boot != previous:
                    return boot
            except subprocess.CalledProcessError:
                pass
            time.sleep(5)
        raise RuntimeError(
            "Instance did not become SSH-ready; inspect the saved EC2 receipt"
        )

    boot = available()
    call(ssh, ["cloud-init", "status", "--wait"])
    cmdline = call(ssh, ["cat", "/proc/cmdline"]).stdout.decode()
    if (
        "intel_idle.max_cstate=1" not in cmdline
        or "processor.max_cstate=1" not in cmdline
    ):
        # The receipt refers to our new instance, not a supplied shared SSH host.
        lock = raw["machine"]["root"].rstrip("/") + "/.measurement.lock"
        code = """import fcntl,pathlib,subprocess,sys
p=pathlib.Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)
with p.open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 subprocess.run(['sudo','-n','reboot'],check=True)
"""
        try:
            call(ssh, ["python3", "-c", code, lock])
        except subprocess.CalledProcessError as error:
            if error.returncode != 255:
                raise
        available(boot)
        call(ssh, ["cloud-init", "status", "--wait"])
        cmdline = call(ssh, ["cat", "/proc/cmdline"]).stdout.decode()
        if (
            "intel_idle.max_cstate=1" not in cmdline
            or "processor.max_cstate=1" not in cmdline
        ):
            raise RuntimeError("CPU boot settings did not take effect after reboot")
    receipt["bootstrap_complete"] = True
    core.write_json(receipt_path, receipt)


def source_archive() -> tuple[str, bytes]:
    members = []
    for name in ("src/vldb_experiments", "configs", "environments"):
        members += [
            p
            for p in (ROOT / name).rglob("*")
            if p.is_file()
            and p.name != ".DS_Store"
            and not {"__pycache__", ".git", ".venv"}.intersection(p.parts)
        ]
    if any(p.is_symlink() for p in members):
        raise ValueError("Launcher source must contain regular files, not symlinks")
    inventory = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(members)
    }
    identity = core.digest(inventory)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for p in members:
            archive.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
        data = json.dumps(
            dict(
                repository="zephon-vldb-experiments", revision=identity, files=inventory
            ),
            sort_keys=True,
        ).encode()
        info = tarfile.TarInfo("SOURCE.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return identity, stream.getvalue()


def deploy(path: Path, raw: dict) -> tuple[list[str], str, str]:
    target = profiles.local_target(raw)
    ssh = connection(path, raw)
    identity, archive = source_archive()
    root = raw["machine"]["root"].rstrip("/")
    source = root + "/launcher/" + identity
    # Extraction is confined to a new run-owned directory. Never update a
    # controller snapshot that an active measurement might be importing.
    code = """import hashlib,io,json,os,pathlib,shutil,sys,tarfile,tempfile
destination=pathlib.Path(sys.argv[1]); parent=destination.parent
parent.mkdir(parents=True,exist_ok=True)
if not destination.exists():
 temp=pathlib.Path(tempfile.mkdtemp(prefix='.upload-',dir=parent))
 try:
  with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
   if any(not m.isfile() for m in archive.getmembers()): raise ValueError('Snapshot must contain regular files')
   archive.extractall(temp,filter='data')
  inventory=json.loads((temp/'SOURCE.json').read_text())['files']
  for name,digest in inventory.items():
   if hashlib.sha256((temp/name).read_bytes()).hexdigest()!=digest: raise ValueError('Snapshot hash mismatch')
  if hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()!=sys.argv[2]: raise ValueError('Snapshot identity mismatch')
  try: temp.rename(destination)
  except FileExistsError: pass
 finally:
  if temp.exists(): shutil.rmtree(temp)
inventory=json.loads((destination/'SOURCE.json').read_text())['files']
for name,digest in inventory.items():
 if hashlib.sha256((destination/name).read_bytes()).hexdigest()!=digest: raise ValueError('Existing snapshot changed')
"""
    call(
        ssh,
        [raw["machine"].get("python", "python3"), "-c", code, source, identity],
        data=archive,
    )
    # Explicit controller sources travel with the launcher unless a target path
    # was selected. Each upload gets its own immutable, content-addressed path.
    for name in ("arena", "plots", "zephon"):
        local = raw.get("controller", {}).get(name)
        if not local or raw.get("paths", {}).get(name):
            continue
        identity, data = repository_archive(Path(local))
        destination = f"{root}/uploaded/{name}/{identity}"
        call(
            ssh,
            [
                raw["machine"].get("python", "python3"),
                "-c",
                code,
                destination,
                identity,
            ],
            data=data,
        )
        target.setdefault("paths", {})[name] = destination
        if name == "arena" and "arena" in target.get("checkpoint", {}):
            target["checkpoint"]["arena"] = destination
    profile_data = json.dumps(target, sort_keys=True).encode()
    target_path = (
        root + "/profiles/" + hashlib.sha256(profile_data).hexdigest() + ".json"
    )
    receipt = dict(
        ssh=ssh, python=raw["machine"].get("python", "python3"), source=source
    )
    agent(receipt, "profile", target_path, data=profile_data)
    return ssh, source, target_path


def repository_archive(path: Path) -> tuple[str, bytes]:
    """Copy a supplied checkout or plain source directory without Git history."""
    state = provenance.source_state(path)
    inventory = provenance.source_files(path)
    identity = core.digest(inventory)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name in inventory:
            archive.add(
                provenance.source_path(path, name), arcname=name, recursive=False
            )
        data = json.dumps(state | {"files": inventory}, sort_keys=True).encode()
        info = tarfile.TarInfo("SOURCE.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    if provenance.source_files(path) != inventory:
        raise ValueError(f"Source changed while preparing upload: {path}")
    return identity, stream.getvalue()


def command(raw, source, argv):
    return [
        "env",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=" + source + "/src",
        raw["machine"].get("python", "python3"),
        "-m",
        "vldb_experiments.cli",
        *argv,
    ]


def agent(receipt, action, path, *, data=None):
    argv = [
        "env",
        "PYTHONPATH=" + receipt["source"] + "/src",
        "PYTHONDONTWRITEBYTECODE=1",
        receipt["python"],
        "-m",
        "vldb_experiments.remote_worker",
        action,
        str(path),
    ]
    return call(receipt["ssh"], argv, data=data)


def status(run: Path) -> dict:
    receipt = core.read_json(run / "remote.json")
    return json.loads(agent(receipt, "status", receipt["job"]).stdout)


def collect(run: Path, *, plots: Path | None = None, finish=True):
    receipt = core.read_json(run / "remote.json")
    state = status(run)
    if state["state"] != "complete":
        raise RuntimeError(
            f"Remote job is {state['state']}; logs: {receipt['job']}/command.log"
        )
    if receipt["action"] != "run":
        return state
    with tempfile.TemporaryDirectory(prefix="vldb-collect-") as directory:
        temp = Path(directory)
        archive_path = temp / "results.tar.gz"
        with archive_path.open("wb") as file:
            argv = [
                "env",
                "PYTHONPATH=" + receipt["source"] + "/src",
                "PYTHONDONTWRITEBYTECODE=1",
                receipt["python"],
                "-m",
                "vldb_experiments.remote_worker",
                "archive",
                receipt["output"],
            ]
            subprocess.run([*receipt["ssh"], shlex.join(argv)], stdout=file, check=True)
        extracted = temp / "raw"
        with tarfile.open(archive_path) as archive:
            if any(not m.isfile() for m in archive.getmembers()):
                raise ValueError("Collected archive must contain regular files")
            archive.extractall(extracted, filter="data")
        files = list(extracted.rglob("*"))
        # Check all existing files before writing any new ones. Local renderings
        # and the controller receipt are outside the raw result namespace.
        for file in files:
            if file.is_file():
                dest = run / file.relative_to(extracted)
                if dest.exists() and dest.read_bytes() != file.read_bytes():
                    raise ValueError(
                        f"Collected evidence conflicts with existing file: {dest}"
                    )
        for file in files:
            if file.is_file():
                dest = run / file.relative_to(extracted)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(file.read_bytes())
    if finish:
        from . import summary

        summary.finish(run, plots=plots or Path(receipt["plots"]))
    return state


def wait(run: Path):
    previous = None
    while True:
        state = status(run)
        if state["state"] in ("complete", "failed", "orphaned"):
            if state["state"] != "complete":
                raise RuntimeError(
                    f"Remote job failed; logs: {core.read_json(run / 'remote.json')['job']}/command.log"
                )
            return state
        current = (state["state"], state.get("progress", ""))
        if current != previous:
            print(
                f"Remote job: {current[0]}" + (": " + current[1] if current[1] else ""),
                file=sys.stderr,
            )
            previous = current
        time.sleep(10)


def execute(args, argv: list[str]) -> int:
    """Submit once; repeated commands attach to the recorded remote owner."""
    path = args.profile.expanduser().absolute()
    raw = profiles.read(path)
    action = args.command
    argv = [argument for argument in argv if argument != "--detach"]
    # Plans stay local and never need SSH, even before a host exists.
    if action == "plan" or getattr(args, "plan", False):
        return -1
    # A remote setup also prepares reporting on the controller before starting
    # expensive target setup. The same destination is saved for collection.
    plots = prepare.controller_plots(raw)
    if action == "machine" and args.machine_command == "setup":
        with redirect_stdout(sys.stderr):
            prepare.plotting(
                plots, existing=bool(raw.get("controller", {}).get("plots"))
            )
    local_output = getattr(args, "output", None)
    identity = core.digest(
        dict(
            arguments=argv,
            profile=raw,
            source=source_archive()[0],
            repositories={
                name: provenance.source_state(Path(directory))
                for name, directory in raw.get("controller", {}).items()
                if name in ("arena", "plots", "zephon")
            },
        )
    )
    run = (
        Path(local_output).expanduser().absolute()
        if local_output and action == "run"
        else path.parent / ".jobs" / identity[:20]
    )
    run.mkdir(parents=True, exist_ok=True)
    receipt_path = run / "remote.json"
    if receipt_path.exists():
        receipt = core.read_json(receipt_path)
        if receipt["identity"] != identity:
            raise ValueError(
                "Remote command, profile or launcher changed; use a new output directory"
            )
    else:
        if (
            raw["machine"]["kind"] == "aws"
            and action == "machine"
            and args.machine_command == "setup"
        ):
            bootstrap_aws(path, raw)
        ssh, source, target_profile = deploy(path, raw)
        root = raw["machine"]["root"].rstrip("/")
        job = root + "/jobs/" + identity
        output = root + "/runs/" + identity
        remote_args = list(argv)

        def replace_option(name, value):
            for index, argument in enumerate(remote_args):
                if argument == name:
                    remote_args[index + 1] = value
                    return
                if argument.startswith(name + "="):
                    remote_args[index] = name + "=" + value
                    return
            raise ValueError("Missing remote option: " + name)

        replace_option("--profile", target_profile)
        if any(a == "--output" or a.startswith("--output=") for a in remote_args):
            replace_option("--output", output)
        if action == "prepare" and args.component == "checkpoint-data" and not args.yes:
            call(
                ssh,
                command(
                    raw,
                    source,
                    [
                        "prepare",
                        "checkpoint-data",
                        "--profile",
                        target_profile,
                        "--plan",
                    ],
                ),
                capture=False,
            )
            prompt = "Type yes to accept this preparation on the target: "
            if args.abandon:
                prompt = "Type yes to discard this unfinished preparation and its owned output: "
            if input(prompt).strip() != "yes":
                raise ValueError("Checkpoint preparation declined")
            remote_args.append("--yes")
        receipt = dict(
            identity=identity,
            action=action,
            ssh=ssh,
            job=job,
            output=output,
            source=source,
            python=raw["machine"].get("python", "python3"),
            plots=str(plots),
            argv=command(raw, source, remote_args),
        )
        # Persist before starting the remote owner. An uncertain SSH disconnect
        # attaches to this job; it never submits a second benchmark.
        with receipt_path.open("x") as file:
            json.dump(receipt, file, indent=2)
    agent(receipt, "launch", receipt["job"], data=json.dumps(receipt).encode())
    if getattr(args, "detach", False):
        print(f"Submitted remote job. Inspect with: reproduce status {run}")
        return 0
    wait(run)
    if action == "run":
        collect(run)
    else:
        result = agent(receipt, "log", receipt["job"])
        sys.stdout.write(result.stdout.decode())
        if action == "check" and local_output:
            data = call(receipt["ssh"], ["cat", receipt["output"]]).stdout
            destination = Path(local_output).expanduser().absolute()
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError(
                    "Validation receipt differs; choose a new --output path"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    return 0


def resume(run: Path):
    """Explicitly retry a failed owner, keeping its source and completed cases."""
    receipt = core.read_json(run / "remote.json")
    request = json.loads(agent(receipt, "resume", receipt["job"]).stdout)
    core.write_json(run / "remote.json", request)
    wait(run)
    collect(run)
