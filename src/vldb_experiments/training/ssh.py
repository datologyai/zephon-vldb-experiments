"""Detach the single-node experiment on a user-supplied SSH/Docker host."""

from __future__ import annotations

import csv
import io
import json
import re
import shlex
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath

from .. import ROOT
from . import protocol, sources, trace
from .execution import write_json

PROBE_IMAGE = "nvidia/cuda:13.0.3-base-ubuntu24.04"
HOST_PREFLIGHT = r"""set -eu
fail() { printf '%s\n' "$*" >&2; exit 1; }
for tool in uname docker nvidia-smi tar gzip mktemp mkdir rmdir rm cat id dirname; do
    command -v "$tool" >/dev/null 2>&1 || fail "Missing prerequisite: $tool"
done
[ "$(uname -s)" = Linux ] || fail "The SSH training host must run Linux"
os=$(docker info --format '{{.OSType}}') || fail "Cannot access the Docker daemon as the SSH user"
[ "$os" = linux ] || fail "Docker must use a Linux daemon"
docker build --help >/dev/null 2>&1 || fail "Docker image building is unavailable"
nvidia-smi --query-gpu=uuid,name,compute_cap,driver_version --format=csv,noheader,nounits
"""

CONTAINER_PREFLIGHT = r"""set -eu
fail() { printf '%s\n' "$*" >&2; exit 1; }
ancestor=$1
while [ ! -d "$ancestor" ]; do
    [ ! -e "$ancestor" ] || fail "Remote output path has a non-directory component: $ancestor"
    ancestor=$(dirname -- "$ancestor")
done
probe=$(mktemp -d "$ancestor/.zephon-preflight.XXXXXX") || fail "Remote output directory is not writable: $ancestor"
cleanup() { rm -f -- "$probe/host-check" "$probe/container-check"; rmdir -- "$probe"; }
trap cleanup 0
trap 'exit 1' 1 2 15
printf 'ok\n' > "$probe/host-check"
docker run --rm --gpus "$2" --network none --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$probe,dst=/preflight" "$3" sh -c '
    test -r /preflight/host-check &&
    printf "ok\n" > /preflight/container-check &&
    exec nvidia-smi --query-gpu=uuid,name,compute_cap,driver_version --format=csv,noheader,nounits
' || fail "CUDA container probe failed; check the NVIDIA Container Toolkit, driver, GPU access and bind-mount permissions"
[ -s "$probe/container-check" ] || fail "Docker cannot write to the SSH host output directory"
"""


def ssh_command(host: str, args: list[str]) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9_.@:\[\]-]+", host) or host.startswith("-"):
        raise ValueError(
            "Use an SSH hostname or configured host alias, not SSH options"
        )
    return ["ssh", "-o", "BatchMode=yes", "--", host, shlex.join(args)]


def remote(host: str, args: list[str], **kwargs):
    return subprocess.run(ssh_command(host, args), check=True, **kwargs)


def gpu_inventory(raw: bytes, required: int, *, container: bool = False) -> list[dict]:
    rows = [
        row
        for row in csv.reader(io.StringIO(raw.decode()), skipinitialspace=True)
        if row
    ]
    if len(rows) < required:
        raise ValueError(
            f"SSH preflight: this campaign needs {required} GPUs; detected {len(rows)}"
        )
    if container and len(rows) != required:
        raise ValueError(
            "SSH preflight: Docker exposed a different GPU count than requested"
        )
    inventory = []
    for row in rows[:required]:
        if len(row) != 4:
            raise ValueError("SSH preflight: unrecognized nvidia-smi GPU inventory")
        gpu, name, capability, driver = (field.strip() for field in row)
        if (
            not re.fullmatch(r"\d+\.\d+", capability)
            or int(capability.split(".")[0]) < 8
        ):
            raise ValueError(
                f"SSH preflight: {name} needs Ampere or newer capability for BF16"
            )
        if (
            not re.fullmatch(r"\d+(?:\.\d+)+", driver)
            or int(driver.split(".")[0]) < 580
        ):
            raise ValueError(
                f"SSH preflight: CUDA 13 requires an NVIDIA driver >= 580; found {driver}"
            )
        inventory.append(
            {
                "uuid": gpu,
                "name": name,
                "compute_capability": capability,
                "driver": driver,
            }
        )
    if len({gpu["uuid"] for gpu in inventory}) != required:
        raise ValueError("SSH preflight: NVIDIA GPU identifiers are not distinct")
    return inventory


def preflight(host: str, root: PurePosixPath, required: int) -> dict:
    """Fail before uploading code or building the training image; never train here."""

    def probe(script: str, args: list[str], timeout: int) -> bytes:
        try:
            return remote(
                host,
                ["sh", "-s", "--", *args],
                input=script.encode(),
                capture_output=True,
                timeout=timeout,
            ).stdout
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or b"").decode(errors="replace").strip()
            raise RuntimeError(
                f"SSH preflight failed on {host}: {detail[-2000:]}"
            ) from error
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(
                f"SSH preflight could not finish on {host}: {error}"
            ) from error

    # Count physical host GPUs before Docker can pull even the small probe image.
    host_gpus = gpu_inventory(probe(HOST_PREFLIGHT, [], 60), required)
    container_gpus = gpu_inventory(
        probe(CONTAINER_PREFLIGHT, [str(root), str(required), PROBE_IMAGE], 180),
        required,
        container=True,
    )
    return {
        "schema": 1,
        "host": host,
        "required_gpus": required,
        "probe_image": PROBE_IMAGE,
        "host_gpus": host_gpus,
        "container_gpus": container_gpus,
        "writable_bind_mount": True,
    }


def dockerfile(config: dict) -> str:
    commit = config["sources"]["torchtitan"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("The trainer requires an immutable public source revision")
    local = sources.image_sources(config)
    requirements = "--no-emit-package zephon" if "zephon" in local else ""
    install = ""
    if "zephon" in local:
        install += "COPY local_sources/zephon /opt/local_sources/zephon\nRUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ZEPHON=0+unknown uv pip install --python /opt/venv/bin/python --no-deps -e /opt/local_sources/zephon\n"
    if "torchtitan" in local:
        install += "COPY local_sources/torchtitan /opt/local_sources/torchtitan\nRUN ln -s /opt/local_sources/torchtitan /opt/torchtitan\n"
    else:
        install += f"RUN git clone https://github.com/datologyai/torchtitan-zephon.git /opt/torchtitan && git -C /opt/torchtitan checkout --detach {commit}\n"
    return f"""FROM nvidia/cuda:13.0.3-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3.12-venv git libibverbs1 ibverbs-providers rdma-core ca-certificates && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv && /opt/venv/bin/pip install uv==0.11.8
ENV PATH=/opt/venv/bin:$PATH
COPY environments/training /opt/experiments/environments/training
RUN uv export --project /opt/experiments/environments/training --locked --no-emit-project --no-hashes {requirements} --output-file /tmp/training-requirements.txt && uv pip install --python /opt/venv/bin/python -r /tmp/training-requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu130 --index-strategy unsafe-best-match
{install}RUN uv pip freeze --python /opt/venv/bin/python > /opt/environment.txt
COPY src /opt/experiments/src
ENV PYTHONPATH=/opt/experiments/src:/opt/torchtitan PYTORCH_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
CMD ["python", "-m", "vldb_experiments.training.ssh_runner"]
"""


def source_archive(manifest: dict) -> bytes:
    """Upload only fingerprinted training code and its locked public dependencies."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, expected in manifest["code"]["files"].items():
            path = ROOT / name
            if protocol.file_digest(path) != expected:
                raise ValueError(f"Source changed while preparing the image: {name}")
            archive.add(path, arcname=name, recursive=False)
        for name, entry in sources.image_sources(manifest["config"]).items():
            sources.add_to_archive(archive, name, entry)
        content = dockerfile(manifest["config"]).encode()
        info = tarfile.TarInfo("Dockerfile")
        info.size = len(content)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def read_manifest(run: Path) -> dict:
    return json.loads((run / "campaign.json").read_text())


def inspect(host: str, container: str) -> dict:
    fields = '{"id":{{json .Id}},"image":{{json .Image}},"state":{{json .State}}}'
    result = remote(
        host, ["docker", "inspect", "--format", fields, container], capture_output=True
    )
    return json.loads(result.stdout)


def storage_root(value: str | None) -> PurePosixPath:
    if (
        not value
        or not PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
        or any(c in value for c in ",\n\r")
    ):
        raise ValueError("--ssh requires an absolute --remote-root without '..'")
    return PurePosixPath(value)


def assets_directory(root: PurePosixPath, prefix: str) -> PurePosixPath:
    # Keep input preparation independent of the campaign's ownership directory.
    return root / f"{prefix}-training-assets"


def image_name(manifest: dict) -> str:
    return (
        "zephon-training:"
        + protocol.digest(
            {
                "code": manifest["code"],
                "sources": manifest["config"]["sources"],
                "local_sources": manifest["config"].get("local_sources", {}),
            }
        )[:24]
    )


def input_command(
    host: str, image: str, assets: PurePosixPath, config: dict, action: str
) -> dict:
    if action == "prepare":
        remote(host, ["mkdir", "-p", "--", str(assets)])
    command = ["docker", "run", "--rm", "--interactive", "--pull=never"]
    if action == "check":
        command += ["--network", "none"]
    command += [
        "--mount",
        f"type=bind,src={assets},dst=/assets"
        + (",readonly" if action == "check" else ""),
        image,
        "python",
        "-m",
        "vldb_experiments.training.assets",
        action,
    ]
    result = remote(
        host, command, input=json.dumps(config).encode(), stdout=subprocess.PIPE
    )
    return json.loads(result.stdout)


def run_campaign(args) -> None:
    config = sources.configure(
        args, protocol.load(args.config, args.smoke, getattr(args, "small", False))
    )
    stages = protocol.plan(config)
    if any(s["nodes"] != 1 or s["dp"] > 8 for s in stages):
        raise ValueError(
            "SSH needs a single-node config; select --small for DP2 -> DP1/4/8"
        )
    root = storage_root(args.remote_root)
    ssh_command(args.ssh, ["true"])
    prefix = protocol.validate_prefix(args.prefix)
    base = root / prefix
    assets = assets_directory(root, prefix)
    backend = {
        "kind": "ssh",
        "host": args.ssh,
        "directory": str(base),
        "container": f"{prefix}-training",
    }
    manifest = {
        "schema": 1,
        "config": config,
        "prefix": prefix,
        "code": protocol.code_identity(),
        "backend": backend,
    }
    manifest["identity"] = protocol.digest(manifest)
    output = args.output.expanduser().resolve()
    if (output / "campaign.json").exists() and read_manifest(output) != manifest:
        raise ValueError(
            "Campaign identity changed; use a new output directory and prefix"
        )
    launch_path = output / "launch.json"
    if launch_path.exists():
        previous = json.loads(launch_path.read_text())
        if previous["state"] != "build_failed":
            try:
                info = inspect(
                    args.ssh, previous.get("container_id", backend["container"])
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    "Uncertain SSH submission; inspect the original host before any retry"
                ) from error
            write_json(
                launch_path,
                previous
                | {
                    "state": "submitted",
                    "container_id": info["id"],
                    "image_id": info["image"],
                },
            )
            print(f"Already submitted: {info['id']}")
            return
    proof = preflight(args.ssh, base.parent, max(s["dp"] for s in stages))
    from .preparation import require_inputs

    require_inputs(args, config)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "campaign.json", manifest)
    write_json(output / "preflight.json", proof)
    if not launch_path.exists():
        remote(args.ssh, ["mkdir", "-p", "--", str(base.parent)])
        # Atomic ownership claim: an existing directory is never overwritten.
        remote(args.ssh, ["mkdir", "--", str(base)])
        remote(
            args.ssh,
            [
                "mkdir",
                "--",
                str(base / "runs"),
            ],
        )
        remote(
            args.ssh,
            ["sh", "-c", f"cat > {shlex.quote(str(base / 'runs/campaign.json'))}"],
            input=(output / "campaign.json").read_bytes(),
        )
    image = image_name(manifest)
    launch = {"state": "prepared", "identity": manifest["identity"], **backend}
    write_json(launch_path, launch)
    required = max(s["dp"] for s in stages)
    result = remote(
        args.ssh,
        [
            "docker",
            "run",
            "--detach",
            "--name",
            backend["container"],
            "--gpus",
            str(required),
            "--ipc=host",
            "--init",
            "--mount",
            f"type=bind,src={base / 'runs'},dst=/runs",
            "--mount",
            f"type=bind,src={assets},dst=/assets,readonly",
            image,
        ],
        capture_output=True,
    )
    container = result.stdout.decode().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", container):
        raise RuntimeError("Container creation is uncertain; inspect the original host")
    info = inspect(args.ssh, container)
    write_json(
        launch_path,
        launch
        | {"state": "submitted", "container_id": info["id"], "image_id": info["image"]},
    )
    print(
        f"Submitted SSH container {container}; inspect with: reproduce training status {output}"
    )


def target(run: Path) -> tuple[dict, dict]:
    manifest = read_manifest(run)
    launch = json.loads((run / "launch.json").read_text())
    if not launch.get("container_id"):
        raise RuntimeError(
            "Uncertain SSH submission; inspect the original host before continuing"
        )
    info = inspect(manifest["backend"]["host"], launch["container_id"])
    if info["id"] != launch["container_id"] or info["image"] != launch["image_id"]:
        raise ValueError(
            "The SSH container or immutable image differs from the launch receipt"
        )
    return manifest, info


def status(run: Path) -> dict:
    try:
        manifest, info = target(run)
        backend = manifest["backend"]
        try:
            result = remote(
                backend["host"],
                ["cat", str(PurePosixPath(backend["directory"]) / "runs/status.json")],
                capture_output=True,
            )
            recorded = json.loads(result.stdout)
        except subprocess.CalledProcessError:
            if info["state"]["Running"]:
                return {
                    "state": "running",
                    "phase": "initializing",
                    "container_id": info["id"],
                }
            return {
                "state": "failed",
                "exit_code": info["state"]["ExitCode"],
                "container_id": info["id"],
            }
    except (
        subprocess.CalledProcessError,
        OSError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        return {"state": "unknown", "error": str(error)}
    if info["state"]["Running"]:
        return recorded | {"state": "running", "container_id": info["id"]}
    if info["state"]["ExitCode"] == 0 and recorded["state"] == "complete":
        return recorded | {"container_id": info["id"]}
    return recorded | {
        "state": "failed",
        "exit_code": info["state"]["ExitCode"],
        "container_id": info["id"],
    }


def continue_campaign(run: Path) -> None:
    manifest, info = target(run)
    if info["state"]["Running"] or info["state"].get("Restarting"):
        print(f"Already running: {info['id']}")
        return
    if status(run)["state"] == "complete":
        print(f"Already complete: {info['id']}")
        return
    # A terminal container has no surviving GPU workers. An explicit continuation
    # permits a fresh attempt of unfinished work, never a completed stage.
    backend = manifest["backend"]
    request = {"identity": manifest["identity"], "request_id": uuid.uuid4().hex}
    path = str(PurePosixPath(backend["directory"]) / "runs/continue.json")
    lock = str(PurePosixPath(backend["directory"]) / "continue-lock")
    state = shlex.join(
        ["docker", "inspect", "--format", "{{.State.Running}}", info["id"]]
    )
    cleanup = shlex.join(["rmdir", "--", lock])
    script = (
        f"mkdir -- {shlex.quote(lock)} || exit 1; "
        f"trap {shlex.quote(cleanup)} EXIT; "
        f'node_state=$({state}) || exit 1; [ "$node_state" = false ] || exit 0; '
        f"set -C; cat > {shlex.quote(path)} && {shlex.join(['docker', 'start', info['id']])}"
    )
    remote(backend["host"], ["sh", "-c", script], input=json.dumps(request).encode())
    print(f"Continued original SSH container {info['id']}; completed stages are reused")


def evidence_files(config: dict) -> list[str]:
    files = ["campaign.json", "assets.json", "result.json"]
    for stage in protocol.plan(config):
        name = stage["name"]
        files.extend(
            f"{name}/diagnostics/rank-{rank}.jsonl" for rank in range(stage["dp"])
        )
        files.extend(
            f"{name}/{suffix}"
            for suffix in (
                "observed.json",
                "checkpoint-proof.json",
                "final.metadata",
                "launch.json",
            )
        )
        if protocol.source_name(stage) is None:
            files.append(f"{name}/source.metadata")
    return files


def unpack_evidence(raw: bytes, destination: Path, expected: list[str]) -> None:
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != set(
            expected
        ):
            raise ValueError(
                "SSH evidence archive contains unexpected or missing files"
            )
        for member in members:
            if (
                not member.isfile()
                or PurePosixPath(member.name).is_absolute()
                or ".." in PurePosixPath(member.name).parts
            ):
                raise ValueError(
                    "SSH evidence must contain regular files inside the campaign"
                )
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as stream:
                path.write_bytes(stream.read())


def collect(run: Path) -> dict:
    manifest, _ = target(run)
    if status(run)["state"] != "complete":
        raise ValueError(
            "Wait for the complete SSH campaign before collecting its final report"
        )
    backend = manifest["backend"]
    names = evidence_files(manifest["config"])
    raw = remote(
        backend["host"],
        [
            "tar",
            "-C",
            str(PurePosixPath(backend["directory"]) / "runs"),
            "-cf",
            "-",
            "--",
            *names,
        ],
        capture_output=True,
    ).stdout
    with tempfile.TemporaryDirectory(prefix="zephon-training-evidence-") as temporary:
        staging = Path(temporary)
        unpack_evidence(raw, staging, names)
        if read_manifest(staging) != manifest:
            raise ValueError("Remote and local campaign identities differ")
        trace.report(staging)
        for name in names:
            target_path = run / name
            content = (staging / name).read_bytes()
            if target_path.exists() and target_path.read_bytes() != content:
                raise ValueError(f"Existing collected evidence differs: {name}")
        for name in names:
            path = run / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes((staging / name).read_bytes())
    return trace.report(run)
