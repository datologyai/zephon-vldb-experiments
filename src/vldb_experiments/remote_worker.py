"""Target-side job lifecycle and evidence transfer for the CPU launcher."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from . import core


def process_start(pid: int) -> str | None:
    path = Path(f"/proc/{pid}/stat")
    return path.read_text().rsplit(") ", 1)[1].split()[19] if path.exists() else None


def status(job: Path) -> dict:
    path = job / "state.json"
    state = core.read_json(path) if path.exists() else dict(state="submitted")
    if state["state"] == "running":
        try:
            os.kill(state["pid"], 0)
            if state.get("process_start") != process_start(state["pid"]):
                state["state"] = "orphaned"
        except ProcessLookupError:
            state["state"] = "orphaned"
    log = job / "command.log"
    if log.exists():
        with log.open("rb") as file:
            file.seek(max(0, log.stat().st_size - 4096))
            lines = file.read().decode(errors="replace").splitlines()
        state["progress"] = lines[-1][-400:] if lines else ""
    return state


def spawn(job: Path, request: dict):
    env = os.environ | {
        "PYTHONPATH": request["source"] + "/src",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    with (job / "job.log").open("ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", __spec__.name, "run", str(job)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=env,
            start_new_session=True,
        )


def launch(job: Path, request: dict):
    job.mkdir(parents=True, exist_ok=True)
    path = job / "request.json"
    try:
        with path.open("x") as file:
            json.dump(request, file, indent=2)
    except FileExistsError:
        if core.read_json(path) != request:
            raise ValueError("Existing remote request differs")
    # Concurrent starts can create two short-lived owners, but only the owner
    # holding .owner.lock can spawn a benchmark. An uncertain submission is safe
    # to attach to again using this same job directory.
    with (job / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = status(job)["state"]
        renew = request["action"] != "run" and (
            current == "complete"
            or (current == "failed" and request["action"] == "check")
        )
        if renew:
            with (job / ".owner.lock").open("a") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX)
                preserve_attempt(job)
            current = "submitted"
        if current == "submitted":
            spawn(job, request)


def preserve_attempt(job: Path):
    attempt = len(list(job.glob("attempt-*-state.json"))) + 1
    (job / "state.json").rename(job / f"attempt-{attempt}-state.json")
    log = job / "command.log"
    if log.exists():
        log.rename(job / f"attempt-{attempt}-command.log")


def run(job: Path):
    with (job / ".owner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        state = job / "state.json"
        if state.exists():
            return
        request = core.read_json(job / "request.json")
        core.write_json(
            state,
            dict(
                state="running",
                pid=os.getpid(),
                process_start=process_start(os.getpid()),
                started_at=core.utc_now(),
            ),
        )
        try:
            code = core.run_process_group(
                request["argv"],
                str(job),
                os.environ | {"VLDB_REMOTE_WORKER": "1"},
                job / "command.log",
                7 * 24 * 3600,
            )
            result = dict(state="complete" if code == 0 else "failed", returncode=code)
        except BaseException as error:
            core.write_json(
                state,
                dict(state="failed", error=str(error), finished_at=core.utc_now()),
            )
            raise
        else:
            core.write_json(state, result | dict(finished_at=core.utc_now()))


def resume(job: Path) -> dict:
    with (job / ".owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status(job)["state"] != "failed":
            raise ValueError(
                "Only a failed, stopped owner may be resumed; inspect orphaned jobs before recovery"
            )
        preserve_attempt(job)
        request = core.read_json(job / "request.json")
        if (
            "checkpoint" in request["argv"]
            and "run" in request["argv"]
            and "--retry-failed" not in request["argv"]
        ):
            request["argv"].append("--retry-failed")
        core.write_json(job / "request.json", request)
    spawn(job, request)
    return request


def archive(root: Path):
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as output:
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("Result archive contains a symlink")
            if path.is_file():
                output.add(path, arcname=str(path.relative_to(root)), recursive=False)


def write_profile(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as file:
            file.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ValueError("Existing target profile differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("launch", "run", "status", "resume", "archive", "profile", "log"),
    )
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.action == "launch":
        launch(args.path, json.load(sys.stdin))
    elif args.action == "run":
        run(args.path)
    elif args.action == "status":
        print(json.dumps(status(args.path)))
    elif args.action == "resume":
        print(json.dumps(resume(args.path)))
    elif args.action == "archive":
        archive(args.path)
    elif args.action == "profile":
        write_profile(args.path, sys.stdin.buffer.read())
    elif args.action == "log":
        print((args.path / "command.log").read_text(), end="")


if __name__ == "__main__":
    main()
