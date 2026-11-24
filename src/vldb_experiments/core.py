"""Run planned cases through dataloader-arena's sweep.py and keep their results.

A campaign directory contains ``campaign.json``, which records the cases and the
identity of every input, and one folder per case under ``cases/`` with its arena
configs, logs, status and raw result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import ROOT, provenance

# Reported for a case that exceeded its time limit, as timeout(1) does.
TIMEOUT_RETURNCODE = 124


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_profile(path: Path) -> dict[str, Any]:
    from .profiles import load

    return load(path)


def environment_path(
    profile: dict[str, Any], case: dict[str, Any], revision: str
) -> Path:
    """All CPU benchmarks share the same package set per interpreter."""
    environment = f"cpu-{case['python']}"
    return Path(profile["paths"]["environments"]) / environment / "bin/python"


def is_free_threaded(case: dict[str, Any]) -> bool:
    return case["python"].endswith("t")


def case_state(directory: Path) -> str:
    """A case's recorded state: pending, running, complete, failed or interrupted."""
    path = directory / "status.json"
    return read_json(path)["state"] if path.exists() else "pending"


def sweep_command(
    profile: dict[str, Any],
    case: dict[str, Any],
    revision: str,
    attempt: Path,
) -> list[str]:
    """Write the arena configs for one case and return its sweep.py invocation.

    All benchmark execution belongs to sweep.py; these configs only describe the case.
    """
    python = environment_path(profile, case, revision)
    benchmark_path = attempt / "benchmark.json"
    sweep_path = attempt / "sweep.json"
    write_json(
        benchmark_path,
        {
            "loaders": [
                {
                    "name": case["loader_name"],
                    "factory": case["factory"],
                    "config": case["config"],
                }
            ]
        },
    )
    # Arena's CLI accepts target samples and scales them to batch iterations.
    batch_size = case["batch_size"] or 1
    bench_args = [
        "--batch-size",
        str(case["batch_size"]) if case["batch_size"] is not None else "none",
        "--num-workers",
        str(case["num_workers"]),
        "--seed",
        str(case["seed"]) if case["seed"] is not None else "none",
        "--warmup",
        str(case["warmup_batches"] * batch_size),
        "--steps",
        str(case["measure_batches"] * batch_size),
        "--strict",
        "--no-progress",
    ]
    if case.get("packing_stats"):
        bench_args.append("--packing-stats")
    if is_free_threaded(case):
        bench_args.append("--require-no-gil")
    write_json(
        sweep_path,
        {
            "python_versions": [case["python"]],
            "runs": [{"name": case["id"], "local": profile["paths"][case["dataset"]]}],
            "bench_config": str(benchmark_path),
            "bench_args": bench_args,
            "preinstall": False,
            "force_gil_off": is_free_threaded(case),
            "uv_project_environment_template": str(python.parent.parent),
        },
    )
    return [
        str(python),
        str(Path(profile["paths"]["arena"]) / "sweep.py"),
        "--config",
        str(sweep_path),
        "--results-dir",
        str(attempt / "sweep"),
        "--no-sync",
        "--show-live",
    ]


def sweep_environment(profile: dict[str, Any], case: dict[str, Any]) -> dict[str, str]:
    """Fix library threading, hashing and the GIL state for one case."""
    return (
        os.environ
        | profile.get("env", {})
        | {
            "PYTHONPATH": str(Path(profile["paths"]["arena"]) / "src"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "SWEEPING_RUN": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0",
            "PYTHON_GIL": "0" if is_free_threaded(case) else "1",
        }
    )


def process_table() -> list[tuple[int, int, int, str]]:
    """Read PID, parent PID, process group and state without inspecting arguments."""
    output = subprocess.check_output(
        ["ps", "-axo", "pid=,ppid=,pgid=,stat="], text=True
    )
    return [
        (int(pid), int(parent), int(group), state)
        for pid, parent, group, state in (line.split() for line in output.splitlines())
    ]


def stop_process_group(
    process: subprocess.Popen, groups: set[int] | None = None
) -> None:
    """Stop all owned groups: torchrun creates a separate session for each rank."""
    groups = groups or {process.pid}
    for sig, timeout in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()
            if not any(
                group in groups and not state.startswith("Z")
                for _, _, group, state in process_table()
            ):
                return
            time.sleep(0.1)
    raise RuntimeError(f"Benchmark worker groups did not stop: {sorted(groups)}")


def run_process_group(
    command: list[str], cwd: str, env: dict[str, str], log_path: Path, timeout: float
) -> int:
    """Run one case and stop every rank and worker before returning.

    Track descendant groups while their parents are alive, so orphaned torchrun
    workers can be reaped after the timed results have been written.
    """
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        groups = {process.pid}
        deadline = time.monotonic() + timeout
        try:
            while True:
                table = process_table()
                descendants = {process.pid}
                while True:
                    children = {
                        pid for pid, parent, _, _ in table if parent in descendants
                    }
                    if children <= descendants:
                        break
                    descendants.update(children)
                groups.update(group for pid, _, group, _ in table if pid in descendants)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return TIMEOUT_RETURNCODE
                try:
                    return process.wait(timeout=min(0.2, remaining))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            stop_process_group(process, groups)


def collect_sweep_result(case: dict[str, Any], attempt: Path, directory: Path) -> None:
    """Validate the one-case sweep and preserve its actual CSV and JSON observation."""
    csv_paths = list((attempt / "sweep/csv").glob("*.csv"))
    if len(csv_paths) != 1:
        raise ValueError("Expected exactly one sweep CSV")
    # Public arena owns both output paths and writes JSON beside the sweep CSV.
    payload = read_json(csv_paths[0].with_suffix(".json"))
    observations = payload.get("results", [])
    if (
        payload.get("schema_version") != 1
        or payload.get("failures")
        or len(observations) != 1
    ):
        raise ValueError("Expected one successful arena benchmark observation")
    observation = observations[0]
    row = observation["csv_row"]
    expected = {
        "loader_name": case["loader_name"],
        "run.batch_size": case["batch_size"],
        "run.num_workers": case["num_workers"],
        "run.seed": case["seed"],
        "benchmark.warmup_steps": case["warmup_batches"],
        "benchmark.measure_steps": case["measure_batches"],
        "result.steps": case["measure_batches"],
    }
    for key, value in expected.items():
        if row.get(key) != ("" if value is None else value):
            raise ValueError(f"Arena result does not match case: {key}")
    result = observation["result"]
    if (
        result["name"] != case["loader_name"]
        or result["steps"] != case["measure_batches"]
    ):
        raise ValueError("Arena result identity/window differs from the selected case")
    duration = result["total_time"]
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Arena returned invalid measured duration")
    if case.get("packing_stats") and not result.get("total_usable_tokens"):
        raise ValueError("Arena packing result has no usable tokens")
    environment = observation["environment"]
    if is_free_threaded(case) and (
        not environment["free_threaded_build"] or environment["gil_enabled"]
    ):
        raise ValueError("Arena observation does not have the required GIL disabled")
    with csv_paths[0].open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if csv_rows != [{k: "" if v is None else str(v) for k, v in row.items()}]:
        raise ValueError("Sweep CSV does not match the JSON observation")
    observation["execution"] = {
        "entrypoint": "sweep.py -> dataloader-bench",
        "artifacts": str(attempt.relative_to(directory)),
    }
    shutil.copyfile(csv_paths[0], directory / "result.csv")
    write_json(directory / "result.json", observation)


def run_case(
    case: dict[str, Any],
    profile: dict[str, Any],
    revision: str,
    directory: Path,
    timeout: float,
) -> bool:
    """Run one attempt of a case and record its status. Returns True on success."""
    status_path = directory / "status.json"
    write_json(directory / "case.json", case)
    previous_attempt = (
        read_json(status_path).get("attempt", 0) if status_path.exists() else 0
    )
    attempt = previous_attempt + 1
    # Keep whatever an unfinished attempt left behind, under that attempt's number.
    for suffix in ("json", "csv"):
        old_result = directory / f"result.{suffix}"
        if old_result.exists():
            old_result.rename(directory / f"attempt-{previous_attempt}-result.{suffix}")
    attempt_directory = directory / f"attempt-{attempt}"
    attempt_directory.mkdir(exist_ok=False)
    log_path = directory / f"attempt-{attempt}.log"

    command = sweep_command(profile, case, revision, attempt_directory)
    status = {
        "state": "running",
        "attempt": attempt,
        "started_at": utc_now(),
        "command": command,
    }
    write_json(status_path, status)
    try:
        returncode = run_process_group(
            command,
            profile["paths"]["arena"],
            sweep_environment(profile, case),
            log_path,
            timeout,
        )
    except KeyboardInterrupt:
        status.update(state="interrupted", finished_at=utc_now())
        write_json(status_path, status)
        raise

    completed = False
    if returncode == 0:
        try:
            collect_sweep_result(case, attempt_directory, directory)
            completed = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            status["validation_error"] = str(error)
    status.update(
        state="complete" if completed else "failed",
        returncode=returncode,
        finished_at=utc_now(),
    )
    write_json(status_path, status)
    if not completed:
        print(f"FAILED: {log_path}", flush=True)
    return completed


def record_campaign_inputs(
    selected: list[dict[str, Any]],
    profile: dict[str, Any],
    revision: str,
    output: Path,
) -> None:
    """Write the campaign manifest, or require unchanged inputs when resuming."""
    manifest_path = output / "campaign.json"
    if (
        manifest_path.exists()
        and read_json(manifest_path).get("kind") == "assembled-measurements"
    ):
        raise ValueError(
            "Assembled results are read-only; choose a new campaign directory"
        )
    interpreters = {environment_path(profile, case, revision) for case in selected}
    from .prepare import validate_environments, verify_assets

    environments = provenance.environment_states(interpreters)
    if profile.get("assets_receipt"):
        validate_environments(environments, profile)
    identity = {
        **({"assets": verify_assets(profile)} if profile.get("assets_receipt") else {}),
        "execution_protocol": "arena-sweep-v1",
        "cases": selected,
        "profile": profile,
        "revision": revision,
        "arena": provenance.source_state(Path(profile["paths"]["arena"])),
        "plots": provenance.source_state(Path(profile["paths"]["plots"])),
        "controller": provenance.source_state(ROOT),
        "environments": environments,
        "datasets": provenance.validate_datasets(
            profile,
            sorted({case["dataset"] for case in selected}),
            purpose=(
                "ttfb"
                if all(c.get("measurement") == "ttfb" for c in selected)
                else "throughput"
            ),
        ),
    }
    identity_sha256 = digest(identity)
    if manifest_path.exists():
        if read_json(manifest_path)["identity_sha256"] != identity_sha256:
            raise ValueError("Campaign inputs changed; use a new output directory")
        return
    write_json(
        manifest_path,
        {
            "identity_sha256": identity_sha256,
            "created_at": utc_now(),
            "host": provenance.host_snapshot(),
            **identity,
        },
    )


def execute(
    selected: list[dict[str, Any]],
    profile: dict[str, Any],
    revision: str,
    output: Path,
    timeout: float,
    keep_going: bool,
    cooldown: float,
) -> int:
    """Measure the selected cases one at a time. Returns 1 if any case failed.

    Rerunning with the same inputs skips completed cases and retries the others.
    """
    import fcntl  # POSIX only; planning and reporting do not need it.

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # One launcher per campaign directory, and one per set of benchmark
    # environments: overlapping campaigns would disturb each other's timings.
    host_lock_path = Path(
        profile.get("measurement", {}).get(
            "lock", Path(profile["paths"]["environments"]).parent / ".measurement.lock"
        )
    )
    with (
        (output / ".campaign.lock").open("a") as campaign_lock,
        host_lock_path.open("a") as host_lock,
    ):
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(host_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record_campaign_inputs(selected, profile, revision, output)

        any_failed = False
        for index, case in enumerate(selected, 1):
            progress = f"[{index}/{len(selected)}]"
            directory = output / "cases" / case["id"]
            directory.mkdir(parents=True, exist_ok=True)
            if case_state(directory) == "complete":
                if not (directory / "result.json").exists():
                    raise ValueError(f"Missing completed result: {directory}")
                print(f"{progress} skip {case['id']}", flush=True)
                continue
            print(f"{progress} run {case['id']}", flush=True)
            if not run_case(case, profile, revision, directory, timeout):
                any_failed = True
                if not keep_going:
                    return 1
            if cooldown:
                time.sleep(cooldown)
        return int(any_failed)


def status(output: Path) -> dict[str, Any]:
    """Count a campaign's cases by state and list the failed ones."""
    folders = [
        output / name
        for name in ("fetch-and-batch", "vlm-pipeline")
        if (output / name / "campaign.json").exists()
    ]
    if folders:
        states = {folder.name: status(folder) for folder in folders}
        counts = Counter()
        for state in states.values():
            counts.update(state["counts"])
        return dict(
            total=sum(state["total"] for state in states.values()),
            counts=dict(counts),
            experiments=states,
        )
    planned = read_json(output / "campaign.json")["cases"]
    counts = Counter()
    failed = []
    for case in planned:
        state = case_state(output / "cases" / case["id"])
        counts[state] += 1
        if state == "failed":
            failed.append(case["id"])
    return {"total": len(planned), "counts": dict(counts), "failed": failed}
