"""Run the paper's startup experiment with arena's existing TTFB sweep command.

Each sweep keeps its configurations in one Python process, as the archived sweeps
did. Individual fresh processes would repeatedly charge first-use imports.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from . import ROOT, core, matrix, provenance, ttfb_data

SCENARIOS = ("fetch-text", "fetch-vlm", "vlm-pipeline")


def settings() -> dict:
    return core.read_json(ROOT / "configs/ttfb/experiment.json")


def plan(
    revision: str = "v0.1.0", *, scenarios=(), systems=(), dops=(), seeds=()
) -> list[dict]:
    spec = settings()
    values = []
    for experiment in ("fetch-and-batch", "vlm-pipeline"):
        for case in matrix.plan(experiment, revision):
            if case["variant"] != "baseline":
                continue
            scenario = (
                "fetch-" + case["dataset"]
                if experiment == "fetch-and-batch"
                else experiment
            )
            expected_dops = spec[
                "fetch_dops" if experiment == "fetch-and-batch" else "vlm_dops"
            ]
            if case["dop"] not in expected_dops or case["seed"] not in spec["seeds"]:
                continue
            if (scenarios and scenario not in scenarios) or (
                systems and case["system"] not in systems
            ):
                continue
            if (dops and case["dop"] not in dops) or (
                seeds and case["seed"] not in seeds
            ):
                continue
            case.update(
                measurement="ttfb", scenario=scenario, repetitions=spec["repetitions"]
            )
            case["id"] += "-ttfb"
            case["batch_size"] = spec[
                "fetch_batch_size"
                if experiment == "fetch-and-batch"
                else "vlm_batch_size"
            ]
            for key in ("warmup_batches", "measure_batches", "packing_stats", "panels"):
                case.pop(key, None)
            values.append(case)
    if not values:
        raise ValueError("Selection contains no TTFB cases")
    return values


def jobs(cases: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for case in cases:
        family = "zephon" if case["system"] == "zephon" else "baselines"
        grouped[f"{case['scenario']}-{case['python']}-{family}"].append(case)
    return dict(sorted(grouped.items()))


def sweep_command(
    cases: list[dict], profile: dict, revision: str, attempt: Path
) -> list[str]:
    """Group by scenario, Python build and loader family for arena sweep.py."""
    first = cases[0]
    python = core.environment_path(profile, first, revision)
    loaders = {}
    for case in cases:
        loader = dict(
            name=case["loader_name"], factory=case["factory"], config=case["config"]
        )
        previous = loaders.setdefault(case["loader_name"], loader)
        if previous != loader:
            raise ValueError("Conflicting TTFB loader configuration")
    core.write_json(attempt / "benchmark.json", {"loaders": list(loaders.values())})
    arguments = [
        "--batch-size",
        str(first["batch_size"]),
        "--repetitions",
        str(first["repetitions"]),
        "--strict",
        "--verbose",
    ]
    for workers in sorted({c["num_workers"] for c in cases}):
        arguments += ["--num-workers", str(workers)]
    for seed in sorted({c["seed"] for c in cases}):
        arguments += ["--seed", str(seed)]
    if core.is_free_threaded(first):
        arguments.append("--require-no-gil")
    core.write_json(
        attempt / "sweep.json",
        dict(
            python_versions=[first["python"]],
            runs=[
                dict(name=first["scenario"], local=profile["paths"][first["dataset"]])
            ],
            bench_config=str(attempt / "benchmark.json"),
            bench_command="dataloader-ttfb",
            bench_args=arguments,
            preinstall=False,
            force_gil_off=core.is_free_threaded(first),
            uv_project_environment_template=str(python.parent.parent),
        ),
    )
    return [
        str(python),
        str(Path(profile["paths"]["arena"]) / "sweep.py"),
        "--config",
        str(attempt / "sweep.json"),
        "--results-dir",
        str(attempt / "sweep"),
        "--no-sync",
        "--show-live",
    ]


def observation_key(row: dict) -> tuple:
    return (
        row["loader_name"],
        int(row["run.batch_size"]),
        int(row["run.num_workers"]),
        int(row["run.seed"]),
    )


def case_key(case: dict) -> tuple:
    return (case["loader_name"], case["batch_size"], case["num_workers"], case["seed"])


def require_cold_cache(observation: dict, repetitions: int) -> None:
    drops = observation.get("page_cache_dropped")
    if not isinstance(drops, list) or len(drops) != repetitions:
        raise ValueError(
            "TTFB page-cache drop confirmation is missing or incomplete. "
            "Rerun the sweep with the current Arena and passwordless sudo on the benchmark machine."
        )
    if any(dropped is not True for dropped in drops):
        raise ValueError(
            "TTFB requires a successful page-cache drop before every trial. "
            "Check passwordless sudo on the benchmark machine and rerun the sweep. "
            "The raw measurements have been kept."
        )


def validate_observation(case: dict, row: dict, observation: dict) -> None:
    result = observation["result"]
    identity = (
        result["name"],
        result["batch_size"],
        result["num_workers"],
        result["seed"],
    )
    if identity != case_key(case) or observation_key(row) != case_key(case):
        raise ValueError("TTFB observation does not match the selected case")
    if (
        int(row["benchmark.repetitions"]) != case["repetitions"]
        or result["repetitions"] != case["repetitions"]
    ):
        raise ValueError("TTFB repetition count changed")
    if json.loads(row["config.plaintext"]) != case["config"]:
        raise ValueError("Arena changed the requested TTFB loader configuration")
    trials = result["trials"]
    if len(trials) != case["repetitions"]:
        raise ValueError("Missing individual TTFB trials")
    for trial in trials:
        if (
            any(not math.isfinite(v) or v < 0 for v in trial.values())
            or trial["total_ttfb"] <= 0
        ):
            raise ValueError("Invalid TTFB trial timing")
        if not math.isclose(
            trial["factory_time"] + trial["iter_to_batch"],
            trial["total_ttfb"],
            abs_tol=1e-7,
        ):
            raise ValueError("TTFB phase timings do not sum to the total")
        if not math.isclose(
            trial["iter_time"] + trial["first_next_time"],
            trial["iter_to_batch"],
            abs_tol=1e-7,
        ):
            raise ValueError(
                "Iterator timings do not sum to the iterator-to-batch span"
            )
    if not math.isclose(
        result["ttfb_median"],
        statistics.median(t["total_ttfb"] for t in trials),
        abs_tol=1e-7,
    ):
        raise ValueError("TTFB result differs from its individual trials")
    for metric in (
        "ttfb_median",
        "factory_time_median",
        "iter_time_median",
        "first_next_time_median",
        "iter_to_batch_median",
    ):
        if not math.isclose(
            float(row[f"result.{metric}_s"]), result[metric], abs_tol=1e-7
        ):
            raise ValueError("TTFB CSV and JSON timings differ")
    if observation["cache_policy"] != settings()["cache_policy"]:
        raise ValueError("TTFB cache policy differs from the paper protocol")
    require_cold_cache(observation, case["repetitions"])
    environment = observation["environment"]
    if bool(environment["gil_enabled"]) == core.is_free_threaded(case):
        raise ValueError("TTFB observation has the wrong GIL state")
    if core.is_free_threaded(case) and not environment["free_threaded_build"]:
        raise ValueError("TTFB observation requires a free-threaded interpreter")


def collect(cases: list[dict], attempt: Path, output: Path) -> None:
    csv_paths = [
        p
        for p in (attempt / "sweep/csv").glob("*.csv")
        if not p.name.endswith("-trials.csv")
    ]
    if len(csv_paths) != 1:
        raise ValueError("Expected exactly one TTFB sweep CSV")
    csv_path = csv_paths[0]
    payload = core.read_json(csv_path.with_suffix(".json"))
    if (
        payload.get("schema_version") != 1
        or payload.get("metric") != "ttfb"
        or payload.get("failures")
    ):
        raise ValueError("TTFB sweep did not finish successfully")
    with csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    observations = payload["results"]
    if len(rows) != len(cases) or len(observations) != len(cases):
        raise ValueError("TTFB sweep observation count differs from the planned matrix")
    expected = {case_key(c): c for c in cases}
    if len(expected) != len(cases) or len({observation_key(r) for r in rows}) != len(
        cases
    ):
        raise ValueError("Duplicate TTFB cases or observations")
    trial_path = csv_path.with_name(csv_path.stem + "-trials.csv")
    with trial_path.open(newline="") as file:
        trial_rows = list(csv.DictReader(file))
    if len(trial_rows) != sum(c["repetitions"] for c in cases):
        raise ValueError("Missing per-trial TTFB CSV rows")
    trial_lookup = {}
    for row in trial_rows:
        key = (*observation_key(row), int(row["trial"]))
        if key in trial_lookup:
            raise ValueError("Duplicate per-trial TTFB CSV row")
        trial_lookup[key] = row
    validated = []
    for row, observation in zip(rows, observations, strict=True):
        case = expected[observation_key(row)]
        validate_observation(case, row, observation)
        for number, trial in enumerate(observation["result"]["trials"], start=1):
            recorded = trial_lookup[(*case_key(case), number)]
            if any(
                not math.isclose(
                    float(recorded[f"result.{key}_s"]), value, abs_tol=1e-7
                )
                for key, value in trial.items()
            ):
                raise ValueError("TTFB trial CSV and JSON differ")
        validated.append((case, row, observation))
    # Validate the whole sweep before promoting any rows. Partial attempts remain in jobs/.
    for case, row, observation in validated:
        directory = output / "cases" / case["id"]
        directory.mkdir(parents=True, exist_ok=True)
        core.write_json(directory / "case.json", case)
        core.write_json(
            directory / "result.json",
            dict(
                **observation,
                csv_row=row,
                execution=dict(
                    entrypoint="sweep.py -> dataloader-ttfb",
                    artifacts=str(attempt.relative_to(output)),
                ),
            ),
        )
        with (directory / "result.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        core.write_json(
            directory / "status.json",
            dict(state="complete", finished_at=core.utc_now()),
        )


def run(
    cases: list[dict], profile: dict, revision: str, output: Path, timeout: float
) -> int:
    import fcntl

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_path = (
        Path(
            profile.get("measurement", {}).get(
                "lock",
                Path(profile["paths"]["environments"]).parent / ".measurement.lock",
            )
        )
        .expanduser()
        .resolve()
    )
    with (
        (output / ".campaign.lock").open("a") as campaign_lock,
        lock_path.open("a") as host_lock,
    ):
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(host_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        datasets = sorted({c["dataset"] for c in cases})
        preflight = ttfb_data.check(profile, datasets)
        core.write_json(output / "input-preflight.json", preflight)
        ttfb_data.require_valid(preflight)
        core.record_campaign_inputs(cases, profile, revision, output)
        hashes = provenance.validate_datasets(
            profile, datasets, hashes=True, purpose="ttfb"
        )
        previous_hashes = output / "input-hashes-before.json"
        if previous_hashes.exists() and core.read_json(previous_hashes) != hashes:
            raise ValueError("TTFB input contents changed since the previous attempt")
        core.write_json(previous_hashes, hashes)
        for name, selected in jobs(cases).items():
            directory = output / "jobs" / name
            directory.mkdir(parents=True, exist_ok=True)
            status_path = directory / "status.json"
            if core.case_state(directory) == "complete":
                if any(
                    core.case_state(output / "cases" / c["id"]) != "complete"
                    or not (output / "cases" / c["id"] / "result.json").is_file()
                    for c in selected
                ):
                    raise ValueError("Missing observations from completed TTFB sweep")
                for case in selected:
                    require_cold_cache(
                        core.read_json(output / "cases" / case["id"] / "result.json"),
                        case["repetitions"],
                    )
                print("skip", name, flush=True)
                continue
            attempt_number = (
                core.read_json(status_path)["attempt"] + 1
                if status_path.exists()
                else 1
            )
            attempt = directory / f"attempt-{attempt_number}"
            attempt.mkdir(exist_ok=False)
            command = sweep_command(selected, profile, revision, attempt)
            status = dict(
                state="running",
                attempt=attempt_number,
                started_at=core.utc_now(),
                command=command,
            )
            core.write_json(status_path, status)
            print("run", name, f"({len(selected)} observations)", flush=True)
            try:
                code = core.run_process_group(
                    command,
                    profile["paths"]["arena"],
                    core.sweep_environment(profile, selected[0]),
                    directory / f"attempt-{attempt_number}.log",
                    timeout,
                )
                if code:
                    raise RuntimeError(f"arena sweep returned {code}")
                collect(selected, attempt, output)
            except (KeyboardInterrupt, Exception) as error:
                status.update(
                    state="interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "failed",
                    error=str(error),
                    finished_at=core.utc_now(),
                )
                core.write_json(status_path, status)
                raise
            status.update(state="complete", finished_at=core.utc_now())
            core.write_json(status_path, status)
        after = provenance.validate_datasets(
            profile, datasets, hashes=True, purpose="ttfb"
        )
        core.write_json(output / "input-hashes-after.json", after)
        if after != hashes:
            raise ValueError("TTFB input contents changed during measurement")
        core.record_campaign_inputs(cases, profile, revision, output)
    return 0
