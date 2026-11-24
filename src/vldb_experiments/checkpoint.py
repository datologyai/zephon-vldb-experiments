"""Reproduce the paper's CPU checkpoint experiment using arena's checkpoint entrypoint."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

from . import ROOT, core


def settings() -> dict:
    return core.read_json(ROOT / "configs/checkpoint/experiment.json")


def plan(suite: str = "all", seeds=(), repetitions=()) -> list[dict]:
    spec = settings()
    points = {}

    def add(
        group,
        k,
        chunk,
        seed,
        rep,
        lanes=8,
        steps=100,
        modes=("fresh", "save", "restore"),
    ):
        point = f"k{k}-c{chunk}-s{seed}-r{rep}-l{lanes}-n{steps}"
        for mode in modes:
            name = point + "-" + mode
            case = points.setdefault(
                name,
                dict(
                    id=name,
                    point=point,
                    mode=mode,
                    k=k,
                    chunk_size=chunk,
                    seed=seed,
                    repetition=rep,
                    lanes=lanes,
                    optimizer_steps=steps,
                    analysis_tags=[],
                ),
            )
            if group not in case["analysis_tags"]:
                case["analysis_tags"].append(group)

    for rep in spec["repetitions"]:
        for k in spec["flush_intervals"]:
            for chunk in spec["chunk_sizes"]:
                for seed in spec["seeds"]:
                    add("main", k, chunk, seed, rep)
        for steps in spec["progress_steps"]:
            add("progress", 8, 1024, 42, rep, steps=steps, modes=("save", "restore"))
        for lanes in spec["lane_counts"]:
            add("lanes", 8, 1024, 42, rep, lanes=lanes)
    return [
        c
        for c in points.values()
        if (suite == "all" or suite in c["analysis_tags"])
        and (not seeds or c["seed"] in seeds)
        and (not repetitions or c["repetition"] in repetitions)
    ]


def require_s3_locations(profile: dict) -> None:
    """The paper measures cold S3 reads and S3 checkpoint/aggregation IO."""
    for name in ("dataset", "artifacts", "aggregation"):
        uri = profile.get(name)
        parsed = urlsplit(uri) if isinstance(uri, str) else None
        if (
            parsed is None
            or parsed.scheme != "s3"
            or not parsed.netloc
            or any(c in parsed.netloc for c in "@:")
            or parsed.query
            or parsed.fragment
            or any(c.isspace() for c in uri)
        ):
            raise ValueError(
                f"checkpoint.{name} must be an s3://bucket/prefix URI; "
                "local data and local checkpoint storage do not reproduce the paper protocol"
            )


def load_profile(path: Path) -> dict:
    from .profiles import load

    raw = load(path, "checkpoint")
    data = raw["checkpoint"]
    data["data_preparation"] = raw.get("checkpoint_data", {})
    require_s3_locations(data)
    for name in ("arena", "zephon", "cache", "measurement_lock"):
        data[name] = str(Path(data[name]).expanduser().resolve())
    # Keep the venv executable path: resolving its symlink selects system Python.
    data["python"] = os.path.abspath(os.path.expanduser(data["python"]))
    for name in ("tokenizer_revision", "image_processor_revision"):
        if len(data[name]) != 40 or any(
            c not in "0123456789abcdef" for c in data[name]
        ):
            raise ValueError(f"checkpoint.{name} must be an immutable asset commit")
    return data


def loader_config(case: dict, profile: dict) -> dict:
    result = copy.deepcopy(settings()["loader"])
    config = result["config"]
    config["work_source"]["chunk_size"] = case["chunk_size"]
    config["pipeline"].update(
        canonical_replicas=case["lanes"], flush_every_k_chunks=case["k"]
    )
    config["vlm"].update(
        tokenizer_revision=profile["tokenizer_revision"],
        image_processor_revision=profile["image_processor_revision"],
    )
    return {"loaders": [result]}


def command(case: dict, profile: dict, attempt: Path, campaign_id: str) -> list[str]:
    """Arena measures; this repository supplies paper settings and independent processes."""
    require_s3_locations(profile)
    core.write_json(attempt / "loaders.json", loader_config(case, profile))
    base = profile["artifacts"].rstrip("/") + "/" + campaign_id
    return [
        profile["python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(settings()["world_size"]),
        "--module",
        "--",  # Keep torchrun from interpreting benchmark options such as --local.
        "dataloader_arena.checkpoint",
        "--mode",
        case["mode"],
        "--config",
        str(attempt / "loaders.json"),
        "--local",
        profile["cache"],
        "--remote",
        profile["dataset"],
        "--cache-namespace",
        campaign_id + "-" + case["id"] + "-" + attempt.name,
        "--cache-policy",
        "cold",
        "--cache-scope",
        settings()["cache_scope"],
        "--seed",
        str(case["seed"]),
        "--batch-size",
        str(settings()["micro_batch_size"]),
        "--global-batch-size",
        str(settings()["global_batch_size"]),
        "--optimizer-steps",
        str(case["optimizer_steps"]),
        "--aggregate-dir",
        profile["aggregation"].rstrip("/") + "/" + campaign_id,
        "--checkpoint-uri",
        base + "/" + case["point"] + ".pkl",
        "--verify-batches",
        str(settings()["verify_batches"]),
        "--json-output",
        str(attempt / "result.json"),
    ]


def source_digest(profile: dict) -> dict:
    """Detect changes to frozen code and installed package identities between cases."""
    arena = Path(profile["arena"])
    files = list((arena / "src").rglob("*.py")) + [
        arena / "pyproject.toml",
        arena / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(arena)).encode() + b"\0" + path.read_bytes())
    script = """
import hashlib
import importlib.metadata as metadata
import json
from contextlib import redirect_stdout
from pathlib import Path
import zephon
import dataloader_arena
import platform

root = Path(zephon.__file__).parent
source = hashlib.sha256()
for path in sorted(root.rglob("*.py")):
    source.update(str(path.relative_to(root)).encode())
    source.update(path.read_bytes())
packages = sorted(
    [(d.metadata["Name"] or "", d.version, d.read_text("direct_url.json"))
     for d in metadata.distributions()],
    key=lambda package: (package[0], package[1], package[2] or ""),
)
print(json.dumps({
    "python_version": platform.python_version(),
    "packages": packages,
    "zephon_version": metadata.version("zephon"),
    "litdata_version": metadata.version("litdata"),
    "zephon_sha256": source.hexdigest(),
    "zephon_path": str(Path(zephon.__file__).resolve()),
    "arena_path": str(Path(dataloader_arena.__file__).resolve()),
}))
"""
    frozen = json.loads(
        subprocess.check_output([profile["python"], "-c", script], text=True)
    )
    for name in ("arena", "zephon"):
        if name in profile and not Path(frozen[name + "_path"]).is_relative_to(
            Path(profile[name]).resolve()
        ):
            raise ValueError(
                f"Installed {name} does not use the selected source: {profile[name]}"
            )
    controller = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        controller.update(str(path.relative_to(Path(__file__).parent)).encode())
        controller.update(path.read_bytes())
    expected = settings()["sources"]
    if frozen["python_version"] != expected["python"]:
        raise ValueError(
            f"Expected Python {expected['python']}, got {frozen['python_version']}"
        )
    for name in ("litdata_version",):
        if frozen[name] != expected[name]:
            raise ValueError(f"Expected {name}={expected[name]}, got {frozen[name]}")
    from .provenance import source_state

    revision = source_state(arena)["revision"]
    from .prepare import verify_assets

    return dict(
        **(
            {"assets_sha256": core.digest(verify_assets(profile))}
            if profile.get("assets_receipt")
            else {}
        ),
        arena_sha256=digest.hexdigest(),
        packages_sha256=core.digest(frozen),
        arena_revision=revision,
        zephon_version=frozen["zephon_version"],
        zephon_sha256=frozen["zephon_sha256"],
        controller_sha256=controller.hexdigest(),
    )


def counts(output: Path) -> dict:
    campaign = core.read_json(output / "campaign.json")
    states = [core.case_state(output / "cases" / c["id"]) for c in campaign["cases"]]
    return {state: states.count(state) for state in sorted(set(states))}


def validate_result(case: dict, result: dict) -> None:
    if (
        result["mode"] != case["mode"]
        or result["world_size"] != settings()["world_size"]
    ):
        raise ValueError("Result mode or rank count differs from the planned case")
    ranks = result["ranks"]
    if len(ranks) != settings()["world_size"] or {r["rank"] for r in ranks} != set(
        range(settings()["world_size"])
    ):
        raise ValueError("Missing or duplicate ranks")
    if case["mode"] == "restore" and not all(
        r["continuation"]["matches"] for r in ranks
    ):
        raise ValueError(
            "Restored continuation differs from the uninterrupted pipeline"
        )
    if case["mode"] == "save":
        expected = (
            case["optimizer_steps"]
            * settings()["global_batch_size"]
            // (settings()["micro_batch_size"] * settings()["world_size"])
        )
        if not all(r["batches_before_save_per_rank"] == expected for r in ranks):
            raise ValueError("Saved at the wrong workload position")
        if not all(
            len(r["continuation_hashes"]) == settings()["verify_batches"] for r in ranks
        ):
            raise ValueError("Missing uninterrupted continuation hashes")


def run(
    profile: dict, cases: list[dict], output: Path, *, timeout=7200, retry_failed=False
) -> int:
    require_s3_locations(profile)
    if not cases:
        raise ValueError("Selection contains no checkpoint cases")
    output = output.resolve()
    if any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        for c in output.name
    ):
        raise ValueError("Campaign directory name must be a simple path component")
    output.mkdir(parents=True, exist_ok=True)
    lock = Path(profile["measurement_lock"])
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another benchmark owns the measurement lock") from None
        from . import checkpoint_data

        input_validation = checkpoint_data.execute(profile, check=True)
        core.write_json(output / "input-validation.json", input_validation)
        fingerprint = source_digest(profile)
        frozen = dict(
            cases=cases,
            profile=profile,
            source=fingerprint,
            settings_sha256=core.digest(settings()),
            input_index_sha256=input_validation["index_sha256"],
        )
        manifest_path = output / "campaign.json"
        if manifest_path.exists():
            campaign = core.read_json(manifest_path)
            if campaign["frozen"] != frozen:
                raise ValueError(
                    "Frozen campaign differs; use a separate output directory"
                )
        else:
            campaign = dict(
                id=output.name, started_at=core.utc_now(), cases=cases, frozen=frozen
            )
            core.write_json(manifest_path, campaign)
        for index, case in enumerate(cases):
            directory = output / "cases" / case["id"]
            state = core.case_state(directory)
            if state == "complete":
                validate_result(case, core.read_json(directory / "result.json"))
                continue
            if state != "pending" and not retry_failed:
                raise RuntimeError(
                    f"Inspect {directory} before resuming with --retry-failed"
                )
            if source_digest(profile) != fingerprint:
                raise RuntimeError("Frozen sources or environment changed")
            directory.mkdir(parents=True, exist_ok=True)
            attempt = (
                directory
                / "attempts"
                / f"{len(list((directory / 'attempts').glob('*'))) + 1:03d}"
            )
            attempt.mkdir(parents=True)
            argv = command(case, profile, attempt, campaign["id"])
            core.write_json(attempt / "command.json", argv)
            core.write_json(
                directory / "status.json",
                dict(state="running", started_at=core.utc_now(), attempt=str(attempt)),
            )
            core.write_json(
                output / "progress.json",
                dict(
                    current=case["id"],
                    completed=index,
                    total=len(cases),
                    state="measuring",
                    updated_at=core.utc_now(),
                ),
            )
            env = (
                os.environ
                | {
                    "PYTHONPATH": str(Path(profile["arena"]) / "src"),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "PYTHONHASHSEED": "0",
                    "PYTHONUNBUFFERED": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                }
                | profile.get("env", {})
            )
            print(f"[{index + 1}/{len(cases)}] {case['id']}", flush=True)
            try:
                code = core.run_process_group(
                    argv, profile["arena"], env, attempt / "run.log", timeout
                )
                if code:
                    raise RuntimeError(
                        f"Arena exited {code}; inspect {attempt / 'run.log'}"
                    )
                result = core.read_json(attempt / "result.json")
                validate_result(case, result)
                if any(
                    r["provenance"]["dataset_index_sha256"]
                    != input_validation["index_sha256"]
                    for r in result["ranks"]
                ):
                    raise ValueError(
                        "Measured dataset differs from the validated campaign input"
                    )
                if source_digest(profile) != fingerprint:
                    raise RuntimeError(
                        "Frozen sources or environment changed during measurement"
                    )
                core.write_json(directory / "result.json", result)
                core.write_json(
                    directory / "status.json",
                    dict(
                        state="complete",
                        finished_at=core.utc_now(),
                        attempt=str(attempt),
                    ),
                )
            except BaseException as error:
                core.write_json(
                    directory / "status.json",
                    dict(
                        state="failed",
                        finished_at=core.utc_now(),
                        error=str(error),
                        attempt=str(attempt),
                    ),
                )
                raise
            # Cache cleanup cannot invalidate an already saved, verified observation.
            try:
                cleanup_cache(profile, argv, result)
            except OSError as error:
                core.write_json(
                    directory / "cache-cleanup-error.json", {"error": str(error)}
                )
                raise
            time.sleep(2)
        core.write_json(
            output / "progress.json",
            dict(
                state="complete",
                completed=len(cases),
                total=len(cases),
                finished_at=core.utc_now(),
            ),
        )
    return 0


def cleanup_cache(profile: dict, argv: list[str], result: dict) -> None:
    """Release only this successful invocation's isolated cache after recording residency."""
    namespace = argv[argv.index("--cache-namespace") + 1]
    root = Path(profile["cache"]).resolve() / ".arena-checkpoint-cache" / namespace
    for rank in result["ranks"]:
        recorded = Path(rank["cache"]["path"]).resolve()
        scope = (
            argv[argv.index("--cache-scope") + 1] if "--cache-scope" in argv else "rank"
        )
        leaf = "host" if scope == "host" else f"rank-{rank['rank']}"
        if recorded != root / leaf:
            raise ValueError("Result names a cache outside this invocation")

    def ignore_disappearing_file(_function, _path, error_info):
        # A cancelled download can remove its .part file during the directory walk.
        if not isinstance(error_info[1], FileNotFoundError):
            raise error_info[1]

    shutil.rmtree(root, onerror=ignore_disappearing_file)
