"""Turn the checked-in paper configs into the list of benchmark cases.

``cases`` expands one experiment into every case, in execution order.
``plan`` applies a command-line selection and the chosen Zephon revision.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Sequence
from copy import deepcopy
from itertools import product
from typing import Any

from . import ROOT

CONFIGS = ROOT / "configs"

# Arena loader names of the worker-based baselines in Figure 6.
BASELINE_LOADER_NAMES = {
    "grain": "grain-litdata-shuffled",
    "litdata": "litdata-shuffled",
}


def cases(experiment: str) -> list[dict[str, Any]]:
    """Every case of one experiment, in execution order. Each call returns new dicts."""
    spec = json.loads((CONFIGS / "throughput" / f"{experiment}.json").read_text())
    if experiment == "fetch-and-batch":
        values = fetch_and_batch(spec)
    elif experiment == "vlm-pipeline":
        values = vlm_pipeline(spec)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")
    values.sort(
        key=lambda c: (c["seed"], c["dataset"], c["dop"], c["python"], c["loader_name"])
    )
    return values


def make_case(
    *,
    experiment,
    dataset,
    system,
    python,
    dop,
    name,
    config,
    seed,
    batch,
    warmup,
    measure,
    panels,
    variant="baseline",
):
    """One benchmark run: a loader config, its seed and its measurement window."""
    family = "load_and_batch" if experiment == "fetch-and-batch" else "vlm"
    return {
        "id": f"{dataset}-{python}-{name}-{variant}-d{dop}-s{seed}",
        "experiment": experiment,
        "dataset": dataset,
        "system": system,
        "python": python,
        "dop": dop,
        "loader_name": name,
        "factory": f"dataloader_arena.benchmarks.{family}.{system}:create_loader",
        "config": config,
        "seed": seed,
        "batch_size": batch,
        "num_workers": 0 if system == "zephon" else dop,
        "warmup_batches": warmup,
        "measure_batches": measure,
        "panels": panels,
        "variant": variant,
        "packing_stats": experiment == "vlm-pipeline",
    }


def fetch_and_batch(spec: dict) -> list[dict]:
    """Figure 6: every seed, dataset and DOP for each loader variant."""
    values = []
    for seed, dataset, dop, variant in product(
        spec["seeds"], spec["datasets"], spec["dops"], spec["variants"]
    ):
        system = variant["system"]
        config = deepcopy(spec["configs"][system])
        if system == "zephon":
            config.update(fetch_parallelism=dop, runner=variant["runner"])
            runner = "processes" if variant["runner"] == "process" else "threads"
            name = f"zephon-cs2048-shuffled-{runner}-fdop{dop}"
            if variant["variant"] == "noshm":
                # The pipe-only condition of the shared-memory panel.
                config["pipeline"]["coalesce_tensors"] = False
                name += "-noshm"
        else:
            name = BASELINE_LOADER_NAMES[system]
        measure = spec["measure_batches"][dataset]
        values.append(
            make_case(
                experiment="fetch-and-batch",
                dataset=dataset,
                system=system,
                python=variant["python"],
                dop=dop,
                name=name,
                config=config,
                seed=seed,
                batch=spec["batch_size"],
                warmup=spec["warmup_batches"],
                measure=measure,
                panels=variant["panels"],
                variant=variant.get("variant", "baseline"),
            )
        )
    return values


def vlm_pipeline(spec: dict) -> list[dict]:
    """Figure 7: every Zephon stage allocation, and each baseline at every DOP."""
    values = []
    for seed, loader in product(spec["seeds"], spec["loaders"]):
        system = loader["system"]
        if system == "zephon":
            allocations = spec["allocations"]
        else:
            allocations = [{"name": loader["name"], "dop": dop} for dop in spec["dops"]]
        for allocation in allocations:
            config = deepcopy(spec["configs"][system])
            if system == "zephon":
                config["fetch_parallelism"] = allocation["fetch"]
                config["pipeline"]["prefetch_batches"] = allocation["prefetch_batches"]
                options = config["vlm_zephon"]
                options["fused_parallelism"] = allocation["fused"]
                # Omitted stages keep the pinned arena defaults.
                if "pack" in allocation:
                    options["shuffle_pack_parallelism"] = allocation["pack"]
                if "batch" in allocation:
                    options["batch_construct_parallelism"] = allocation["batch"]
            values.append(
                make_case(
                    experiment="vlm-pipeline",
                    dataset=spec["dataset"],
                    system=system,
                    python=loader["python"],
                    dop=allocation["dop"],
                    name=allocation["name"],
                    config=config,
                    seed=seed,
                    batch=spec["batch_size"],
                    warmup=loader["warmup_batches"],
                    measure=loader["measure_batches"],
                    panels=["overview"],
                )
            )
    return values


def revision_pin(revision: str) -> dict[str, Any]:
    """The commit pinned for a Zephon revision label."""
    pins = tomllib.loads((CONFIGS / "shared/revisions.toml").read_text())
    if revision not in pins:
        raise ValueError(
            f"Unknown revision {revision!r}; add a pin in configs/shared/revisions.toml"
        )
    return pins[revision]


def plan(
    experiment: str,
    revision: str,
    *,
    panel: str = "all",
    systems: Sequence[str] = (),
    runner: str | None = None,
    seeds: Sequence[int] = (),
    dops: Sequence[int] = (),
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Select cases for one Zephon revision. An empty filter keeps every value."""
    revision_pin(revision)
    selected = []
    for case in cases(experiment):
        if panel != "all" and panel not in case["panels"]:
            continue
        if systems and case["system"] not in systems:
            continue
        if runner and case["config"].get("runner", "workers") != runner:
            continue
        if seeds and case["seed"] not in seeds:
            continue
        if dops and case["dop"] not in dops:
            continue
        if smoke:
            # Short windows for checking a setup; their IDs keep them out of reports.
            case["warmup_batches"] = 2
            case["measure_batches"] = 8
            case["id"] += "-smoke"
        if case["system"] == "zephon":
            case["revision_profile"] = revision
        else:
            case["revision_profile"] = "baseline"
        selected.append(case)
    if not selected:
        raise ValueError("Selection contains no cases")
    return selected


def is_full_matrix(experiment: str, selected: list[dict[str, Any]]) -> bool:
    """Whether a selection is the whole paper experiment, without smoke windows."""
    return {case["id"] for case in selected} == {
        case["id"] for case in cases(experiment)
    }
