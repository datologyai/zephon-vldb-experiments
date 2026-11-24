"""Results discussed in §8.1.3, with its seed aggregation and VLM allocation.

Aggregate per-seed TTFB medians by arithmetic mean and sample standard deviation.
Full VLM measurements use the checked-in DOP64 paper allocation.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from . import core
from .presentation import document, table, value
from .ttfb import require_cold_cache, settings

TIMINGS = {
    "ttfb": "result.ttfb_median_s",
    "factory": "result.factory_time_median_s",
    "iter": "result.iter_time_median_s",
    "first_next": "result.first_next_time_median_s",
    "iter_to_batch": "result.iter_to_batch_median_s",
}
TABLE_SYSTEMS = (
    ("Grain", "grain", "workers", "3.13"),
    ("LitData", "litdata", "workers", "3.13"),
    ("Streaming", "streaming", "workers", "3.13"),
    ("Zephon processes (GIL)", "zephon", "process", "3.13"),
    ("Zephon processes (no GIL)", "zephon", "process", "3.14t"),
    ("Zephon threads (GIL)", "zephon", "threads", "3.13"),
    ("Zephon threads (no GIL)", "zephon", "threads", "3.14t"),
)


def parse_loader(name: str, workers: int, scenario: str) -> tuple:
    if scenario == "vlm-pipeline":
        match = re.fullmatch(r"zephon-fused-dop(\d+)-(.+)", name)
        if match:
            return "zephon", "process", int(match[1]), match[2]
        return (
            {"grain-shuffled": "grain"}[name],
            "workers",
            workers,
            "default",
        )
    match = re.fullmatch(
        r"zephon-cs\d+-(shuffled|unshuffled)-(processes|threads)-fdop(\d+)", name
    )
    if match:
        return (
            "zephon",
            {"processes": "process", "threads": "threads"}[match[2]],
            int(match[3]),
            match[1],
        )
    match = re.fullmatch(r"(grain-litdata|litdata)-(shuffled|unshuffled)", name)
    if not match:
        raise ValueError("Unrecognized TTFB loader: " + name)
    return (
        {"grain-litdata": "grain", "litdata": "litdata"}[match[1]],
        "workers",
        workers,
        match[2],
    )


def observations(source: Path) -> tuple[list[dict], dict]:
    campaign = core.read_json(source / "campaign.json")
    rows = []
    for case in campaign["cases"]:
        if case.get("measurement") != "ttfb":
            raise ValueError("This is not a TTFB campaign")
        folder = source / "cases" / case["id"]
        if core.case_state(folder) != "complete":
            raise ValueError(
                "TTFB reporting requires every selected case to be complete"
            )
        result = core.read_json(folder / "result.json")
        require_cold_cache(result, case["repetitions"])
        rows.append(
            dict(
                result["csv_row"],
                sweep_python=case["python"],
                scenario=case["scenario"],
                source="campaign",
            )
        )
    return rows, dict(datasets=campaign["datasets"], profile=campaign["profile"])


def aggregate(rows: list[dict], *, allow_partial: bool = False) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        system, runner, dop, config = parse_loader(
            row["loader_name"], int(row["run.num_workers"]), row["scenario"]
        )
        key = (
            row["scenario"],
            system,
            runner,
            row["sweep_python"],
            dop,
            int(row["run.batch_size"]),
            config,
            row["loader_name"],
            row["source"],
        )
        grouped[key].append(row)
    points = []
    for key, values in sorted(grouped.items()):
        seeds = [int(v["run.seed"]) for v in values]
        if len(seeds) != len(set(seeds)) or (
            not allow_partial and set(seeds) != set(settings()["seeds"])
        ):
            raise ValueError(
                "Expected exactly one observation for each of the three paper seeds: "
                + str(key)
            )
        if any(int(v["benchmark.repetitions"]) != 1 for v in values):
            raise ValueError("Paper TTFB uses one repetition per seed")
        point = dict(
            zip(
                (
                    "scenario",
                    "system",
                    "runner",
                    "python",
                    "dop",
                    "batch_size",
                    "config",
                    "loader_name",
                    "source",
                ),
                key,
            )
        )
        point["n_seeds"] = len(seeds)
        for metric, column in TIMINGS.items():
            numbers = [float(v[column]) for v in values]
            if any(not math.isfinite(v) or v < 0 for v in numbers):
                raise ValueError("Invalid TTFB metric")
            point[metric + "_s"] = statistics.mean(numbers)
            point[metric + "_std_s"] = (
                statistics.stdev(numbers) if len(numbers) > 1 else 0.0
            )
        points.append(point)
    return points


def report(source: Path, *, allow_partial: bool = False) -> dict:
    source = source.expanduser().resolve()
    raw, metadata = observations(source)
    points = aggregate(raw, allow_partial=allow_partial)
    selected = settings()["paper_vlm_selection"]
    table = []

    def add(label, scenario, system, runner, python, loader=None):
        candidates = [
            p
            for p in points
            if p["scenario"] == scenario
            and p["system"] == system
            and p["runner"] == runner
            and p["python"] == python
            and p["dop"] == 64
            and p["batch_size"] == (4 if scenario == "vlm-pipeline" else 32)
            and (scenario == "vlm-pipeline" or p["config"] == "shuffled")
            and (loader is None or p["loader_name"] == loader)
        ]
        if len(candidates) > 1:
            raise ValueError("Ambiguous TTFB table cell")
        chosen = candidates[0] if candidates else None
        reason = ""
        if chosen is None:
            reason = (
                "Private paper implementation absent from public arena"
                if system in ("streaming", "internalds")
                else "Not measured in the supplied inputs"
            )
        table.append(
            dict(
                metric=f"{scenario}: {label}",
                scenario=scenario,
                system=label,
                ttfb_s=chosen["ttfb_s"] if chosen else None,
                std_s=chosen["ttfb_std_s"] if chosen else None,
                n_seeds=chosen["n_seeds"] if chosen else 0,
                source=chosen["source"] if chosen else "",
                loader_name=chosen["loader_name"] if chosen else "",
                unavailable_reason=reason,
            )
        )

    for label, system, runner, python in TABLE_SYSTEMS:
        for scenario in ("fetch-text", "fetch-vlm"):
            add(label, scenario, system, runner, python)
    add("Grain", "vlm-pipeline", "grain", "workers", "3.13")
    add("InternalDs", "vlm-pipeline", "internalds", "workers", "3.12")
    add(
        "Zephon processes (GIL)",
        "vlm-pipeline",
        "zephon",
        "process",
        "3.13",
        selected["loader_name"],
    )
    if not allow_partial and not any(
        p["loader_name"] == selected["loader_name"] and p["scenario"] == "vlm-pipeline"
        for p in points
    ):
        raise ValueError("The paper VLM allocation has no TTFB measurements")
    prose = [
        dict(
            metric=r["metric"],
            value=r["ttfb_s"],
            unit="seconds",
            unavailable_reason=r["unavailable_reason"],
        )
        for r in table
    ]
    by_metric = {r["metric"]: r["ttfb_s"] for r in table}
    z = by_metric["vlm-pipeline: Zephon processes (GIL)"]
    for label, reference, percentage in [
        ("VLM startup reduction versus Grain", "Grain", True),
        ("VLM startup speedup versus InternalDs", "InternalDs", False),
    ]:
        baseline = by_metric["vlm-pipeline: " + reference]
        value = (
            ((baseline - z) / baseline * 100 if percentage else baseline / z)
            if baseline and z
            else None
        )
        prose.append(
            dict(
                metric=label,
                value=value,
                unit="percent" if percentage else "times",
                unavailable_reason=""
                if value is not None
                else "Required baseline not measured",
            )
        )
    kind = metadata["profile"].get("ttfb_data", {}).get("source", "public")
    sizes = ", ".join(
        f"{name}: {dataset['samples']:,} samples in {len(dataset['files']):,} shards"
        for name, dataset in metadata["datasets"].items()
    )
    input_note = f"Inputs: {kind}; {sizes}."
    input_note += " Regenerated public records differ from the original paper inputs; the counts above identify this startup workload."
    summary = dict(
        observations=len(raw),
        points=len(points),
        metric_count=len(prose),
        measured_metrics=sum(r["value"] is not None for r in prose),
        vlm_selection=selected,
        aggregation="mean of per-seed medians; sample standard deviation across three seeds",
        protocol="factory -> iter -> first next; one repetition per seed",
        input_note=input_note,
        **metadata,
    )
    summary.update(table=table, prose=prose, points_data=points)
    return summary


def summary(run):
    from . import ttfb

    summary = report(run, allow_partial=True)
    rows = {r["metric"]: r for r in summary["table"]}

    def cell(scenario, label):
        r = rows[f"{scenario}: {label}"]
        if r["ttfb_s"] is None:
            return "—"
        if r["n_seeds"] < 2:
            return f"{r['ttfb_s']:.2f} (1 seed)"
        return f"{r['ttfb_s']:.2f} ± {r['std_s']:.2f}"

    labels = [
        "Zephon threads (GIL)",
        "Zephon processes (GIL)",
        "Zephon threads (no GIL)",
        "Zephon processes (no GIL)",
        "Streaming",
        "Grain",
        "LitData",
    ]
    paper = [
        table(
            "§8.1.3 Process overhead, DOP 64",
            ["Loader", "Text (s)", "Raw VLM (s)"],
            [
                [label, cell("fetch-text", label), cell("fetch-vlm", label)]
                for label in labels
            ],
        ),
        table(
            "Full VLM pipeline, DOP 64",
            ["Loader", "TTFB (s)"],
            [
                [label, cell("vlm-pipeline", label)]
                for label in ("InternalDs", "Zephon processes (GIL)", "Grain")
            ],
        ),
        table(
            "Startup comparisons",
            ["Measurement", "Observed", "Unit"],
            [
                [r["metric"], value(r["value"]), r["unit"]]
                for r in summary["prose"][-2:]
            ],
        ),
    ]
    points = summary["points_data"]
    phases = []
    for selected in summary["table"]:
        python = (
            "3.14t"
            if "(no GIL)" in selected["system"]
            else "3.12"
            if selected["system"] == "InternalDs"
            else "3.13"
        )
        candidates = [
            p
            for p in points
            if p["dop"] == 64
            and p["scenario"] == selected["scenario"]
            and p["source"] == selected["source"]
            and p["loader_name"] == selected["loader_name"]
            and p["python"] == python
            and p["batch_size"] == (4 if p["scenario"] == "vlm-pipeline" else 32)
        ]
        if candidates:
            if len(candidates) != 1:
                raise ValueError("Ambiguous startup phase breakdown")
            phases.append(candidates[0])
    extra = [
        table(
            "Startup phase breakdown at DOP 64",
            ["Scenario / loader", "Factory (s)", "Iterator (s)", "First next (s)"],
            [
                [
                    p["scenario"]
                    + " / "
                    + p["system"]
                    + " "
                    + p["runner"]
                    + " "
                    + p["python"],
                    value(p["factory_s"]),
                    value(p["iter_s"]),
                    value(p["first_next_s"]),
                ]
                for p in phases
            ],
        ),
        table(
            "Measurement coverage",
            ["Observations", "Aggregated points"],
            [[summary["observations"], summary["points"]]],
        ),
    ]
    campaign = core.read_json(run / "campaign.json")
    complete = len(campaign["cases"]) == len(ttfb.plan()) and all(
        p["n_seeds"] == 3 for p in points
    )
    for section, scenarios in zip(
        paper[:2], (("fetch-text", "fetch-vlm"), ("vlm-pipeline",))
    ):
        missing = dict.fromkeys(
            f"{r['system']}: {r['unavailable_reason']}."
            for r in summary["table"]
            if r["scenario"] in scenarios and r["unavailable_reason"]
        )
        section["note"] = " ".join(missing)
    paper[2]["note"] = " ".join(
        f"{r['metric']}: {r['unavailable_reason']}."
        for r in summary["prose"][-2:]
        if r["unavailable_reason"]
    )
    return document(
        "Startup latency reproduction — §8.1.3",
        [
            "Mean ± sample standard deviation across workload seeds; one trial per seed. Fetch batch 32; full VLM batch 4.",
            summary["input_note"],
            "This subsection has no paper figure.",
        ],
        paper,
        extra,
        status="Complete selected paper matrix"
        if complete
        else "Partial/diagnostic selection — not the final paper aggregates",
    )
