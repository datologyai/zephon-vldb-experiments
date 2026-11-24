"""Tables and prose quantities from the measured checkpoint campaign."""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from . import core
from .checkpoint import validate_result

PAPER_METRICS = {
    "fresh_to_first_batch",
    "save_end_to_end",
    "checkpoint_mib",
    "inflight_chunks_mean_per_rank",
    "inflight_source_records_mean_per_rank",
    "restore_to_first_batch",
    "restore_to_fresh_ratio_of_medians",
}
PHASE_METRICS = (
    "loader_construction",
    "iterator_build",
    "first_batch_pull",
    "pipeline_checkpoint",
    "checkpoint_serialize",
    "checkpoint_write",
    "save_sync_barrier",
    "checkpoint_read",
    "checkpoint_deserialize",
    "restore_stash",
    "restore_sync_barrier",
    "consume_after_first",
)


def report(output: Path) -> list[dict]:
    campaign = core.read_json(output / "campaign.json")
    rows = []
    for case in campaign["cases"]:
        directory = output / "cases" / case["id"]
        if core.case_state(directory) != "complete":
            continue
        result = core.read_json(directory / "result.json")
        validate_result(case, result)
        row = {
            k: case[k]
            for k in (
                "id",
                "point",
                "mode",
                "k",
                "chunk_size",
                "seed",
                "repetition",
                "lanes",
                "optimizer_steps",
            )
        }
        row["analysis_tags"] = ",".join(case["analysis_tags"])
        row.update(result["max_across_ranks_s"])
        if all("construction_s" in rank for rank in result["ranks"]):
            row["loader_construction"] = max(
                rank["construction_s"] for rank in result["ranks"]
            )
        if case["mode"] != "fresh":
            rank = result["ranks"][0]
            row["checkpoint_bytes"] = rank["checkpoint_bytes"]
            row["checkpoint_mib"] = rank["checkpoint_bytes"] / 2**20
            source = rank["checkpoint_source_state"]
            for metric in ("chunks", "source_records"):
                row[f"inflight_{metric}_total"] = source["total"][metric]
                row[f"inflight_{metric}_mean_per_rank"] = source["total"][metric] / len(
                    source["per_dp_rank"]
                )
                row[f"inflight_{metric}_max_per_rank"] = max(
                    v[metric] for v in source["per_dp_rank"].values()
                )
                row[f"inflight_{metric}_max_per_lane"] = max(
                    v[metric] for v in source["per_lane"].values()
                )
        windows = {rank["memory"]["window"] for rank in result["ranks"]}
        if len(windows) != 1:
            raise ValueError("Ranks have different RSS measurement windows")
        row["rss_window"] = windows.pop()
        for metric in ("baseline", "peak", "incremental"):
            row[f"rss_{metric}_max_per_rank_gib"] = (
                max(rank["memory"][f"rss_{metric}_bytes"] for rank in result["ranks"])
                / 2**30
            )
        rows.append(row)
    return rows


def prose_values(rows: list[dict]) -> list[dict]:
    """Paper quantities grouped by their actual main, progress and lane protocols."""
    grouped = defaultdict(list)
    metrics = {
        "fresh": ["fresh_to_first_batch"],
        "save": [
            "save_end_to_end",
            "checkpoint_mib",
            "inflight_chunks_max_per_rank",
            "inflight_source_records_max_per_rank",
        ],
        "restore": ["restore_to_first_batch"],
    }
    for row in rows:
        for suite in row["analysis_tags"].split(","):
            # Main latency/size prose pools K, C, seeds and repeats. Retained-source
            # bounds additionally need each C; progress/lanes keep their x-axis.
            rss_metrics = [
                f"rss_{metric}_max_per_rank_gib"
                for metric in ("baseline", "peak", "incremental")
            ]
            optional = list(PHASE_METRICS) + (
                rss_metrics
                + [
                    "inflight_chunks_mean_per_rank",
                    "inflight_source_records_mean_per_rank",
                ]
                if row["mode"] == "save"
                else rss_metrics
            )
            for metric in metrics[row["mode"]] + [m for m in optional if m in row]:
                axis = (
                    ("all", "all", "all")
                    if suite == "main"
                    else (row["optimizer_steps"], row["lanes"], row["chunk_size"])
                )
                if suite == "main" and metric.startswith("inflight_"):
                    axis = ("all", "all", row["chunk_size"])
                grouped[(suite, row["mode"], *axis, metric)].append(float(row[metric]))
    values = []
    for key, numbers in sorted(grouped.items(), key=lambda item: str(item[0])):
        low, _, high = (
            statistics.quantiles(numbers, n=4, method="inclusive")
            if len(numbers) > 1
            else [numbers[0]] * 3
        )
        suite, mode, steps, lanes, chunk, metric = key
        values.append(
            dict(
                suite=suite,
                mode=mode,
                optimizer_steps=steps,
                lanes=lanes,
                chunk_size=chunk,
                metric=metric,
                n=len(numbers),
                median=statistics.median(numbers),
                q25=low,
                q75=high,
                unit="MiB"
                if metric.endswith("_mib")
                else "GiB"
                if metric.endswith("_gib")
                else "count"
                if metric.startswith("inflight_")
                else "seconds",
            )
        )
    fresh = {
        row["point"]: row["fresh_to_first_batch"]
        for row in rows
        if row["mode"] == "fresh" and "main" in row["analysis_tags"].split(",")
    }
    restore = {
        row["point"]: row["restore_to_first_batch"]
        for row in rows
        if row["mode"] == "restore" and "main" in row["analysis_tags"].split(",")
    }
    paired = sorted(fresh.keys() & restore.keys())
    if paired:
        ratios = [restore[p] / fresh[p] for p in paired]
        q25, _, q75 = (
            statistics.quantiles(ratios, n=4, method="inclusive")
            if len(ratios) > 1
            else ratios * 3
        )
        values.append(
            dict(
                suite="main",
                mode="paired",
                optimizer_steps="all",
                lanes="all",
                chunk_size="all",
                metric="restore_to_fresh_paired_ratio",
                n=len(paired),
                median=statistics.median(ratios),
                q25=q25,
                q75=q75,
                unit="ratio",
            )
        )
        ratio = statistics.median([restore[p] for p in paired]) / statistics.median(
            [fresh[p] for p in paired]
        )
        values.append(
            dict(
                suite="main",
                mode="paired",
                optimizer_steps="all",
                lanes="all",
                chunk_size="all",
                metric="restore_to_fresh_ratio_of_medians",
                n=len(paired),
                median=ratio,
                q25=None,
                q75=None,
                unit="ratio",
            )
        )
    for value in values:
        value["category"] = (
            "paper"
            if value["metric"] in PAPER_METRICS
            and not (value["suite"] == "lanes" and value["mode"] == "fresh")
            and not (value["suite"] == "main" and value["metric"] == "checkpoint_mib")
            else "additional"
        )
    return values
