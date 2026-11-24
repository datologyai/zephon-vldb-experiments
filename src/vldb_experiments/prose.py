"""Show the measured throughput numbers described in the paper's prose."""

from __future__ import annotations


def prose_values(fetch_result: dict, vlm_result: dict) -> list[dict]:
    """Use the plotting repository's aggregates and selected VLM allocations."""
    rows = []

    def fetch(
        dataset, system="zephon", python="3.13", runner="process", variant="baseline"
    ):
        """Peak mean samples/s of one Figure 6 line, and the DOP where it occurs."""
        points = [
            p
            for p in fetch_result["points"]
            if p["data_type"] == ("vlm-raw" if dataset == "vlm" else "text")
            and p["system"] == system
            and p["py"] == python
            and p["variant"] == variant
            and (
                system != "zephon"
                or p["runner"] == ("processes" if runner == "process" else "threads")
            )
        ]
        peak = max(points, key=lambda p: p["sps_mean"])
        line = f"{dataset}/{system}/{python}/{runner}/{variant}"
        return peak["sps_mean"], f"{line}, DOP {peak['parallelism']}"

    def pipeline(system, dop):
        """Usable tokens/s per DOP of the allocation plotted in Figure 7."""
        plotted_system = "zephon-fused" if system == "zephon" else system
        point = next(
            p
            for p in vlm_result["selected_allocations"]
            if p["system"] == plotted_system and p["dop"] == dop
        )
        return point["utps_mean"] / dop, f"DOP {dop}: {point['loader_name']}"

    def ratio(numerator, denominator):
        value, configuration = numerator
        divisor, denominator_configuration = denominator
        return value / divisor, configuration + " / " + denominator_configuration

    def percent_change(after, before):
        value, configuration = ratio(after, before)
        return 100 * (value - 1), configuration

    def add(metric, measurement, unit):
        """Format a measured paper quantity."""
        value, configuration = measurement
        row = {
            "metric": metric,
            "value": value,
            "unit": unit,
            "configuration": configuration,
        }
        rows.append(row)

    if fetch_result:
        for dataset, system, python, runner, variant in [
            ("text", "grain", "3.13", "workers", "baseline"),
            ("text", "litdata", "3.13", "workers", "baseline"),
            ("text", "zephon", "3.13", "process", "baseline"),
            ("text", "zephon", "3.13", "threads", "baseline"),
            ("vlm", "litdata", "3.13", "workers", "baseline"),
            ("vlm", "grain", "3.13", "workers", "baseline"),
            ("vlm", "zephon", "3.13", "process", "baseline"),
            ("vlm", "zephon", "3.13", "threads", "baseline"),
            ("text", "zephon", "3.14t", "threads", "baseline"),
            ("text", "zephon", "3.14t", "process", "baseline"),
            ("vlm", "zephon", "3.14t", "threads", "baseline"),
            ("vlm", "zephon", "3.14t", "process", "baseline"),
            ("text", "zephon", "3.13", "process", "noshm"),
            ("vlm", "zephon", "3.13", "process", "noshm"),
        ]:
            add(
                f"{dataset}: {system} {python} {runner} {variant} peak",
                fetch(dataset, system, python, runner, variant),
                "samples/s",
            )

        best = max((fetch("vlm", s) for s in ("grain", "litdata")), key=lambda p: p[0])
        add(
            "raw VLM: process / best worker baseline",
            ratio(fetch("vlm"), best),
            "×",
        )
        add(
            "text: no-GIL thread / GIL thread",
            ratio(
                fetch("text", python="3.14t", runner="threads"),
                fetch("text", runner="threads"),
            ),
            "×",
        )
        add(
            "text: no-GIL process / GIL process",
            ratio(fetch("text", python="3.14t"), fetch("text")),
            "×",
        )
        add(
            "raw VLM: no-GIL thread / GIL thread",
            ratio(
                fetch("vlm", python="3.14t", runner="threads"),
                fetch("vlm", runner="threads"),
            ),
            "×",
        )
        add(
            "raw VLM: no-GIL thread / no-GIL process",
            ratio(
                fetch("vlm", python="3.14t", runner="threads"),
                fetch("vlm", python="3.14t"),
            ),
            "×",
        )
        add(
            "text: throughput change when disabling SHM",
            percent_change(fetch("text", variant="noshm"), fetch("text")),
            "%",
        )
        add(
            "raw VLM: throughput change when disabling SHM",
            percent_change(fetch("vlm", variant="noshm"), fetch("vlm")),
            "%",
        )
        add(
            "raw VLM: SHM / pipe-only",
            ratio(fetch("vlm"), fetch("vlm", variant="noshm")),
            "×",
        )

    if vlm_result:
        for system, dop in [
            ("zephon", 32),
            ("grain", 32),
            ("zephon", 64),
            ("grain", 64),
        ]:
            add(
                f"VLM pipeline: {system} at DOP {dop}",
                pipeline(system, dop),
                "usable tokens/s/DOP",
            )
        add(
            "VLM pipeline: Grain lead over Zephon at DOP 64",
            percent_change(pipeline("grain", 64), pipeline("zephon", 64)),
            "%",
        )
    return rows
