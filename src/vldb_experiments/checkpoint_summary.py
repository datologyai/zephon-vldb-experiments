"""Readable checkpoint results, organized by paper claims and extra diagnostics."""

from __future__ import annotations

import statistics

from .presentation import document, table, value


def summary(run):
    from . import checkpoint as benchmark
    from . import checkpoint_report, core

    observations = checkpoint_report.report(run)
    metrics = checkpoint_report.prose_values(observations)

    def get(suite, mode, metric, steps="all", lanes="all", chunk="all", iqr=False):
        r = next(
            (
                r
                for r in metrics
                if (
                    r["suite"],
                    r["mode"],
                    r["metric"],
                    r["optimizer_steps"],
                    r["lanes"],
                    r["chunk_size"],
                )
                == (suite, mode, metric, steps, lanes, chunk)
            ),
            None,
        )
        if r is None:
            return "—"
        digits = (
            int(
                any(
                    v is not None and v != int(v)
                    for v in (r["median"], r["q25"], r["q75"])
                )
            )
            if r["unit"] == "count"
            else 2
        )
        text = value(r["median"], digits)
        return (
            text + f" [{value(r['q25'], digits)}–{value(r['q75'], digits)}]"
            if iqr and r["q25"] is not None
            else text
        )

    paper = [
        table(
            "§8.2 In-flight state, step 100",
            ["C", "Chunks/rank", "Source records/rank"],
            [
                [
                    c,
                    get(
                        "main",
                        "save",
                        "inflight_chunks_mean_per_rank",
                        chunk=c,
                        iqr=True,
                    ),
                    get(
                        "main",
                        "save",
                        "inflight_source_records_mean_per_rank",
                        chunk=c,
                        iqr=True,
                    ),
                ]
                for c in (1024, 2048, 4096)
            ],
        ),
        table(
            "State versus training progress",
            ["Steps", "Source records/rank"],
            [
                [
                    step,
                    get(
                        "progress",
                        "save",
                        "inflight_source_records_mean_per_rank",
                        step,
                        8,
                        1024,
                    ),
                ]
                for step in (100, 1000, 10000)
            ],
            "K=8, C=1,024, L=8, seed 42.",
        ),
        table(
            "Save cost",
            ["Step-100 sweep", "Median [IQR] (s)"],
            [
                [
                    "Save to S3, including aggregation",
                    get("main", "save", "save_end_to_end", iqr=True),
                ]
            ],
        ),
        table(
            "Checkpoint size and save cost versus progress",
            ["Steps", "Size (MiB)", "Save (s)"],
            [
                [
                    step,
                    get("progress", "save", "checkpoint_mib", step, 8, 1024),
                    get("progress", "save", "save_end_to_end", step, 8, 1024),
                ]
                for step in (100, 1000, 10000)
            ],
        ),
        table(
            "Restore cost",
            ["Step-100 sweep", "Median [IQR] (s)"],
            [
                [
                    "Cold-start TTFB",
                    get("main", "fresh", "fresh_to_first_batch", iqr=True),
                ],
                [
                    "Restore to first batch",
                    get("main", "restore", "restore_to_first_batch", iqr=True),
                ],
            ],
        ),
        table(
            "Restore cost versus progress",
            ["Steps", "Restore TTFB (s)"],
            [
                [
                    step,
                    get("progress", "restore", "restore_to_first_batch", step, 8, 1024),
                ]
                for step in (100, 1000, 10000)
            ],
        ),
        table(
            "Lane scaling",
            ["Lanes", "Records/rank", "Size (MiB)", "Save (s)", "Restore (s)"],
            [
                [
                    lanes,
                    get(
                        "lanes",
                        "save",
                        "inflight_source_records_mean_per_rank",
                        100,
                        lanes,
                        1024,
                    ),
                    get("lanes", "save", "checkpoint_mib", 100, lanes, 1024),
                    get("lanes", "save", "save_end_to_end", 100, lanes, 1024),
                    get("lanes", "restore", "restore_to_first_batch", 100, lanes, 1024),
                ]
                for lanes in (8, 64, 512)
            ],
        ),
    ]
    ratio = next(
        (r for r in metrics if r["metric"] == "restore_to_fresh_ratio_of_medians"), None
    )
    if ratio:
        paper[4]["note"] = (
            f"Restore overhead {100 * (ratio['median'] - 1):+.1f}%: ratio of medians over {ratio['n']} complete fresh/restore pairs."
        )
    extra = [
        table(
            "Process-tree RSS (GiB), maximum rank",
            ["Window", "Baseline", "Peak", "Increase"],
            [
                [
                    label,
                    *(
                        get("main", mode, f"rss_{metric}_max_per_rank_gib")
                        for metric in ("baseline", "peak", "incremental")
                    ),
                ]
                for mode, label in (
                    ("fresh", "Startup → first batch"),
                    ("save", "Checkpoint call → final write"),
                    ("restore", "Checkpoint read → first batch"),
                )
            ],
            "RSS is additional, not a paper metric. Shared pages may be counted repeatedly. Restore RSS ends at the first batch, not full replay.",
        ),
        table(
            "Paired restore/fresh ratio",
            ["Median paired ratio"],
            [[get("main", "paired", "restore_to_fresh_paired_ratio")]],
            "Additional aggregation; distinct from the paper's ratio of medians.",
        ),
    ]
    state_rows = []
    main_saves = [
        row
        for row in observations
        if row["mode"] == "save" and "main" in row["analysis_tags"].split(",")
    ]
    for chunk in (1024, 2048, 4096):
        selected = [row for row in main_saves if row["chunk_size"] == chunk]
        state_rows.append(
            [
                chunk,
                *[
                    value(statistics.median(row[field] for row in selected), 0)
                    if selected
                    else "—"
                    for field in (
                        "inflight_source_records_max_per_rank",
                        "inflight_source_records_max_per_lane",
                    )
                ],
                len(selected),
            ]
        )
    extra.append(
        table(
            "Retained source-state diagnostics",
            ["C", "Maximum rank", "Maximum lane", "n"],
            state_rows,
            "Median maximum-rank/lane source records; distinct from the paper's average across ranks.",
        )
    )
    for mode in ("fresh", "save", "restore"):
        phase_rows = [
            [r["metric"], value(r["median"]), r["n"]]
            for r in metrics
            if r["suite"] == "main"
            and r["mode"] == mode
            and r["metric"] in checkpoint_report.PHASE_METRICS
        ]
        phase_rows.sort(key=lambda row: checkpoint_report.PHASE_METRICS.index(row[0]))
        if phase_rows:
            extra.append(
                table(
                    mode.capitalize() + " phase timings",
                    ["Phase", "Median (s)", "n"],
                    phase_rows,
                )
            )
    total = len(core.read_json(run / "campaign.json")["cases"])
    complete = len(observations) == total and total == len(benchmark.plan())
    return document(
        "Checkpoint reproduction — §8.2",
        [
            "Eight CPU ranks; cold S3 cache; independent fresh/save/restore processes. Medians [IQR]; state is averaged across ranks before aggregation.",
            "This subsection has no paper figure. — means no completed measurement.",
        ],
        paper,
        extra,
        status=f"Complete: {len(observations)}/{total} jobs"
        if complete
        else f"Partial/diagnostic campaign: {len(observations)}/{total} jobs — not final paper aggregates",
    )
