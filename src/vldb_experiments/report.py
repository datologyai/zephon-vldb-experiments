"""Export arena observations and use zephon-vldb-plots for tables and figures."""

from __future__ import annotations

import csv
import os
import shlex
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import ROOT, core
from .core import case_state, read_json
from .matrix import is_full_matrix
from .presentation import document, table, value


def collected(run: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every completed case of a campaign, with its arena observation."""
    values = []
    for case in read_json(run / "campaign.json")["cases"]:
        folder = run / "cases" / case["id"]
        if case_state(folder) == "complete":
            values.append((case, read_json(folder / "result.json")))
    return values


def vlm_payload_layout(case: dict, result: dict) -> str | None:
    """Whether a Zephon VLM case used flat tensors or nested objects, if known."""
    if case["experiment"] != "vlm-pipeline" or case["system"] != "zephon":
        return None
    observed = result.get("resolved_config", {}).get("vlm_zephon", {})
    configured = case["config"].get("vlm_zephon", {})
    value = observed.get("flatten_payload", configured.get("flatten_payload"))
    return None if value is None else ("flat-tensors" if value else "nested-objects")


def plot_filename(case: dict) -> str:
    """The CSV filenames consumed by the existing paper plotting modules."""
    if case["experiment"] == "vlm-pipeline":
        return {
            "zephon": "vldb_vlm_zephon-fused.csv",
            "grain": "vldb_vlm_grain.csv",
        }[case["system"]]
    dataset = "vlm-raw" if case["dataset"] == "vlm" else "text"
    if case["system"] == "zephon":
        suffix = "-noshm" if case["variant"] == "noshm" else ""
        return f"vldb_fnb_zephon-{dataset}{suffix}.csv"
    return f"vldb_fnb_others-{dataset}.csv"


def export_results(run: Path, destination: Path | None = None) -> int:
    """Adapt the measured CSV rows to the plotting repository's input layout."""
    exports = defaultdict(list)
    layouts = defaultdict(set)
    for case, result in collected(run):
        if case["id"].endswith("-smoke"):
            continue
        key = (
            case["dataset"],
            case["system"],
            case["python"],
            case["loader_name"],
            case["variant"],
            case["dop"],
        )
        layouts[key].add(vlm_payload_layout(case, result))
        row = dict(
            result["csv_row"],
            sweep_python=case["python"],
            sweep_run_name="litdata-local",
        )
        exports[plot_filename(case)].append(row)
    if any(len(values) > 1 for values in layouts.values()):
        raise ValueError("Cannot pool different VLM payload layouts")
    if not exports:
        raise ValueError("No completed full-length measurements to report")
    destination = destination or run / "report/plot-data"
    destination.mkdir(parents=True, exist_ok=True)
    # This directory contains only generated inputs; avoid including stale rows.
    for path in destination.glob("*.csv"):
        path.unlink()
    for name, rows in exports.items():
        with (destination / name).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=sorted(set().union(*(r.keys() for r in rows))),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
    return sum(map(len, exports.values()))


def require_complete(run: Path) -> None:
    """Keep truncated runs and smoke windows out of complete paper outputs."""
    planned = read_json(run / "campaign.json")["cases"]
    measured = collected(run)
    if not planned or len(measured) != len(planned):
        raise ValueError("Paper output requires a complete experiment")
    if not is_full_matrix(planned[0]["experiment"], planned):
        raise ValueError(
            "Paper output requires the full experiment matrix and three seeds"
        )
    if any(vlm_payload_layout(c, r) == "nested-objects" for c, r in measured):
        raise ValueError("The paper VLM experiment uses flat tensor payloads")


def plotting_python(plots: Path | None = None) -> Path:
    """Check the prepared plotting environment without installing dependencies."""
    plots = (plots or ROOT.parent / "zephon-vldb-plots").expanduser().resolve()
    interpreter = plots / ".venv/bin/python"
    setup = shlex.join(
        ["uv", "run", "reproduce", "prepare", "plotting", "--plots", str(plots)]
    )
    message = f"Plotting environment is not ready. Prepare it with:\n  {setup}"
    if not interpreter.is_file():
        raise RuntimeError(message)
    try:
        subprocess.run(
            [str(interpreter), "-I", "-c", "import matplotlib, polars"],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ | {"PYTHON_GIL": "1", "MPLBACKEND": "Agg"},
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{message}\n{error.stderr.strip()}") from error
    return interpreter


def analyze(run: Path, plots: Path | None = None, *, render: bool = False) -> dict:
    """Aggregate raw observations in scratch; only requested figures persist."""
    run = run.expanduser().resolve()
    if render:
        require_complete(run)
    if plots is None:
        plots = Path(read_json(run / "campaign.json")["profile"]["paths"]["plots"])
    plots = plots.expanduser().resolve()
    interpreter = plotting_python(plots)
    with tempfile.TemporaryDirectory(prefix="vldb-throughput-") as temporary:
        scratch = Path(temporary)
        export_results(run, scratch / "plot-data")
        command = [
            str(interpreter),
            "-m",
            "vldb_experiments.plotting",
            str(run),
            str(plots),
            "--workdir",
            str(scratch),
        ]
        if render:
            command += ["--render", "--figures", str(run / "report/figures")]
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ
            | {
                "PYTHONPATH": str(ROOT / "src"),
                "PYTHON_GIL": "1",
                "MPLBACKEND": "Agg",
            },
        )
        return read_json(scratch / "summary.json")


def summary(run, plots=None, results=None):
    from . import matrix
    from .prose import prose_values

    folders = {
        n: run / n
        for n in ("fetch-and-batch", "vlm-pipeline")
        if (run / n / "campaign.json").exists()
    }
    if not folders:
        folders = {core.read_json(run / "campaign.json")["cases"][0]["experiment"]: run}
    results = (
        results
        if results is not None
        else {name: analyze(folder, plots) for name, folder in folders.items()}
    )
    complete = all(
        matrix.is_full_matrix(name, core.read_json(p / "campaign.json")["cases"])
        and len(collected(p)) == len(core.read_json(p / "campaign.json")["cases"])
        for name, p in folders.items()
    )
    if complete:
        try:
            for folder in folders.values():
                require_complete(folder)
        except ValueError:
            complete = False
    metrics = (
        prose_values(results.get("fetch-and-batch"), results.get("vlm-pipeline"))
        if complete
        else []
    )
    groups = [[], [], [], []]
    for row in metrics:
        name = row["metric"]
        index = (
            3
            if name.startswith("VLM pipeline:")
            else 2
            if any(s in name for s in ("SHM", "pipe-only", "noshm"))
            else 1
            if any(s in name for s in ("no-GIL", "3.14t"))
            else 0
        )
        # Fixed-DOP VLM rates appear once in the full selected-curve table below.
        if name.startswith("VLM pipeline:") and row["unit"] == "usable tokens/s/DOP":
            continue
        groups[index].append(
            [
                name,
                value(row["value"]),
                row["unit"],
            ]
        )
    paper = [
        table(
            title,
            ["Measurement", "Observed", "Unit"],
            rows,
        )
        for title, rows in zip(
            (
                "§8.1.1 Fetch-and-batch",
                "Effect of GIL removal",
                "SHM coalescing",
            ),
            groups[:3],
        )
        if rows
    ]
    selected = results.get("vlm-pipeline", {}).get("selected_allocations", [])
    if selected and complete:
        paper.append(
            table(
                "§8.1.2 VLM pipeline — Figure 7",
                ["Loader", "DOP", "Usable tokens/s/DOP"],
                [
                    [p["system"], p["dop"], value(p["utps_mean"] / p["dop"])]
                    for p in sorted(selected, key=lambda p: (p["dop"], p["system"]))
                ],
            )
        )
    if groups[3]:
        paper.append(
            table(
                "VLM pipeline comparisons",
                ["Measurement", "Observed", "Unit"],
                groups[3],
            )
        )
    extra = []
    if selected and not complete:
        extra.append(
            table(
                "Diagnostic VLM observations",
                ["Loader", "DOP", "Usable tokens/s/DOP"],
                [
                    [p["system"], p["dop"], value(p["utps_mean"] / p["dop"])]
                    for p in selected
                ],
                "Selection is incomplete; these are not final paper allocations.",
            )
        )
    for name, result in results.items():
        extra.append(
            table(
                "Measurement coverage: " + name,
                ["Measured cases", "Aggregated points"],
                [[len(collected(folders[name])), len(result.get("points", []))]],
                "Unselected VLM allocations and individual seed observations remain available in the raw cases.",
            )
        )
    figures = (
        "Figures 6 and 7"
        if len(folders) == 2
        else ("Figure 6" if "fetch-and-batch" in folders else "Figure 7")
    )
    return document(
        "Throughput reproduction — " + figures,
        [
            "Seed means use the original zephon-vldb-plots aggregation. Fetch peaks are selected independently; VLM tokens/s is normalized by DOP.",
            "Figures retain the original paper renderer and public-loader panel order.",
            "Results cover the public loaders; private paper baselines are omitted.",
        ],
        paper,
        extra,
        status="Complete selected paper matrix"
        if complete
        else "Partial/diagnostic selection — not the final paper aggregates",
    )
