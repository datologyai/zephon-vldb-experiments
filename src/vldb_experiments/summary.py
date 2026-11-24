"""Print experiment summaries and render paper figures from saved raw data."""

from __future__ import annotations

import json
from pathlib import Path

from . import ROOT, core
from .presentation import render


def kind(run: Path) -> str:
    if (run / "combined.json").exists():
        return "training"
    if (run / "fetch-and-batch/campaign.json").exists() or (
        run / "vlm-pipeline/campaign.json"
    ).exists():
        return "throughput"
    campaign = core.read_json(run / "campaign.json")
    if "config" in campaign and "cases" not in campaign:
        return "training"
    cases = campaign.get("cases", [])
    if cases and cases[0].get("measurement") == "ttfb":
        return "ttfb"
    if cases and "optimizer_steps" in cases[0]:
        return "checkpoint"
    if cases and cases[0].get("experiment") in ("fetch-and-batch", "vlm-pipeline"):
        return "throughput"
    raise ValueError("Unrecognized campaign manifest: " + str(run))


def summarize(
    run: Path,
    *,
    plots: Path | None = None,
    format: str = "terminal",
) -> dict:
    run = run.expanduser().resolve()
    experiment = kind(run)
    if plots is None and (run / "remote.json").exists():
        plots = Path(core.read_json(run / "remote.json")["plots"])
    if experiment == "throughput":
        from .report import summary

        result = summary(run, plots)
    elif experiment == "ttfb":
        from .ttfb_report import summary

        result = summary(run)
    elif experiment == "checkpoint":
        from .checkpoint_summary import summary

        result = summary(run)
    else:
        from .training.trace import summary

        result = summary(run)
    print(
        json.dumps(result, indent=2) if format == "json" else render(result),
        end="\n" if format == "json" else "",
    )
    return result


def plot(run: Path, *, plots: Path | None = None) -> list[Path]:
    """Create the paper's figures. Text-only subsections have nothing to render."""
    import os
    import subprocess

    from .report import analyze, plotting_python

    run = run.expanduser().resolve()
    experiment = kind(run)
    if plots is None and (run / "remote.json").exists():
        plots = Path(core.read_json(run / "remote.json")["plots"])
    if experiment in ("ttfb", "checkpoint"):
        return []
    if experiment == "training":
        from .training import protocol

        manifest = (
            core.read_json(run / "combined.json")
            if (run / "combined.json").exists()
            else None
        )
        source = run / manifest["sources"][0]["path"] if manifest else run
        if not protocol.plot_geometry(
            core.read_json(source / "campaign.json")["config"]
        ):
            return []
        plots = (plots or ROOT.parent / "zephon-vldb-plots").expanduser().resolve()
        interpreter = plotting_python(plots)
        subprocess.run(
            [
                str(interpreter),
                "-m",
                "vldb_experiments.training.plotting",
                str(run),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ
            | {"PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg", "PYTHON_GIL": "1"},
        )
        folder = run / "report/figures"
        return sorted(
            p for p in folder.iterdir() if p.suffix in (".svg", ".png", ".pdf")
        )
    folders = [
        run / n
        for n in ("fetch-and-batch", "vlm-pipeline")
        if (run / n / "campaign.json").exists()
    ] or [run]
    for folder in folders:
        analyze(folder, plots, render=True)
    return sorted(
        p
        for folder in folders
        for p in (folder / "report/figures").iterdir()
        if p.suffix in (".svg", ".png", ".pdf")
    )


def finish(run: Path, *, plots: Path | None = None):
    result = summarize(run, plots=plots)
    if result["status"].startswith("Partial"):
        return []
    paths = plot(run, plots=plots)
    if paths:
        print("\nPaper figures:")
        for folder in sorted({p.parent for p in paths}):
            print(f"  {folder}")
    return paths
