"""Thin adapter to the paper plotting functions; run in their Python environment."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .core import read_json, write_json


def generate(
    run: Path,
    plots: Path,
    render: bool = False,
    *,
    workdir: Path | None = None,
    figures: Path | None = None,
) -> dict:
    sys.path.insert(0, str(plots))
    campaign = read_json(run / "campaign.json")
    experiment = campaign["cases"][0]["experiment"]
    module_name = (
        "plot_fig_6_fetchnbatch"
        if experiment == "fetch-and-batch"
        else "plot_fig_7_vlm_scaling"
    )
    module = importlib.import_module(f"plotting.system.{module_name}")
    destination = workdir or run / "report"

    # Both tables and figures use the plotting repository's seed aggregation.
    data = module.load_and_process(destination / "plot-data")
    data.write_csv(destination / "points.csv")
    selected = (
        module.select_best_configs(data) if experiment == "vlm-pipeline" else None
    )
    if selected is not None:
        selected.write_csv(destination / "selected.csv")
    summary = {
        "experiment": experiment,
        "points": data.to_dicts(),
        "selected_allocations": selected.to_dicts() if selected is not None else [],
    }
    write_json(destination / "summary.json", summary)
    if render:
        directory = figures or destination / "figures"
        if experiment == "fetch-and-batch":
            module.plot_fig_6(data, directory, filename_prefix="figure-6-")
        else:
            module.plot_fig_7(selected, directory, filename_prefix="figure-7-")
    print(f"Measured numbers: {destination / 'points.csv'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("plots", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    generate(
        args.run, args.plots, args.render, workdir=args.workdir, figures=args.figures
    )


if __name__ == "__main__":
    main()
