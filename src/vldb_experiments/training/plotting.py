"""Adapt verified training results to the paper checkpoint-resume renderer.

Run in zephon-vldb-plots' Python environment, just like the throughput adapter.
Plots use only the campaign's verified traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from . import protocol, trace


def paper_data(run: Path) -> tuple[list[dict], int]:
    if (run / "combined.json").exists():
        binding = json.loads((run / "combined.json").read_text())
        summary = trace.validate_report(run)
        if summary["identity"] != binding["identity"]:
            raise ValueError("Combined report and source binding identities differ")
        rows, checkpoints, arms = [], set(), set()
        for source in binding["sources"]:
            directory = run / source["path"]
            manifest = json.loads((directory / "campaign.json").read_text())
            if manifest["identity"] != source["identity"]:
                raise ValueError("Combined source campaign identity changed")
            selected = set(source["arms"])
            if selected & arms:
                raise ValueError("Combined plotting sources duplicate a loader arm")
            arms.update(selected)
            measured, checkpoint = paper_data(directory)
            checkpoints.add(checkpoint)
            rows.extend(
                row | {"campaign_identity": source["identity"]} for row in measured
            )
        if arms != {"control", "zephon"} or len(checkpoints) != 1:
            raise ValueError(
                "Combined plotting requires both matching loader-arm experiments"
            )
        return rows, checkpoints.pop()
    manifest = json.loads((run / "campaign.json").read_text())
    config = manifest["config"]
    if not protocol.plot_geometry(config):
        raise ValueError(
            "Plotting requires the complete DP16 -> DP8/32/64 or DP2 -> DP1/4/8 experiment"
        )
    summary = trace.report(run)
    if summary["identity"] != manifest["identity"]:
        raise ValueError("Report and campaign identities differ")
    checkpoint = config["checkpoint_step"]
    rows = []
    for stage in protocol.plan(config):
        source_name = protocol.source_name(stage)
        if source_name is None:
            continue
        baseline_stage = next(
            s for s in protocol.plan(config) if s["name"] == source_name
        )
        baseline = trace.read_trace(
            run / source_name, baseline_stage, manifest["identity"]
        )
        dp = stage["dp"]
        compared = summary["comparisons"][stage["name"]]["steps"]
        if [int(row["step"]) for row in compared] != list(
            range(checkpoint + 1, config["steps"] + 1)
        ):
            raise ValueError(f"Incomplete comparison for {stage['name']}")
        by_step = {int(row["step"]): row for row in compared}
        for step, loss in baseline["losses"].items():
            post = step > checkpoint
            source = by_step.get(step)
            reference_hash = protocol.digest(sorted(baseline["inputs"][step]))
            # Each loader arm has its own measured reference history. Only
            # updates after the checkpoint come from resumed GPU executions.
            rows.append(
                {
                    "arm": "elastic"
                    if protocol.stage_arm(stage) == "zephon"
                    else "control",
                    "resume_dp": dp,
                    "reference_dp": config["baseline_dp"],
                    "step": step,
                    "origin": "resume" if post else "shared_reference",
                    "is_post_fault": post,
                    "reference_loss": loss,
                    "fault_loss": float(source["resume_loss"]) if post else loss,
                    "reference_model_batch_multiset_sha256": reference_hash,
                    "fault_model_batch_multiset_sha256": source["resume_input_sha256"]
                    if post
                    else reference_hash,
                    "full_hash_match": source["resume_input_sha256"] == reference_hash
                    if post
                    else True,
                }
            )
    return rows, checkpoint


def render(run: Path, *, output: Path | None = None) -> list[Path]:
    import polars as pl
    from plotting.system import plot_fig_1_fault_resume as paper

    rows, checkpoint = paper_data(run)
    references = {row["reference_dp"] for row in rows}
    if len(references) != 1:
        raise ValueError("Plotting sources have different baseline GPU counts")
    reference = references.pop()
    resumes = sorted({row["resume_dp"] for row in rows})
    data = pl.DataFrame(rows)
    output = output or run / "report/figures"
    paths = paper.plot_checkpoint_resume(
        data,
        checkpoint,
        output,
        reference_width=reference,
        resume_widths=tuple(resumes),
    )
    arms = [arm for arm in ("control", "elastic") if arm in data["arm"].unique()]
    data_sha256 = hashlib.sha256(data.write_csv().encode()).hexdigest()
    plots = Path(paper.__file__).resolve().parents[2]
    files = [
        Path(paper.__file__),
        plots / "plotting/common/common.py",
        plots / "plotting/common/plotcolours.py",
    ]
    provenance = {
        "schema": 1,
        "campaign_identity": json.loads(
            (
                run
                / (
                    "combined.json"
                    if (run / "combined.json").exists()
                    else "campaign.json"
                )
            ).read_text()
        )["identity"],
        "renderer": "plotting.system.plot_fig_1_fault_resume.plot_checkpoint_resume",
        "variant": "small",
        "reference_dp": reference,
        "resume_dps": resumes,
        "arms": arms,
        "adapter_sha256": protocol.file_digest(Path(__file__)),
        "code_sha256": {
            str(p.relative_to(plots)): protocol.file_digest(p) for p in files
        },
        "packages": {name: version(name) for name in ("matplotlib", "numpy", "polars")},
        "data_sha256": data_sha256,
        "shared_history": "Steps through the checkpoint come from the reference; only later steps are resume measurements.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for path in render(args.run, output=args.output):
        print(path)


if __name__ == "__main__":
    main()
