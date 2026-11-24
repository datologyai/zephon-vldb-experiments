"""Shared public-asset staging and native checkpoint selection for both backends."""

from __future__ import annotations

import json
from pathlib import Path

from .protocol import source_name
from .trace import read_trace


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def completed_attempt(run: Path, stage: dict, identity: str) -> Path:
    """Select a complete single execution, preserving interrupted evidence separately."""
    import shutil

    attempts = {}
    for path in sorted((run / "sample-hashes").glob("*.rank-*.jsonl")):
        attempt, name = path.name.split(".rank-", 1)
        attempts.setdefault(attempt, []).append((path, "rank-" + name))
    complete = []
    for attempt, files in attempts.items():
        directory = run / "attempts" / attempt
        diagnostics = directory / "diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=True)
        for path, name in files:
            shutil.copyfile(path, diagnostics / name)
        try:
            read_trace(directory, stage, identity)
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        complete.append(directory)
    if len(complete) != 1:
        raise ValueError(
            f"Expected one complete, uninterrupted execution of {stage['name']}; found {len(complete)}"
        )
    return complete[0]


def stage_source(
    stage: dict, completed: dict, config: dict
) -> tuple[str | None, str | None]:
    reference = source_name(stage)
    if reference is None:
        return None, None
    record = completed[reference]
    return (
        f"{record['directory']}/checkpoint/step-{config['checkpoint_step']}",
        record["proof"]["source"]["metadata_sha256"],
    )
