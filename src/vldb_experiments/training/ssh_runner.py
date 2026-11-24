"""Single-node training coordinator inside the SSH backend's pinned container."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from . import protocol, trace
from .assets import validate_assets
from .execution import stage_source, write_json


def command(stage: dict) -> list[str]:
    if stage["nodes"] != 1:
        raise ValueError("The SSH backend supports one node only")
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={stage['dp']}",
        "-m",
        "torchtitan.train",
        "--module=vldb_experiments.training.runtime",
        "--config=get_config",
    ]


def publish(directory: Path, destination: Path, stage: dict, config: dict) -> dict:
    """Verify native files on disk before exporting the smaller evidence bundle."""
    proof = {
        "final": trace.checkpoint_proof(
            directory / "checkpoint" / f"step-{config['steps']}"
        )
    }
    files = [
        (directory / "diagnostics" / f"rank-{r}.jsonl", f"diagnostics/rank-{r}.jsonl")
        for r in range(stage["dp"])
    ]
    files += [(directory / "observed.json", "observed.json")]
    files += [
        (
            directory / "checkpoint" / f"step-{config['steps']}" / ".metadata",
            "final.metadata",
        )
    ]
    if protocol.source_name(stage) is None:
        step = config["checkpoint_step"]
        proof["source"] = trace.checkpoint_proof(
            directory / "checkpoint" / f"step-{step}"
        )
        files.append(
            (directory / "checkpoint" / f"step-{step}" / ".metadata", "source.metadata")
        )
    for source, name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Existing completed evidence differs: {name}")
        if not target.exists():
            shutil.copyfile(source, target)
    existing = destination / "checkpoint-proof.json"
    if existing.exists() and json.loads(existing.read_text()) != proof:
        raise ValueError("Existing completed native checkpoint proof differs")
    write_json(directory / "checkpoint-proof.json", proof)
    write_json(destination / "checkpoint-proof.json", proof)
    return proof


def preflight(required: int) -> dict:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < required:
        raise ValueError(f"This campaign needs {required} visible NVIDIA GPUs")
    if any(torch.cuda.get_device_capability(i)[0] < 8 for i in range(required)):
        raise ValueError("The pinned BF16 training recipe needs Ampere or newer GPUs")
    return {
        "gpus": [torch.cuda.get_device_name(i) for i in range(required)],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def coordinate(root: Path = Path("/runs"), assets: Path = Path("/assets")) -> dict:
    manifest = json.loads((root / "campaign.json").read_text())
    if protocol.code_identity() != manifest["code"]:
        raise ValueError(
            "Container training sources differ from the campaign fingerprint"
        )
    config = manifest["config"]
    stages = protocol.plan(config)
    if any(stage["nodes"] != 1 or stage["dp"] > 8 for stage in stages):
        raise ValueError(
            "SSH campaigns require single-node placements of at most eight GPUs"
        )
    continuation = root / "continue.json"
    retry = continuation.exists()
    if retry:
        request = json.loads(continuation.read_text())
        request_id = uuid.UUID(hex=request["request_id"]).hex
        if (
            request["identity"] != manifest["identity"]
            or request_id != request["request_id"]
        ):
            raise ValueError("Continuation belongs to a different campaign")
        audit = root / "continuations" / f"{request_id}.json"
        audit.parent.mkdir(exist_ok=True)
        continuation.replace(audit)
    write_json(
        root / "status.json", {"state": "running", "identity": manifest["identity"]}
    )
    try:
        write_json(
            root / "hardware.json", preflight(max(stage["dp"] for stage in stages))
        )
        write_json(root / "assets.json", validate_assets(config, assets))
        completed = {}
        for stage in stages:
            destination = root / stage["name"]
            record_path = destination / "launch.json"
            old = json.loads(record_path.read_text()) if record_path.exists() else None
            source, metadata = stage_source(stage, completed, config)
            ready = False
            if old and retry and old["state"] != "complete":
                # A controller may have exited after training finished but before
                # publishing its record. Reuse a complete original attempt.
                previous = Path(old["directory"])
                for rank in range(stage["dp"]):
                    raw = (
                        previous / "diagnostics" / f"{old['attempt']}.rank-{rank}.jsonl"
                    )
                    if raw.exists():
                        shutil.copyfile(
                            raw, previous / "diagnostics" / f"rank-{rank}.jsonl"
                        )
                try:
                    trace.read_trace(previous, stage, manifest["identity"])
                    trace.checkpoint_proof(
                        previous / "checkpoint" / f"step-{config['steps']}"
                    )
                    if not protocol.source_name(stage):
                        trace.checkpoint_proof(
                            previous
                            / "checkpoint"
                            / f"step-{config['checkpoint_step']}"
                        )
                    ready = True
                except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError):
                    pass
            if old and (old["state"] == "complete" or ready):
                directory = Path(old["directory"])
            else:
                if old and not retry:
                    raise RuntimeError(
                        "Existing stage needs an explicit continuation of its terminal container"
                    )
                attempt = uuid.uuid4().hex
                directory = destination / "attempts" / attempt
                directory.mkdir(parents=True)
                settings = {
                    "config": config,
                    "identity": manifest["identity"],
                    "stage": stage,
                    "directory": str(directory),
                    "trace_directory": str(directory / "diagnostics"),
                    "attempt": attempt,
                    "source": source,
                    "source_metadata": metadata,
                }
                record = {
                    "state": "prepared",
                    "directory": str(directory),
                    "attempt": attempt,
                }
                write_json(record_path, record)
                write_json(directory / "launch.json", record)
                env = os.environ | {
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(stage["dp"]))),
                    "ZEPHON_REPRO_STAGE": json.dumps(settings),
                }
                args = command(stage) + [f"--output-dir={directory}"]
                with (directory / "torchrun.log").open("x") as log:
                    record["state"] = "running"
                    write_json(record_path, record)
                    try:
                        subprocess.run(
                            args,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=config["timeout_seconds"],
                            check=True,
                        )
                    except (subprocess.SubprocessError, OSError) as error:
                        record.update(state="failed", error=str(error))
                        write_json(record_path, record)
                        write_json(directory / "launch.json", record)
                        raise
                # Evidence uses attempt-qualified names; publish untouched complete files.
                for rank in range(stage["dp"]):
                    raw = directory / "diagnostics" / f"{attempt}.rank-{rank}.jsonl"
                    shutil.copyfile(
                        raw, directory / "diagnostics" / f"rank-{rank}.jsonl"
                    )
            candidate = trace.read_trace(directory, stage, manifest["identity"])
            contract = trace.validate_observed(
                json.loads((directory / "observed.json").read_text()), config, stage
            )
            proof = publish(directory, destination, stage, config)
            record = {"state": "complete", "directory": str(directory), "proof": proof}
            reference_name = protocol.source_name(stage)
            if reference_name:
                reference_stage = next(s for s in stages if s["name"] == reference_name)
                reference_dir = Path(completed[reference_name]["directory"])
                if any(load["metadata"] != metadata for load in candidate["loads"]):
                    raise ValueError("Resume loaded the wrong native source checkpoint")
                reference = trace.read_trace(
                    reference_dir, reference_stage, manifest["identity"]
                )
                if (
                    trace.validate_observed(
                        json.loads((reference_dir / "observed.json").read_text()),
                        config,
                        reference_stage,
                    )
                    != contract
                ):
                    raise ValueError(
                        "Reference and resume numerical/data settings differ"
                    )
                comparison = trace.compare(reference, candidate)
                comparison.update(trace.outcome(comparison, protocol.stage_arm(stage)))
                write_json(directory / "comparison.json", comparison)
                record["comparison"] = {
                    k: v for k, v in comparison.items() if k != "steps"
                }
            write_json(record_path, record)
            completed[stage["name"]] = record
        write_json(root / "result.json", completed)
        summary = trace.report(root)
        write_json(
            root / "status.json",
            {
                "state": "complete",
                "identity": manifest["identity"],
                "expected_outcomes_met": summary["expected_outcomes_met"],
                "experiment_scale": summary["experiment_scale"],
                "paper_protocol_match": summary["paper_protocol_match"],
                "stages": list(completed),
            },
        )
        return summary
    except BaseException as error:
        write_json(
            root / "status.json",
            {"state": "failed", "identity": manifest["identity"], "error": str(error)},
        )
        raise


if __name__ == "__main__":
    coordinate()
