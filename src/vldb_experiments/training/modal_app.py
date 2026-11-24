"""Modal execution; each stage is fixed-width and independent of throughput jobs."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

from .. import ROOT
from .assets import validate_assets
from .execution import completed_attempt, stage_source, write_json
from .protocol import plan, source_name, stage_arm
from .trace import checkpoint_proof, compare, outcome, read_trace, validate_observed


def coordinate(
    manifest: dict, functions: dict, specs: dict, controller: dict | None = None
):
    """Persist every submission before waiting; an uncertain launch is never bypassed."""
    import modal

    config, prefix = manifest["config"], manifest["prefix"]
    runs = modal.Volume.from_name(f"{prefix}-training-runs")
    registry = modal.Dict.from_name(f"{prefix}-training-state")
    stages = plan(config)
    root = Path("/runs")
    runs.reload()
    old = root / "campaign.json"
    if old.exists() and json.loads(old.read_text()) != manifest:
        raise ValueError("Campaign identity changed; use a new prefix")
    if (root / "result.json").exists():
        return json.loads((root / "result.json").read_text())
    write_json(old, manifest)
    control = controller or {"code": manifest["code"], "previous_call_id": None}
    write_json(root / "controllers" / f"{control['code']['sha256']}.json", control)
    validate_assets(config, Path("/assets"))
    runs.commit()
    completed = {}
    for stage in stages:
        reference_name = source_name(stage)
        source, source_metadata = stage_source(stage, completed, config)
        record_path = root / stage["name"] / "launch.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else None
        if record is None:
            write_json(
                record_path, {"state": "prepared", "identity": manifest["identity"]}
            )
            runs.commit()
            placement = specs[stage["name"]].placement
            spec = specs[stage["name"]]
            adapter = replace(
                spec.adapter,
                stage=stage,
                source=source,
                source_metadata=source_metadata,
            )
            spec = replace(spec, adapter=adapter)
            call = functions[placement.name].spawn(spec)
            record = {"state": "submitted", "call_id": call.object_id}
            write_json(record_path, record)
            runs.commit()
        elif record["state"] == "prepared":
            raise RuntimeError(
                "Uncertain prepared submission: inspect Modal before any retry; no duplicate submitted"
            )
        if record["state"] != "complete":
            modal.FunctionCall.from_id(record["call_id"]).get()
            runs.reload()
            # The leader's return can precede other nodes' final Volume commits.
            # Wait for complete evidence from every node before publishing a stage.
            deadline = time.monotonic() + 600
            while True:
                runs.reload()
                try:
                    directory = completed_attempt(
                        root / stage["name"], stage, manifest["identity"]
                    )
                    checkpoint_proof(
                        directory / "checkpoint" / f"step-{config['steps']}"
                    )
                    if reference_name is None:
                        checkpoint_proof(
                            directory
                            / "checkpoint"
                            / f"step-{config['checkpoint_step']}"
                        )
                    break
                except (ValueError, FileNotFoundError, EOFError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(5)
            trace = read_trace(Path(directory), stage, manifest["identity"])
            proof = {
                "final": checkpoint_proof(
                    Path(directory) / "checkpoint" / f"step-{config['steps']}"
                )
            }
            if reference_name is None:
                proof["source"] = checkpoint_proof(
                    Path(directory) / "checkpoint" / f"step-{config['checkpoint_step']}"
                )
            elif any(row["metadata"] != source_metadata for row in trace["loads"]):
                raise ValueError(
                    "Resume loaded metadata that differs from the verified baseline"
                )
            write_json(Path(directory) / "checkpoint-proof.json", proof)
            record.update(state="complete", directory=str(directory), proof=proof)
            write_json(record_path, record)
            runs.commit()
        candidate_config = json.loads(
            (Path(record["directory"]) / "observed.json").read_text()
        )
        candidate_contract = validate_observed(candidate_config, config, stage)
        if reference_name is not None:
            reference = read_trace(
                Path(completed[reference_name]["directory"]),
                next(s for s in stages if s["name"] == reference_name),
                manifest["identity"],
            )
            candidate = read_trace(
                Path(record["directory"]), stage, manifest["identity"]
            )
            reference_config = json.loads(
                (
                    Path(completed[reference_name]["directory"]) / "observed.json"
                ).read_text()
            )
            if (
                validate_observed(reference_config, config, reference["stage"])
                != candidate_contract
            ):
                raise ValueError(
                    "Reference and resume observed numerical/data settings differ"
                )
            comparison = compare(reference, candidate)
            comparison.update(outcome(comparison, stage_arm(stage)))
            write_json(Path(record["directory"]) / "comparison.json", comparison)
            record["comparison"] = {
                key: value for key, value in comparison.items() if key != "steps"
            }
            write_json(root / stage["name"] / "launch.json", record)
            runs.commit()
        completed[stage["name"]] = record
    write_json(root / "result.json", completed)
    runs.commit()
    registry.put("result", completed)
    return completed


def training_image(config: dict):
    """The same locked image for input preparation and GPU training."""
    import modal
    from . import sources

    local = sources.image_sources(config)
    requirements = " --no-emit-package zephon" if "zephon" in local else ""
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.3-devel-ubuntu24.04", add_python="3.12"
        )
        .apt_install("git", "libibverbs1", "ibverbs-providers", "rdma-core")
        .pip_install("uv==0.11.8")
        .add_local_file(
            ROOT / "environments/training/pyproject.toml",
            "/opt/training/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            ROOT / "environments/training/uv.lock", "/opt/training/uv.lock", copy=True
        )
        .run_commands(
            "uv export --project /opt/training --locked --no-emit-project --no-hashes"
            + requirements
            + " --output-file /opt/training/requirements.txt",
            "uv pip install --system -r /opt/training/requirements.txt --extra-index-url "
            "https://download.pytorch.org/whl/nightly/cu130 --index-strategy unsafe-best-match",
        )
    )
    for name, entry in local.items():
        source, ignore = sources.image_directory(entry)
        image = image.add_local_dir(
            source, f"/opt/local_sources/{name}", ignore=ignore, copy=True
        )
        image = image.run_commands(
            sources.image_metadata(Path("/opt/local_sources") / name, entry)
        )
        sources.verify(entry)
    if "zephon" in local:
        image = image.run_commands(
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ZEPHON=0+unknown uv pip install --system --no-deps -e /opt/local_sources/zephon"
        )
    if "torchtitan" in local:
        image = image.run_commands(
            "ln -s /opt/local_sources/torchtitan /opt/torchtitan"
        )
    else:
        image = image.run_commands(
            "git clone https://github.com/datologyai/torchtitan-zephon.git /opt/torchtitan",
            f"git -C /opt/torchtitan checkout --detach {config['sources']['torchtitan']}",
        )
    return (
        image.run_commands("uv pip freeze --system > /opt/environment.txt")
        .env(
            {
                "PYTHONPATH": "/root:/opt/torchtitan",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            }
        )
        .add_local_python_source("vldb_experiments", "zephon_modal")
    )


def build_app(manifest: dict):
    """Define the app locally without deploying it or building a training image."""
    import modal
    from . import sources

    sources.use_modal(manifest["config"])
    from zephon_modal import ElasticJob, Placement
    from zephon_modal.zephon import Zephon

    from .adapter import ExperimentAdapter

    config, prefix = manifest["config"], manifest["prefix"]
    runs = modal.Volume.from_name(
        f"{prefix}-training-runs", create_if_missing=True, version=2
    )
    assets = modal.Volume.from_name(
        f"{prefix}-training-assets", create_if_missing=False, version=2
    )
    registry = modal.Dict.from_name(f"{prefix}-training-state", create_if_missing=True)
    image = training_image(config)
    stages = plan(config)
    placements = {
        stage["name"]: Placement(
            f"{config['gpu']}:{stage['gpus_per_node']}", nodes=stage["nodes"]
        )
        for stage in stages
    }
    job = ElasticJob(
        name=f"{prefix}-training",
        image=image,
        adapter=ExperimentAdapter(
            module="vldb_experiments.training.runtime",
            config="get_config",
            manifest=manifest,
            stage=stages[0],
            checkpoints=None,
        ),
        placements=list(
            {placement.name: placement for placement in placements.values()}.values()
        ),
        data=Zephon(canonical_replicas=config["canonical_replicas"]),
        runs_volume=runs,
        volumes={"/assets": assets.with_mount_options(read_only=True)},
        cpu=32,
        memory=131072,
        timeout=config["timeout_seconds"],
    )
    app = job.app
    functions = dict(job.functions)
    specs = {
        stage["name"]: job.run_spec(
            stage["name"], config["steps"], placements[stage["name"]], config["seed"]
        )
        for stage in stages
    }

    coordinator = app.function(
        name="campaign",
        image=image,
        cpu=2,
        memory=8192,
        timeout=86400,
        nonpreemptible=True,
        max_containers=1,
        volumes={"/runs": runs, "/assets": assets.with_mount_options(read_only=True)},
    )(coordinate)
    return app, coordinator, runs, registry, {"functions": functions, "specs": specs}
