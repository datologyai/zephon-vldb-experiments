"""The fixed training experiment; planning has no cloud or torch dependencies."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from pathlib import Path

from .. import ROOT


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path: Path | None = None, smoke: bool = False, small: bool = False) -> dict:
    if path is not None and small:
        raise ValueError("Select --small or --config, not both")
    default = "small.json" if small else "experiment.json"
    config = json.loads((path or ROOT / "configs/training" / default).read_text())
    arms = config.get("arms")
    if not arms or len(set(arms)) != len(arms) or set(arms) - {"control", "zephon"}:
        raise ValueError("Select distinct control/zephon loader arms")
    if smoke:
        config.update(
            steps=4,
            checkpoint_step=2,
            baseline_dp=2,
            resume_dps=[1],
            canonical_replicas=2,
            sequences_per_step=2,
            sequence_length=128,
            gpus_per_node=2,
            local_batch=1,
        )
    if config.get("schema") != 2 or config["model"] != "Llama3-1B-random-init":
        raise ValueError(
            "Training campaigns require the schema-2 native Llama3-1B recipe"
        )
    for key in ("data_seed", "local_batch", "optimizer", "lr_scheduler", "loader"):
        if key not in config:
            raise ValueError(f"Missing declared training setting: {key}")
    if not isinstance(config["local_batch"], int) or config["local_batch"] < 1:
        raise ValueError("Local batch must be a positive integer")
    if not 0 < config["checkpoint_step"] < config["steps"]:
        raise ValueError("Checkpoint must precede the final optimizer step")
    if not config["resume_dps"]:
        raise ValueError("At least one checkpoint continuation is required")
    for key in ("torchtitan", "zephon_modal"):
        if not re.fullmatch(r"[0-9a-f]{40}", config["sources"][key]):
            raise ValueError(
                "Training source dependencies require immutable commit hashes"
            )
    widths = [config["baseline_dp"], *config["resume_dps"]]
    if len(set(widths)) != len(widths) or any(dp < 1 for dp in widths):
        raise ValueError("DP degrees must be positive and distinct")
    for dp in widths:
        if config["canonical_replicas"] % dp or config["sequences_per_step"] % (
            dp * config["local_batch"]
        ):
            raise ValueError("Every DP degree must divide the lanes and global batch")
        if dp > config["gpus_per_node"] and dp % config["gpus_per_node"]:
            raise ValueError("Multi-node placements must use whole nodes")
    if config["sequences_per_step"] % config["canonical_replicas"]:
        raise ValueError("Every update must consume complete canonical lane windows")
    for source in (config["dataset"], config["tokenizer"]):
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise ValueError("Public assets require immutable Hugging Face revisions")
    packages = {
        p["name"]: p.get("version")
        for p in tomllib.loads((ROOT / "environments/training/uv.lock").read_text())[
            "package"
        ]
    }
    for package in ("torch", "zephon", "modal"):
        if packages.get(package) != config["sources"][package]:
            raise ValueError(
                f"{package} source pin differs from the locked training image"
            )
    return config | {"smoke": smoke}


def plan(config: dict) -> list[dict]:
    """Expand the declared loader arms into baseline and checkpoint continuations."""
    arms = config.get("arms")
    if config.get("schema") != 2 or not arms:
        raise ValueError("Training campaigns require schema 2 and explicit loader arms")
    if len(set(arms)) != len(arms) or set(arms) - {"control", "zephon"}:
        raise ValueError("Select distinct control/zephon loader arms")
    return [
        {
            "name": f"{arm}-{name}",
            "arm": arm,
            "reference": f"{arm}-baseline" if start else None,
            "dp": dp,
            "start_step": start,
            "end_step": config["steps"],
            "nodes": max(1, dp // config["gpus_per_node"]),
            "gpus_per_node": min(dp, config["gpus_per_node"]),
            "accumulation_steps": config["sequences_per_step"]
            // (dp * config["local_batch"]),
            "local_batch": config["local_batch"],
        }
        for arm in arms
        for name, dp, start in [
            ("baseline", config["baseline_dp"], 0),
            *[
                (f"resume-dp{dp}", dp, config["checkpoint_step"])
                for dp in config["resume_dps"]
            ],
        ]
    ]


def recipe_status(config: dict) -> dict:
    """Describe the training settings and public experiment scope."""
    matches = all(
        config.get(key) == value
        for key, value in {
            "model": "Llama3-1B-random-init",
            "seed": 42,
            "data_seed": 1338,
            "local_batch": 7,
            "sequences_per_step": 448,
            "sequence_length": 4096,
            "canonical_replicas": 64,
            "optimizer": {
                "lr": 0.0007908,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
                "max_norm": 1.0,
            },
            "lr_scheduler": {
                "warmup_steps": 400,
                "total_steps": 10899,
                "decay_type": "sqrt",
                "decay_ratio": 0.2,
                "min_lr_factor": 0.05,
            },
            "loader": {
                "chunk_size": 16384,
                "shuffle_shards": True,
                "shuffle_within_shard": True,
                "shuffle_block_size": 163800,
                "token_estimation": True,
                "shuffle_after_pack": True,
                "shuffle_buffer_size": 2048,
                "repeat": True,
                "tokenize_special_tokens": "bos_eos",
                "pack_algorithm": "wrap",
                "fetch_parallelism": 4,
                "tokenize_parallelism": 4,
                "pack_parallelism": 1,
                "mtp_mode": True,
            },
        }.items()
    )
    return {
        "paper_training_settings_match": matches,
        "scope": "public-data reproduction" if matches else "custom training recipe",
        "scope_notes": [
            "Inputs are pinned public FineWeb-Edu sample shards with online tokenization and packing.",
            "Native public Llama3 uses fused projections, scaled RoPE and native initializers.",
            "Compute uses BF16 with FP32 master parameters and FP64 gradient reduction and loss logging.",
            "Baselines use seed 42 and verified matching initial parameters.",
            "Execution uses Modal or supplied hardware; observed GPU models are recorded.",
        ],
    }


def stage_arm(stage: dict) -> str:
    return stage["arm"]


def figure1_geometry(config: dict) -> bool:
    return (
        not config["smoke"]
        and config["baseline_dp"] == 16
        and config["resume_dps"] == [8, 32, 64]
        and config["checkpoint_step"] == 150
        and config["steps"] == 300
    )


def small_geometry(config: dict) -> bool:
    return (
        not config["smoke"]
        and config["baseline_dp"] == 2
        and config["resume_dps"] == [1, 4, 8]
        and config["gpus_per_node"] >= 8
        and config["checkpoint_step"] == 150
        and config["steps"] == 300
    )


def plot_geometry(config: dict) -> bool:
    return figure1_geometry(config) or small_geometry(config)


def experiment_scale(config: dict) -> str:
    if figure1_geometry(config):
        return "full"
    if small_geometry(config):
        return "single-node"
    return "smoke" if config["smoke"] else "custom"


def source_name(stage: dict) -> str | None:
    return stage["reference"] if stage["start_step"] else None


def rebuild_loader(stage: dict) -> bool:
    return stage_arm(stage) == "control" and bool(stage["start_step"])


def loader_policy(stage: dict) -> str:
    if not stage["start_step"]:
        return "fresh"
    return "rebuild_and_fast_forward" if rebuild_loader(stage) else "restore"


def loader_iterator(iterable, stage: dict):
    """Advance a rebuilt topology-coupled loader by the target's local history."""
    iterator = iter(iterable)
    if rebuild_loader(stage):
        for _ in range(stage["start_step"] * stage["accumulation_steps"]):
            try:
                next(iterator)
            except StopIteration as error:
                raise ValueError(
                    "Control loader exhausted during fast-forward"
                ) from error
    return iterator


def code_identity() -> dict:
    """Fingerprint uploaded Python and the training project and lock."""
    files = {
        str(path.relative_to(ROOT)): file_digest(path)
        for path in sorted((ROOT / "src/vldb_experiments/training").rglob("*.py"))
    }
    for name in ("__init__.py", "provenance.py"):
        files["src/vldb_experiments/" + name] = file_digest(
            ROOT / "src/vldb_experiments" / name
        )
    for name in ("pyproject.toml", "uv.lock"):
        path = "environments/training/" + name
        files[path] = file_digest(ROOT / path)
    return {"sha256": digest(files), "files": files}


def validate_prefix(prefix: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9-]{2,39}", prefix):
        raise ValueError(
            "Use a personal, unique prefix: 3-40 lowercase letters, digits or hyphens"
        )
    return prefix
