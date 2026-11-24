"""Strict completed-update evidence, independent of the GPU training framework."""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import Counter
from pathlib import Path

from .protocol import (
    digest,
    experiment_scale,
    figure1_geometry,
    file_digest,
    loader_policy,
    plan,
    rebuild_loader,
    recipe_status,
    source_name,
    stage_arm,
)


def read_trace(directory: Path, stage: dict, identity: str) -> dict:
    """Reject partial ranks, duplicate events, missing updates and invalid loss precision."""
    files = sorted((directory / "diagnostics").glob("rank-*.jsonl"))
    if {p.name for p in files} != {f"rank-{r}.jsonl" for r in range(stage["dp"])}:
        raise ValueError(f"{stage['name']}: incomplete rank coverage")
    steps = range(stage["start_step"] + 1, stage["end_step"] + 1)
    inputs, losses, parameters, loads = {}, {}, {}, []
    for rank in range(stage["dp"]):
        path = directory / "diagnostics" / f"rank-{rank}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        expected_events = (
            ["manifest", "loaded"]
            + (["fast_forward"] if rebuild_loader(stage) else [])
            + ["update"] * len(steps)
            + ["complete"]
        )
        if [row.get("event") for row in rows] != expected_events:
            raise ValueError(f"Incomplete or malformed event sequence: {path}")
        if not rows or rows[0] != {
            "event": "manifest",
            "rank": rank,
            "stage": stage,
            "identity": identity,
            "schema": 1,
        }:
            raise ValueError(f"Invalid manifest: {path}")
        if rows[-1] != {"event": "complete", "step": stage["end_step"]}:
            raise ValueError(f"Missing successful footer: {path}")
        if any(
            row["event"]
            not in {"manifest", "loaded", "fast_forward", "update", "complete"}
            for row in rows
        ):
            raise ValueError(f"Unknown event: {path}")
        if sum(row["event"] == "loaded" for row in rows) != 1:
            raise ValueError("Exactly one native checkpoint load must be recorded")
        loaded = next(row for row in rows if row["event"] == "loaded")
        has_source = bool(stage["start_step"])
        if loaded["step"] != stage["start_step"] or loaded["loaded"] != has_source:
            raise ValueError("Loaded training step differs from declared source")
        if bool(loaded["metadata"]) != has_source:
            raise ValueError(
                "Native checkpoint metadata is required exactly for resumes"
            )
        if loaded.get("loader_policy") != loader_policy(stage):
            raise ValueError("Native loader policy differs from the declared arm")
        if rebuild_loader(stage) and rows[2] != {
            "event": "fast_forward",
            "step": stage["start_step"],
            "microbatches": stage["start_step"] * stage["accumulation_steps"],
        }:
            raise ValueError(
                "Control loader fast-forward count differs from its history"
            )
        if not isinstance(loaded["parameters"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", loaded["parameters"]
        ):
            raise ValueError("Invalid initial parameter digest")
        if loaded["metadata"] is not None and not re.fullmatch(
            r"[0-9a-f]{64}", loaded["metadata"]
        ):
            raise ValueError("Invalid native checkpoint metadata digest")
        loads.append(loaded)
        updates = [row for row in rows if row["event"] == "update"]
        if [row["step"] for row in updates] != list(steps):
            raise ValueError(f"Non-contiguous completed optimizer updates: {path}")
        for row in updates:
            step = row["step"]
            groups = row["microbatches"]
            if len(groups) != stage["accumulation_steps"] or any(
                len(g) != stage["local_batch"] for g in groups
            ):
                raise ValueError("Incomplete microbatch geometry")
            for group in groups:
                if any(
                    not isinstance(h, str)
                    or len(h) != 64
                    or any(c not in "0123456789abcdef" for c in h)
                    for h in group
                ):
                    raise ValueError("Invalid sample digest")
            inputs.setdefault(step, []).extend(groups)
            if rank == 0:
                value = row["loss"]
                if (
                    not isinstance(value, float)
                    or not math.isfinite(value)
                    or value.hex() != row["loss_hex"]
                ):
                    raise ValueError("Loss must retain the full FP64 value")
                losses[step], parameters[step] = value, row["parameters"]
                if not isinstance(parameters[step], str) or not re.fullmatch(
                    r"[0-9a-f]{64}", parameters[step]
                ):
                    raise ValueError("Missing post-update logical parameter digest")
    if len({row["parameters"] for row in loads}) != 1:
        raise ValueError("Initial parameters differ across ranks")
    if len({row["metadata"] for row in loads}) != 1:
        raise ValueError("Ranks loaded different native checkpoint metadata")
    return {
        "stage": stage,
        "loads": loads,
        "inputs": inputs,
        "losses": losses,
        "parameters": parameters,
    }


def checkpoint_proof(path: Path) -> dict:
    """Use native DCP metadata to require every referenced shard at full length."""
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    try:
        metadata = FileSystemReader(path).read_metadata()
    except (EOFError, pickle.UnpicklingError) as error:
        raise ValueError(
            "Native checkpoint metadata is truncated or incomplete"
        ) from error
    needed = {}
    for storage in (metadata.storage_data or {}).values():
        relative = Path(storage.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid native checkpoint shard path")
        needed[str(relative)] = max(
            needed.get(str(relative), 0), storage.offset + storage.length
        )
    if not needed:
        raise ValueError("Native checkpoint has no data shards")
    for name, size in needed.items():
        shard = path / name
        if size <= 0 or not shard.is_file() or shard.stat().st_size < size:
            raise ValueError(f"Missing or truncated native checkpoint shard: {shard}")
    keys = list(metadata.state_dict_metadata)
    auxiliary = ("optimizer", "lr_scheduler", "dataloader", "train_state")
    # TorchTitan's native writer flattens ModelWrapper.state_dict() into the
    # checkpoint root, retaining the four auxiliary states under their names.
    # A DCP produced directly from a nested state dict may retain "model" too.
    has_model = any(k == "model" or k.startswith("model.") for k in keys) or any(
        isinstance(value, TensorStorageMetadata)
        and key.split(".", 1)[0] not in auxiliary
        for key, value in metadata.state_dict_metadata.items()
    )
    if not has_model or any(
        not any(k == prefix or k.startswith(prefix + ".") for k in keys)
        for prefix in auxiliary
    ):
        raise ValueError(
            "Checkpoint must include model, optimizer, scheduler, loader and training state"
        )
    return {
        "metadata_sha256": file_digest(path / ".metadata"),
        "shards": {name: (path / name).stat().st_size for name in sorted(needed)},
    }


def compare(reference: dict, candidate: dict) -> dict:
    if stage_arm(reference["stage"]) != stage_arm(candidate["stage"]):
        raise ValueError("Compare each resume with its own loader-arm reference")
    if source_name(candidate["stage"]) != reference["stage"]["name"]:
        raise ValueError("Resume reference differs from its declared checkpoint source")
    start = candidate["stage"]["start_step"]
    expected = reference["parameters"][start]
    if any(row["parameters"] != expected for row in candidate["loads"]):
        raise ValueError(
            "Resume initial parameters differ from baseline checkpoint probe"
        )
    rows = []
    for step in candidate["losses"]:
        ref, got = reference["inputs"][step], candidate["inputs"][step]
        rows.append(
            {
                "step": step,
                "samples_match": Counter(h for g in ref for h in g)
                == Counter(h for g in got for h in g),
                "microbatch_groups_match": Counter(tuple(g) for g in ref)
                == Counter(tuple(g) for g in got),
                "loss_match": reference["losses"][step].hex()
                == candidate["losses"][step].hex(),
                "parameters_match": reference["parameters"][step]
                == candidate["parameters"][step],
                "reference_loss": reference["losses"][step],
                "resume_loss": candidate["losses"][step],
                "reference_parameters": reference["parameters"][step],
                "resume_parameters": candidate["parameters"][step],
                "reference_input_sha256": digest(sorted(ref)),
                "resume_input_sha256": digest(sorted(got)),
            }
        )
    fields = [
        "samples_match",
        "microbatch_groups_match",
        "loss_match",
        "parameters_match",
    ]
    return {
        "exact": all(row[field] for row in rows for field in fields),
        "updates": len(rows),
        "first_mismatch": next(
            (row for row in rows if not all(row[f] for f in fields)), None
        ),
        "steps": rows,
    }


def outcome(result: dict, arm: str) -> dict:
    """Separate valid measured divergence from an invalid or incomplete execution."""
    fields = (
        "samples_match",
        "microbatch_groups_match",
        "loss_match",
        "parameters_match",
    )
    changed = {
        field: any(not row[field] for row in result["steps"]) for field in fields
    }
    return {
        "arm": arm,
        "expected": "divergence" if arm == "control" else "exact",
        "observed_changes": changed,
        "expected_outcome_met": all(changed.values())
        if arm == "control"
        else result["exact"],
    }


def observed_contract(observed: dict, stage: dict | None = None) -> dict:
    """Ignore only process addresses in Python function reprs and planned launch fields."""
    encoded = json.dumps(observed["config"], sort_keys=True)
    encoded = re.sub(r"(<function \S+) at 0x[0-9a-fA-F]+>", r"\1>", encoded)
    config = json.loads(encoded)
    config.pop("dump_folder", None)
    config.get("parallelism", {}).pop("data_parallel_replicate_degree", None)
    for key in ("folder", "initial_load_path"):
        config.get("checkpointer", {}).pop(key, None)
    if stage is not None:
        checkpointer = config.get("checkpointer", {})
        expected = ["dataloader"] if rebuild_loader(stage) else []
        if checkpointer.get("exclude_from_loading", []) != expected:
            raise ValueError("Observed checkpoint loader policy differs from its arm")
        checkpointer.pop("exclude_from_loading", None)
        if checkpointer.get("initial_load_model_only") is not False:
            raise ValueError(
                "Observed checkpoint loading must preserve full training state"
            )
        checkpointer.pop("initial_load_model_only", None)
    return config


def validate_observed(observed: dict, config: dict, stage: dict) -> dict:
    from . import sources

    sources.validate_observed(observed, config)
    numerical = json.loads(json.dumps(observed["config"]))
    if (
        numerical["precision"]
        != {"dtype": "float32", "param_dtype": "bfloat16", "reduce_dtype": "float64"}
        or numerical["loss_reduce_dtype"] != "float64"
        or numerical["dataloader"]["canonical_replicas"]
        != (config["canonical_replicas"] if stage_arm(stage) == "zephon" else None)
        or observed["torch"] != config["sources"]["torch"]
    ):
        raise ValueError(
            "Observed trainer differs from the pinned numerical/loader contract"
        )
    if config.get("schema") != 2:
        raise ValueError("Training campaigns require schema 2")
    model = numerical.get("model", {})
    layers = model.get("layers", [])
    expected_model = {
        "dim": 2048,
        "vocab_size": 128256,
        "enable_weight_tying": True,
        "max_context_length": config["sequence_length"],
    }
    if (
        any(model.get(key) != value for key, value in expected_model.items())
        or len(layers) != 16
    ):
        raise ValueError(
            "Observed model is not the declared tied Llama3 1B architecture"
        )
    if any(
        layer.get("attention", {}).get("n_heads") != 32
        or layer.get("attention", {}).get("n_kv_heads") != 8
        for layer in layers
    ):
        raise ValueError("Observed model attention geometry differs from Llama3 1B")
    expected_training = {
        "num_tokens_per_microbatch_per_dp_rank": config["local_batch"]
        * config["sequence_length"],
        "num_tokens_per_train_step": config["sequences_per_step"]
        * config["sequence_length"],
        "steps": config["steps"],
        "disable_accelerator_graphs": True,
    }
    if any(
        numerical.get("training", {}).get(k) != v for k, v in expected_training.items()
    ):
        raise ValueError(
            "Observed batch, sequence length or training endpoint differs from the recipe"
        )
    optim = numerical.get("optim", {})
    optimizers = optim.get("optimizer", {}).get("optimizers", [])
    expected_optimizer = {
        k: v for k, v in config["optimizer"].items() if k != "max_norm"
    }
    if (
        len(optimizers) != 1
        or any(optimizers[0].get(k) != v for k, v in expected_optimizer.items())
        or optimizers[0].get("fused") is not True
        or optim.get("max_norm") != config["optimizer"]["max_norm"]
    ):
        raise ValueError("Observed optimizer differs from the declared recipe")
    if any(
        optim.get("lr_scheduler", {}).get(k) != v
        for k, v in config["lr_scheduler"].items()
    ):
        raise ValueError("Observed LR schedule differs from the declared recipe")
    if (
        numerical.get("debug", {}).get("seed") != config["seed"]
        or numerical.get("debug", {}).get("deterministic") is not True
    ):
        raise ValueError(
            "Observed model seed or determinism controls differ from the recipe"
        )
    if numerical["dataloader"].get("seed") != config["data_seed"] or any(
        numerical["dataloader"].get(k) != v for k, v in config["loader"].items()
    ):
        raise ValueError("Observed data seed or loader settings differ from the recipe")
    if numerical.get("create_seed_checkpoint") is not False:
        raise ValueError("Observed seed checkpoint stage differs from its role")
    return observed_contract(observed, stage)


def report(run: Path) -> dict:
    manifest = json.loads((run / "campaign.json").read_text())
    expected_assets = {
        f"{folder}/{Path(name).name}": sha
        for key, folder in (("dataset", "data"), ("tokenizer", "tokenizer"))
        for name, sha in manifest["config"][key]["files"].items()
    }
    if json.loads((run / "assets.json").read_text()) != expected_assets:
        raise ValueError(
            "Prepared public assets differ from the pinned data/tokenizer manifest"
        )
    stages = plan(manifest["config"])
    traces = {
        stage["name"]: read_trace(run / stage["name"], stage, manifest["identity"])
        for stage in stages
    }
    numerical_references = {}
    arm_reference = None
    for stage in stages:
        directory = run / stage["name"]
        proof = json.loads((directory / "checkpoint-proof.json").read_text())
        if (
            not proof["final"]["shards"]
            or file_digest(directory / "final.metadata")
            != proof["final"]["metadata_sha256"]
        ):
            raise ValueError("Final native checkpoint proof is missing or changed")
        observed = json.loads((directory / "observed.json").read_text())
        contract = validate_observed(observed, manifest["config"], stage)
        arm = stage_arm(stage)
        if arm not in numerical_references:
            numerical_references[arm] = contract
        elif contract != numerical_references[arm]:
            raise ValueError(
                "Observed numerical/data settings differ between reference and resume"
            )
        # Between arms only the lane policy may change. Equality claims remain
        # strictly within an arm; this checks identical numerical/data recipes.
        common = json.loads(json.dumps(contract))
        common["dataloader"].pop("canonical_replicas")
        if arm_reference is None:
            arm_reference = common
        elif common != arm_reference:
            raise ValueError("Loader arms differ in numerical/data settings")
    baselines = {s["name"]: s for s in stages if source_name(s) is None}
    metadata = {}
    for name in baselines:
        directory = run / name
        proof = json.loads((directory / "checkpoint-proof.json").read_text())["source"]
        if (
            not proof["shards"]
            or file_digest(directory / "source.metadata") != proof["metadata_sha256"]
        ):
            raise ValueError("Source native checkpoint metadata changed")
        metadata[name] = proof["metadata_sha256"]
    if len({traces[name]["loads"][0]["parameters"] for name in baselines}) != 1:
        raise ValueError("Loader arms did not start from the same model initialization")
    comparisons = {}
    for stage in stages:
        source = source_name(stage)
        if source is None:
            continue
        trace = traces[stage["name"]]
        if any(load["metadata"] != metadata[source] for load in trace["loads"]):
            raise ValueError(
                "Resume did not load the verified baseline checkpoint metadata"
            )
        result = compare(traces[source], trace)
        result.update(outcome(result, stage_arm(stage)))
        comparisons[stage["name"]] = result
    summary = {
        "schema": 1,
        "identity": manifest["identity"],
        "smoke": manifest["config"]["smoke"],
        "exact": all(value["exact"] for value in comparisons.values()),
        "complete": True,
        "experiment_scale": experiment_scale(manifest["config"]),
        "arms": list(dict.fromkeys(stage_arm(s) for s in stages)),
        "expected_outcomes_met": all(
            value["expected_outcome_met"] for value in comparisons.values()
        ),
        "paper_protocol_match": figure1_geometry(manifest["config"])
        and recipe_status(manifest["config"])["paper_training_settings_match"]
        and {stage_arm(s) for s in stages} == {"control", "zephon"},
        "reproduction": recipe_status(manifest["config"]),
        "comparisons": comparisons,
    }
    return summary


def combine(runs: list[Path], output: Path, *, save_binding: bool = True) -> dict:
    """Validate complementary completed campaigns without changing their evidence."""
    output = output.resolve()
    runs = [run.resolve() for run in runs]
    if output in runs or any((run / "combined.json").exists() for run in runs):
        raise ValueError("Combine original campaigns into a separate output directory")
    manifests = [json.loads((run / "campaign.json").read_text()) for run in runs]
    common_config = None
    common_observed = None
    initial_parameters = set()
    seen = set()
    sources = []
    for run, manifest in zip(runs, manifests, strict=True):
        config = dict(manifest["config"])
        config.pop("arms", None)
        if common_config is None:
            common_config = config
        elif common_config != config:
            raise ValueError(
                "Combined campaigns have different data/model/numerical recipes"
            )
        stages = plan(manifest["config"])
        arms = {stage_arm(stage) for stage in stages}
        if arms & seen:
            raise ValueError("Combined campaigns duplicate a loader arm")
        seen.update(arms)
        for stage in stages:
            if source_name(stage) is not None:
                continue
            directory = run / stage["name"]
            observed = validate_observed(
                json.loads((directory / "observed.json").read_text()),
                manifest["config"],
                stage,
            )
            observed["dataloader"].pop("canonical_replicas")
            # Independent campaigns must have distinct checkpoint aggregation
            # IDs. Every data and numerical setting still has to match.
            observed["dataloader"].pop("run_id", None)
            if common_observed is None:
                common_observed = observed
            elif common_observed != observed:
                raise ValueError(
                    "Combined campaigns differ in observed numerical/data settings"
                )
            baseline = read_trace(directory, stage, manifest["identity"])
            initial_parameters.add(baseline["loads"][0]["parameters"])
        sources.append(
            {
                "path": os.path.relpath(run, output),
                "identity": manifest["identity"],
                "arms": sorted(arms),
            }
        )
    if seen != {"control", "zephon"} or len(initial_parameters) != 1:
        raise ValueError(
            "Figure 1 requires both loader arms with identical model initialization"
        )
    binding = {"schema": 1, "sources": sources}
    binding["identity"] = digest(binding)
    path = output / "combined.json"
    if path.exists() and json.loads(path.read_text()) != binding:
        raise ValueError(
            "Combined report already belongs to different source campaigns"
        )
    comparisons = {}
    summaries = []
    for run, manifest in zip(runs, manifests, strict=True):
        summary = report(run)
        summaries.append(summary)
        for name, comparison in summary["comparisons"].items():
            stage = next(
                stage for stage in plan(manifest["config"]) if stage["name"] == name
            )
            key = f"{stage_arm(stage)}-resume-dp{stage['dp']}"
            comparisons[key] = comparison | {
                "campaign_identity": manifest["identity"],
                "stage": name,
            }
    summary = {
        "schema": 1,
        "identity": binding["identity"],
        "sources": sources,
        "complete": True,
        "arms": sorted(seen),
        "experiment_scale": experiment_scale(manifests[0]["config"]),
        "comparisons": comparisons,
        "exact": all(item["exact"] for item in summaries),
        "expected_outcomes_met": all(
            item["expected_outcomes_met"] for item in summaries
        ),
        "paper_protocol_match": all(
            figure1_geometry(manifest["config"])
            and recipe_status(manifest["config"])["paper_training_settings_match"]
            for manifest in manifests
        ),
        "reproduction": recipe_status(manifests[0]["config"]),
    }
    if save_binding:
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(binding, indent=2) + "\n")
    return summary


def validate_report(run: Path) -> dict:
    path = run / "combined.json"
    if path.exists():
        sources = json.loads(path.read_text())["sources"]
        return combine(
            [run / source["path"] for source in sources], run, save_binding=False
        )
    return report(run)


def summary(run):
    from .. import core
    from ..presentation import document, table
    from . import protocol

    summary = validate_report(run)
    if (run / "combined.json").exists():
        binding = core.read_json(run / "combined.json")
        config = core.read_json(run / binding["sources"][0]["path"] / "campaign.json")[
            "config"
        ]
    else:
        config = core.read_json(run / "campaign.json")["config"]
    rows = []
    extra = []
    for name, comparison in summary["comparisons"].items():
        steps = comparison["steps"]
        label = (
            "Zephon"
            if "zephon" in name
            or (len(summary["arms"]) == 1 and summary["arms"][0] == "zephon")
            else "Standard loader"
        )
        rows.append(
            [
                label,
                name.rsplit("dp", 1)[-1],
                len(steps),
                sum(r["microbatch_groups_match"] for r in steps),
                sum(r["loss_match"] for r in steps),
            ]
        )
        extra.append(
            [
                label,
                name.rsplit("dp", 1)[-1],
                sum(r["samples_match"] for r in steps),
                sum(r["parameters_match"] for r in steps),
            ]
        )
    rows.sort(key=lambda r: (r[0] == "Zephon", int(r[1])))
    extra.sort(key=lambda r: (r[0] == "Zephon", int(r[1])))
    complete = set(summary["arms"]) == {"control", "zephon"}
    return document(
        "Elastic determinism — Figure 1 checkpoint-resize experiment",
        [
            f"Public recipe: {config['model']}; checkpoint at step {config['checkpoint_step']}; baseline DP{config['baseline_dp']}; continuations {config['resume_dps']}.",
            f"Validation scope: {protocol.recipe_status(config)['scope']}; checkpoint-resize behavior for the declared assets and trainer.",
            *protocol.recipe_status(config)["scope_notes"],
            "Each continuation is compared with its own uninterrupted loader-arm baseline. Matching counts are optimizer updates.",
        ],
        [
            table(
                "Global batch identity and training loss",
                ["Loader", "DP", "Updates", "Batch matches", "Exact loss matches"],
                rows,
            )
        ],
        [
            table(
                "Additional correctness checks",
                ["Loader", "DP", "Sample-multiset matches", "Parameter matches"],
                extra,
            )
        ],
        status=(
            "Complete selected training experiment"
            if complete
            else "Incomplete Figure 1 reproduction: only one loader arm is present"
        )
        + "; scale: "
        + protocol.experiment_scale(config),
    )
