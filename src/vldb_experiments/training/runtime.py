"""Experiment components for the pinned public TorchTitan API (training image only)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine

from .protocol import (
    file_digest,
    loader_iterator,
    loader_policy,
    rebuild_loader,
    stage_arm,
)


def tensor_bytes(value: torch.Tensor) -> bytes:
    if hasattr(value, "to_local"):
        local = value.to_local()
        if tuple(local.shape) != tuple(value.shape):
            raise ValueError("Logical parameter hashing requires replicated tensors")
        value = local
    value = value.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode()
    return (
        len(header).to_bytes(8, "big")
        + header
        + value.reshape(-1).view(torch.uint8).numpy().tobytes()
    )


def parameter_digest(parts: list) -> str:
    if len(parts) != 1:
        raise ValueError("This experiment requires one fully replicated model part")
    result = hashlib.sha256(b"logical-parameters-v1")
    values = {}
    for name, value in parts[0].named_parameters(remove_duplicate=False):
        name = ".".join(p for p in name.split(".") if p != "_orig_mod")
        if name in values:
            raise ValueError("Duplicate normalized logical parameter name")
        values[name] = value
    if not values:
        raise ValueError("Model has no parameters")
    for name, value in sorted(values.items()):
        result.update(name.encode() + b"\0" + tensor_bytes(value))
    return result.hexdigest()


class Evidence:
    def __init__(self, settings: dict):
        Path(settings["directory"]).mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.stage = settings["stage"]
        self.rank = dist.get_rank()
        if dist.get_world_size() != self.stage["dp"]:
            raise ValueError("Allocated DP differs from the experiment plan")
        path = (
            Path(settings["trace_directory"])
            / f"{settings['attempt']}.rank-{self.rank}.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("x")
        self.emit(
            event="manifest",
            schema=1,
            identity=settings["identity"],
            stage=self.stage,
            rank=self.rank,
        )
        self.microbatches = []
        self.loss = None
        self.closed = False

    def emit(self, **row):
        self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()

    def inputs(self, microbatch):
        fields = microbatch.as_input_dict() | {
            "labels": microbatch.labels,
            "loss_token_counts": microbatch.loss_token_counts,
            "routing_token_counts": microbatch.routing_token_counts,
        }
        batch = self.stage["local_batch"]
        length = self.settings["config"]["sequence_length"]
        if microbatch.labels.numel() != batch * length:
            raise ValueError("Observed labels differ from the declared local batch")
        samples = []
        for index in range(batch):
            result = hashlib.sha256(b"model-input-v1")
            for name, value in sorted(fields.items()):
                result.update(name.encode() + b"\0")
                if isinstance(value, torch.Tensor):
                    if value.numel() == batch * length:
                        value = value.reshape(batch, length)[index]
                    elif value.numel() != 1:
                        raise ValueError(
                            f"Unsupported microbatch field geometry: {name}"
                        )
                    result.update(tensor_bytes(value.reshape(-1)))
                elif value is None or isinstance(value, (str, int, float, bool)):
                    result.update(json.dumps(value, allow_nan=False).encode())
                else:
                    raise ValueError(f"Unsupported model input field: {name}")
            samples.append(result.hexdigest())
        self.microbatches.append(samples)


class Checkpointer(CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        pass

    def load(self, step: int = -1) -> bool:
        loaded = super().load(step)
        trainer = self.experiment_trainer
        evidence = trainer.evidence
        expected = evidence.stage["start_step"]
        if loaded != bool(expected) or trainer.engine.num_completed_steps != expected:
            raise ValueError(
                "Native full-state checkpoint did not restore the expected training step"
            )
        source = evidence.settings["source"]
        metadata = file_digest(Path(source) / ".metadata") if source else None
        if metadata != evidence.settings["source_metadata"]:
            raise ValueError("Native checkpoint metadata changed before loading")
        evidence.emit(
            event="loaded",
            step=expected,
            loaded=loaded,
            metadata=metadata,
            parameters=parameter_digest(trainer.engine.model_parts),
            **(
                {"loader_policy": loader_policy(evidence.stage)}
                if "arm" in evidence.stage
                else {}
            ),
        )
        return loaded


class Metrics(MetricsProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(MetricsProcessor.Config):
        pass

    def log(
        self,
        step,
        global_avg_loss,
        global_max_loss,
        grad_norm,
        extra_metrics=None,
        *,
        num_flops,
    ):
        self.evidence.loss = (step, float(global_avg_loss))
        super().log(
            step,
            global_avg_loss,
            global_max_loss,
            grad_norm,
            extra_metrics,
            num_flops=num_flops,
        )


class Engine(TrainingEngine):
    """Retain FP64 gradients locally; synchronize only the last microbatch."""

    def forward_backward(self, **kwargs):
        self.group = 0
        self.groups = len(kwargs["microbatch_groups"])
        return super().forward_backward(**kwargs)

    def _non_pp_forward_backward_microbatch(self, **kwargs):
        self.group += 1
        sync = self.group == self.groups
        root = self.model_parts[0]
        root.set_requires_all_reduce(sync)
        root.set_requires_gradient_sync(sync)
        root.set_is_last_backward(sync)
        root.set_reshard_after_backward(sync)
        return super()._non_pp_forward_backward_microbatch(**kwargs)


class ExperimentTrainer(Trainer):
    engine_cls = Engine

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def __init__(self, config):
        super().__init__(config)
        settings = json.loads(os.environ["ZEPHON_REPRO_STAGE"])
        self.evidence = Evidence(settings)
        self.metrics_processor.evidence = self.evidence
        self.engine.checkpointer.experiment_trainer = self
        self.success = False
        if self.evidence.rank == 0:
            observed = {
                "config": config.to_dict(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "torchtitan_commit": (
                    settings["config"]["local_sources"]["torchtitan"]["state"].get(
                        "revision"
                    )
                    if "torchtitan" in settings["config"].get("local_sources", {})
                    else subprocess.check_output(
                        ["git", "-C", "/opt/torchtitan", "rev-parse", "HEAD"], text=True
                    ).strip()
                ),
            }
            from importlib.metadata import version

            observed["zephon"] = version("zephon")
            from . import sources

            observed["local_sources"] = sources.verify_installed(settings["config"])
            sources.validate_observed(observed, settings["config"])
            if observed["torch"] != settings["config"]["sources"]["torch"]:
                raise ValueError(
                    "Installed training dependencies differ from the campaign pins"
                )
            observed["environment"] = Path("/opt/environment.txt").read_text()
            (Path(settings["directory"]) / "observed.json").write_text(
                json.dumps(observed, indent=2)
            )

    def microbatch_generator(self, data_iterable):
        iterator = loader_iterator(data_iterable, self.evidence.stage)
        if rebuild_loader(self.evidence.stage):
            self.evidence.emit(
                event="fast_forward",
                step=self.evidence.stage["start_step"],
                microbatches=self.evidence.stage["start_step"]
                * self.evidence.stage["accumulation_steps"],
            )
        for microbatch in super().microbatch_generator(iterator):
            self.evidence.inputs(microbatch)
            yield microbatch

    def train_step(self, data_iterator):
        evidence = self.evidence
        evidence.microbatches = []
        evidence.loss = None
        super().train_step(data_iterator)
        step = self.engine.num_completed_steps
        if len(evidence.microbatches) != evidence.stage["accumulation_steps"]:
            raise ValueError("Missing consumed microbatches")
        row = {"event": "update", "step": step, "microbatches": evidence.microbatches}
        if evidence.rank == 0:
            if evidence.loss is None or evidence.loss[0] != step:
                raise ValueError("No full-precision loss for the completed update")
            row.update(
                loss=evidence.loss[1],
                loss_hex=evidence.loss[1].hex(),
                parameters=parameter_digest(self.engine.model_parts),
            )
        evidence.emit(**row)

    def train(self):
        super().train()
        self.success = (
            self.engine.num_completed_steps == self.evidence.stage["end_step"]
        )
        if not self.success:
            raise ValueError("Training ended before the declared final update")

    def close(self):
        try:
            super().close()
        finally:
            if not self.evidence.closed:
                self.evidence.emit(
                    event="complete" if self.success else "failed",
                    step=self.engine.num_completed_steps,
                )
                self.evidence.stream.close()
                self.evidence.closed = True


def get_config() -> ExperimentTrainer.Config:
    """Configure native tied Llama3 1B with the declared batch, seeds and schedule."""
    from torchtitan.components.loss import CrossEntropyLoss
    from torchtitan.components.optim import (
        AdamW,
        LRSchedulersContainer,
        Optim,
        OptimizersContainer,
    )
    from torchtitan.config import PrecisionConfig, TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.common.config_utils import decoder_vocab_size
    from torchtitan.models.common.linear import Linear
    from torchtitan.models.llama3 import build_model_config
    from torchtitan_recipes.zephon import ZephonDataLoader, ZephonSource

    settings = json.loads(os.environ["ZEPHON_REPRO_STAGE"])
    spec, stage = settings["config"], settings["stage"]
    parallelism = ParallelismConfig(
        data_parallel_replicate_degree=stage["dp"], data_parallel_shard_degree=1
    )
    model = build_model_config(
        "1B",
        seq_len=spec["sequence_length"],
        parallelism=parallelism,
        attn_backend="flex",
    )
    for _, linear, _, _ in model.traverse(Linear.Config):
        linear.inplace_wgrad_accum = False
    config = ExperimentTrainer.Config(
        model=model,
        parallelism=parallelism,
        precision=PrecisionConfig(
            dtype="float32", param_dtype="bfloat16", reduce_dtype="float64"
        ),
        loss_reduce_dtype="float64",
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        hf_assets_path="/assets/tokenizer",
        dataloader=ZephonDataLoader.Config(
            sources=[
                ZephonSource(name="fineweb-edu", path="/assets/data", fmt="parquet")
            ],
            text_field="text",
            seed=spec["data_seed"],
            canonical_replicas=spec["canonical_replicas"]
            if stage_arm(stage) == "zephon"
            else None,
            aggregate_store="torch://",
            run_id=settings["identity"],
            **spec["loader"],
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=spec["local_batch"]
            * spec["sequence_length"],
            num_tokens_per_train_step=spec["sequences_per_step"]
            * spec["sequence_length"],
            steps=spec["steps"],
            disable_accelerator_graphs=True,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*",
                        fused=True,
                        **{
                            key: tuple(value) if key == "betas" else value
                            for key, value in spec["optimizer"].items()
                            if key != "max_norm"
                        },
                    )
                ]
            ),
            lr_scheduler=LRSchedulersContainer.Config(**spec["lr_scheduler"]),
            max_norm=spec["optimizer"]["max_norm"],
        ),
        checkpointer=Checkpointer.Config(
            folder=str(Path(settings["directory"]) / "checkpoint"),
            interval=spec["checkpoint_step"],
            initial_load_path=settings["source"],
            initial_load_model_only=False,
            exclude_from_loading=["dataloader"] if rebuild_loader(stage) else [],
            last_save_model_only=False,
            keep_latest_k=0,
            async_mode="disabled",
        ),
        metrics=Metrics.Config(log_freq=1, loggers=[]),
        activation_remat=None,
        validator=None,
        dump_folder=settings["directory"],
    )
    config.debug.seed = spec["seed"]
    config.debug.deterministic = True
    config.debug.save_config_file = None
    return config
