"""The experiment's native TorchTitan adapter to zephon-modal."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from zephon_modal import Torchtitan
from zephon_modal.context import LaunchContext


@dataclass(frozen=True, kw_only=True)
class ExperimentAdapter(Torchtitan):
    manifest: dict = field(default_factory=dict)
    stage: dict = field(default_factory=dict)
    source: str | None = None
    source_metadata: str | None = None

    def command(self, ctx: LaunchContext) -> list[str]:
        return [
            "-m",
            "torchtitan.train",
            f"--module={self.module}",
            f"--config={self.config}",
            f"--output-dir={ctx.run_dir}/attempts/{ctx.launch_id}",
        ]

    def env(self, ctx: LaunchContext) -> dict[str, str]:
        # Each retry starts from the same immutable source, so an interrupted
        # baseline never quietly becomes a checkpoint-resumed reference.
        settings = {
            "config": self.manifest["config"],
            "identity": self.manifest["identity"],
            "stage": self.stage,
            "directory": f"{ctx.run_dir}/attempts/{ctx.launch_id}",
            "trace_directory": ctx.sample_hash_dir,
            "attempt": ctx.launch_id,
            "source": self.source,
            "source_metadata": self.source_metadata,
        }
        return {"ZEPHON_REPRO_STAGE": json.dumps(settings)}
