"""CPU checks against the actual pinned public trainer, without cloud submission.

Run with the pinned trainer on PYTHONPATH and its CPU dependencies installed.
Normal launcher-only environments skip these native integration checks.
"""

import importlib.util
import json
import os
import re
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

from vldb_experiments.training import protocol, trace

NATIVE_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torchtitan") is not None
)


@unittest.skipUnless(
    NATIVE_AVAILABLE, "Requires the pinned public TorchTitan CPU environment"
)
class NativeRecipeTests(unittest.TestCase):
    def test_all_full_small_and_smoke_configs_match_the_declared_recipe(self):
        from vldb_experiments.training.runtime import get_config

        for config in (
            protocol.load(),
            protocol.load(small=True),
            protocol.load(smoke=True),
        ):
            for stage in protocol.plan(config):
                with self.subTest(
                    stage=stage["name"], smoke=config["smoke"], dp=stage["dp"]
                ):
                    settings = {
                        "config": config,
                        "stage": stage,
                        "identity": "native-cpu-check",
                        "directory": "/tmp/native-check",
                        "source": "/tmp/source" if stage["start_step"] else None,
                    }
                    with patch.dict(
                        os.environ, {"ZEPHON_REPRO_STAGE": json.dumps(settings)}
                    ):
                        native = get_config()
                    observed = {
                        "config": native.to_dict(),
                        "torch": config["sources"]["torch"],
                        "zephon": config["sources"]["zephon"],
                        "torchtitan_commit": config["sources"]["torchtitan"],
                    }
                    trace.validate_observed(observed, config, stage)
                    self.assertEqual(
                        native.training.num_tokens_per_train_step,
                        config["sequences_per_step"] * config["sequence_length"],
                    )
                    self.assertFalse(native.create_seed_checkpoint)

    def test_model_is_the_native_1b_flavor_with_only_the_fp64_linear_setting(self):
        from torchtitan.models.common.linear import Linear
        from torchtitan.models.llama3 import build_model_config

        from vldb_experiments.training.runtime import get_config

        config = protocol.load()
        stage = protocol.plan(config)[0]
        settings = {
            "config": config,
            "stage": stage,
            "identity": "native-cpu-check",
            "directory": "/tmp/native-check",
            "source": None,
        }
        with patch.dict(os.environ, {"ZEPHON_REPRO_STAGE": json.dumps(settings)}):
            native = get_config()
        reference = build_model_config(
            "1B", seq_len=4096, parallelism=native.parallelism, attn_backend="flex"
        )
        for _, linear, _, _ in reference.traverse(Linear.Config):
            linear.inplace_wgrad_accum = False

        def canonical(model):
            return re.sub(
                r" at 0x[0-9a-fA-F]+>",
                ">",
                json.dumps(asdict(model), default=str, sort_keys=True),
            )

        self.assertEqual(canonical(native.model), canonical(reference))

    def test_seven_sequence_hashes_cover_every_sample_without_changing_geometry(self):
        import torch

        from vldb_experiments.training.runtime import Evidence

        tokens = torch.arange(7 * 8).reshape(7, 8)
        batch = SimpleNamespace(
            labels=tokens.clone(),
            loss_token_counts=torch.tensor(56),
            routing_token_counts=torch.tensor(56),
            as_input_dict=lambda: {"input": tokens},
        )
        evidence = object.__new__(Evidence)
        evidence.stage = {"local_batch": 7}
        evidence.settings = {"config": {"sequence_length": 8}}
        evidence.microbatches = []
        evidence.inputs(batch)
        first = evidence.microbatches[-1]
        self.assertEqual(len(first), 7)
        batch.labels[-1, -1] += 1
        evidence.inputs(batch)
        self.assertEqual(first[:6], evidence.microbatches[-1][:6])
        self.assertNotEqual(first[-1], evidence.microbatches[-1][-1])
        batch.labels = tokens[:1]
        with self.assertRaisesRegex(ValueError, "declared local batch"):
            evidence.inputs(batch)
