"""Checkpoint-resize evidence must establish every update, not just curve closeness."""

import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from vldb_experiments.training import protocol, trace


def fixture(root, stage, identity="fixture"):
    directory = root / stage["name"]
    (directory / "diagnostics").mkdir(parents=True, exist_ok=True)
    for rank in range(stage["dp"]):
        rows = [
            {
                "event": "manifest",
                "schema": 1,
                "identity": identity,
                "rank": rank,
                "stage": stage,
            },
            {
                "event": "loaded",
                "loaded": bool(stage["start_step"]),
                "step": stage["start_step"],
                "metadata": "a" * 64 if stage["start_step"] else None,
                "parameters": f"{stage['start_step']:064x}",
                **(
                    {"loader_policy": protocol.loader_policy(stage)}
                    if "arm" in stage
                    else {}
                ),
            },
        ]
        control = protocol.rebuild_loader(stage)
        if control:
            rows.append(
                {
                    "event": "fast_forward",
                    "step": stage["start_step"],
                    "microbatches": stage["start_step"] * stage["accumulation_steps"],
                }
            )
        for step in range(stage["start_step"] + 1, stage["end_step"] + 1):
            groups = [
                [
                    f"{step * 10000 + (micro * stage['dp'] + rank) * stage.get('local_batch', 1) + sample + (1000000 if control else 0):064x}"
                    for sample in range(stage.get("local_batch", 1))
                ]
                for micro in range(stage["accumulation_steps"])
            ]
            row = {"event": "update", "step": step, "microbatches": groups}
            if rank == 0:
                loss = 1.0 / step + (0.01 if control else 0)
                row.update(
                    loss=loss,
                    loss_hex=loss.hex(),
                    parameters=f"{step + (1000000 if control else 0):064x}",
                )
            rows.append(row)
        rows.append({"event": "complete", "step": stage["end_step"]})
        (directory / "diagnostics" / f"rank-{rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    return directory


class TrainingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config = protocol.load(smoke=True)
        self.baseline, self.resume = [
            s for s in protocol.plan(self.config) if protocol.stage_arm(s) == "zephon"
        ]

    def test_campaign_has_both_loader_arms_with_their_own_checkpoint_continuations(
        self,
    ):
        config = protocol.load()
        stages = protocol.plan(config)
        self.assertEqual(
            [(s["dp"], s["start_step"], s["end_step"]) for s in stages],
            [(16, 0, 300), (8, 150, 300), (32, 150, 300), (64, 150, 300)] * 2,
        )
        self.assertEqual([s["accumulation_steps"] for s in stages], [4, 8, 2, 1] * 2)
        self.assertEqual([s["nodes"] for s in stages], [2, 1, 4, 8] * 2)
        self.assertEqual({s["arm"] for s in stages}, {"control", "zephon"})
        for stage in stages:
            if stage["start_step"]:
                self.assertEqual(stage["reference"], f"{stage['arm']}-baseline")

    def test_default_recipe_uses_1b_training_settings_and_eight_stages(self):
        config = protocol.load()
        self.assertEqual(len(protocol.plan(config)), 8)
        self.assertEqual(config["model"], "Llama3-1B-random-init")
        self.assertEqual(
            (
                config["local_batch"],
                config["sequences_per_step"],
                config["sequence_length"],
            ),
            (7, 448, 4096),
        )
        self.assertEqual((config["seed"], config["data_seed"]), (42, 1338))
        self.assertEqual(config["lr_scheduler"]["warmup_steps"], 400)
        self.assertEqual(config["lr_scheduler"]["total_steps"], 10899)
        self.assertTrue(protocol.recipe_status(config)["paper_training_settings_match"])
        self.assertEqual(
            protocol.recipe_status(config)["scope"], "public-data reproduction"
        )
        for key, value in (
            ("model", "different-model"),
            ("local_batch", 1),
            ("sequence_length", 1024),
            ("sequences_per_step", 64),
            ("data_seed", 42),
        ):
            changed = deepcopy(config)
            changed[key] = value
            self.assertFalse(
                protocol.recipe_status(changed)["paper_training_settings_match"]
            )

    def test_observed_changes_cannot_pass_as_the_declared_recipe(self):
        config = protocol.load()
        stage = protocol.plan(config)[0]
        original = {
            "config": observed_config_fixture(config, stage),
            "torch": config["sources"]["torch"],
            "zephon": config["sources"]["zephon"],
            "torchtitan_commit": config["sources"]["torchtitan"],
        }
        trace.validate_observed(original, config, stage)
        changes = [
            ("model", "dim", 1024),
            ("model", "enable_weight_tying", False),
            ("training", "num_tokens_per_microbatch_per_dp_rank", 4096),
            ("training", "num_tokens_per_train_step", 65536),
            ("dataloader", "seed", 42),
            ("debug", "seed", 43),
            ("checkpointer", "initial_load_model_only", True),
        ]
        for section, key, value in changes:
            observed = deepcopy(original)
            observed["config"][section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                trace.validate_observed(observed, config, stage)
        for section, key, value in (
            ("lr_scheduler", "warmup_steps", 10),
            ("lr_scheduler", "total_steps", 300),
        ):
            observed = deepcopy(original)
            observed["config"]["optim"][section][key] = value
            with self.assertRaisesRegex(ValueError, "LR schedule"):
                trace.validate_observed(observed, config, stage)

    def test_missing_seventh_sample_in_a_microbatch_is_rejected(self):
        config = protocol.load()
        config.update(steps=2, checkpoint_step=1, baseline_dp=1, resume_dps=[2])
        stage = protocol.plan(config)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = fixture(root, stage)
            path = directory / "diagnostics/rank-0.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[2]["microbatches"][0].pop()
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "microbatch geometry"):
                trace.read_trace(directory, stage, "fixture")

    def test_training_manifests_require_explicit_loader_arms(self):
        config = deepcopy(self.config)
        config.pop("arms")
        with self.assertRaisesRegex(ValueError, "explicit loader arms"):
            protocol.plan(config)

    def test_small_preset_keeps_training_recipe_and_uses_only_one_node(self):
        full, small = protocol.load(), protocol.load(small=True)
        self.assertEqual(
            {k: v for k, v in full.items() if k not in {"baseline_dp", "resume_dps"}},
            {k: v for k, v in small.items() if k not in {"baseline_dp", "resume_dps"}},
        )
        stages = protocol.plan(small)
        self.assertEqual([s["dp"] for s in stages], [2, 1, 4, 8] * 2)
        self.assertEqual([s["accumulation_steps"] for s in stages], [32, 64, 16, 8] * 2)
        self.assertTrue(all(s["nodes"] == 1 for s in stages))
        self.assertEqual(protocol.experiment_scale(small), "single-node")
        self.assertFalse(protocol.figure1_geometry(small))
        self.assertTrue(protocol.plot_geometry(small))

    def test_control_requires_exact_fast_forward_and_never_compares_across_arms(self):
        baseline, resume = [
            s for s in protocol.plan(self.config) if protocol.stage_arm(s) == "control"
        ]
        count = resume["start_step"] * resume["accumulation_steps"]
        self.assertEqual(
            list(protocol.loader_iterator(range(count + 3), resume)),
            list(range(count, count + 3)),
        )
        self.assertEqual(
            list(protocol.loader_iterator(range(3), self.resume)), list(range(3))
        )
        with self.assertRaisesRegex(ValueError, "exhausted"):
            protocol.loader_iterator(range(count - 1), resume)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = trace.read_trace(fixture(root, baseline), baseline, "fixture")
            b = trace.read_trace(fixture(root, resume), resume, "fixture")
            result = trace.compare(a, b)
            self.assertFalse(result["exact"])
            self.assertTrue(trace.outcome(result, "control")["expected_outcome_met"])
            self.assertFalse(trace.outcome(result, "zephon")["expected_outcome_met"])
            other = trace.read_trace(
                fixture(root, self.baseline), self.baseline, "fixture"
            )
            with self.assertRaisesRegex(ValueError, "own loader-arm"):
                trace.compare(other, b)
            path = root / resume["name"] / "diagnostics/rank-0.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[2]["microbatches"] -= 1
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "fast-forward count"):
                trace.read_trace(path.parent.parent, resume, "fixture")

    def test_exact_comparison_requires_all_rank_inputs_and_every_post_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = trace.read_trace(fixture(root, self.baseline), self.baseline, "fixture")
            b = trace.read_trace(fixture(root, self.resume), self.resume, "fixture")
            result = trace.compare(a, b)
            self.assertTrue(result["exact"])
            self.assertEqual(result["updates"], 2)
            b["losses"][3] = math.nextafter(b["losses"][3], math.inf)
            result = trace.compare(a, b)
            self.assertFalse(result["exact"])
            self.assertEqual(result["first_mismatch"]["step"], 3)
            self.assertTrue(result["first_mismatch"]["samples_match"])
            self.assertFalse(result["first_mismatch"]["loss_match"])

    def test_missing_rank_update_or_footer_cannot_be_treated_as_complete(self):
        for mode in ("rank", "update", "footer", "duplicate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                directory = fixture(Path(tmp), self.baseline)
                path = directory / "diagnostics/rank-1.jsonl"
                if mode == "rank":
                    path.unlink()
                else:
                    rows = path.read_text().splitlines()
                    if mode == "update":
                        rows.pop(2)
                    elif mode == "footer":
                        rows.pop()
                    else:
                        rows.insert(2, rows[2])
                    path.write_text("\n".join(rows) + "\n")
                with self.assertRaises(ValueError):
                    trace.read_trace(directory, self.baseline, "fixture")

    def test_source_probe_and_microbatch_groups_are_independent_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = trace.read_trace(fixture(root, self.baseline), self.baseline, "fixture")
            b = trace.read_trace(fixture(root, self.resume), self.resume, "fixture")
            changed = deepcopy(b)
            changed["loads"][0]["parameters"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "initial parameters"):
                trace.compare(a, changed)
            changed = deepcopy(b)
            changed["inputs"][3] = [
                [h for group in changed["inputs"][3] for h in group]
            ]
            result = trace.compare(a, changed)
            self.assertTrue(result["first_mismatch"]["samples_match"])
            self.assertFalse(result["first_mismatch"]["microbatch_groups_match"])

    def test_loss_json_must_retain_its_binary_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = fixture(Path(tmp), self.resume)
            path = directory / "diagnostics/rank-0.jsonl"
            rows = [json.loads(row) for row in path.read_text().splitlines()]
            rows[2]["loss"] = round(rows[2]["loss"], 4)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "FP64"):
                trace.read_trace(directory, self.resume, "fixture")


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional CPU torch checks")
class NativeCheckpointTests(unittest.TestCase):
    def test_native_torchtitan_flattened_model_state_is_recognized(self):
        from torch.distributed.checkpoint import save

        # CheckpointManager._flattened_model_states_sd merges the model's
        # state_dict at the root, keeping auxiliary Stateful objects nested.
        model = torch.nn.Linear(2, 2)
        auxiliary = {
            name: {"value": torch.ones(1)}
            for name in ("optimizer", "lr_scheduler", "dataloader", "train_state")
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(auxiliary | model.state_dict(), checkpoint_id=root / "native")
            self.assertTrue(trace.checkpoint_proof(root / "native")["shards"])
            save(auxiliary, checkpoint_id=root / "missing-model")
            with self.assertRaisesRegex(ValueError, "model"):
                trace.checkpoint_proof(root / "missing-model")

    def test_native_checkpoint_requires_full_training_state_and_all_shards(self):
        from torch.distributed.checkpoint import save

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint"
            state = {
                name: {"value": torch.ones(2)}
                for name in (
                    "model",
                    "optimizer",
                    "lr_scheduler",
                    "dataloader",
                    "train_state",
                )
            }
            save(state, checkpoint_id=checkpoint)
            proof = trace.checkpoint_proof(checkpoint)
            self.assertTrue(proof["shards"])
            shard = checkpoint / next(iter(proof["shards"]))
            shard.write_bytes(shard.read_bytes()[:10])
            with self.assertRaisesRegex(ValueError, "truncated"):
                trace.checkpoint_proof(checkpoint)
            metadata = checkpoint / ".metadata"
            metadata.write_bytes(metadata.read_bytes()[:32])
            with self.assertRaisesRegex(ValueError, "metadata"):
                trace.checkpoint_proof(checkpoint)
            partial = Path(tmp) / "model-only"
            save({"model": state["model"]}, checkpoint_id=partial)
            with self.assertRaisesRegex(ValueError, "optimizer"):
                trace.checkpoint_proof(partial)


class ReportTests(unittest.TestCase):
    def test_combining_complementary_campaigns_preserves_original_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [root / "zephon", root / "control"]
            for run, arm in zip(runs, ("zephon", "control"), strict=True):
                run.mkdir()
                config = protocol.load(smoke=True)
                config["arms"] = [arm]
                prepare_report_fixture(run, config, arm + "-identity")
            before = {
                p: p.read_bytes() for run in runs for p in run.rglob("*") if p.is_file()
            }
            output = root / "combined"
            summary = trace.combine(runs, output)
            self.assertTrue(summary["expected_outcomes_met"])
            self.assertFalse(summary["exact"])
            self.assertEqual(
                set(summary["comparisons"]), {"zephon-resume-dp1", "control-resume-dp1"}
            )
            self.assertFalse(summary["paper_protocol_match"])
            self.assertTrue(all(p.read_bytes() == raw for p, raw in before.items()))
            self.assertEqual(trace.validate_report(output), summary)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                trace.combine([runs[0], runs[0]], root / "invalid")
            manifest = json.loads((runs[1] / "campaign.json").read_text())
            manifest["config"]["seed"] += 1
            (runs[1] / "campaign.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "different data/model"):
                trace.combine(runs, root / "changed-recipe")

    def test_report_checks_provenance_and_computes_unrounded_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_report_fixture(root)
            summary = trace.report(root)
            self.assertFalse(summary["exact"])
            self.assertTrue(summary["expected_outcomes_met"])
            self.assertFalse(summary["paper_protocol_match"])  # reduced local fixture
            self.assertEqual(len(summary["comparisons"]), 2)
            self.assertEqual(
                summary["comparisons"]["zephon-resume-dp1"]["steps"][0][
                    "reference_loss"
                ].hex(),
                (1.0 / 3).hex(),
            )
            self.assertFalse((root / "report").exists())
            path = root / "control-baseline/source.metadata"
            path.write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "Source native"):
                trace.report(root)

    def test_loader_policy_normalization_cannot_hide_wrong_checkpoint_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_report_fixture(root)
            path = root / "control-resume-dp1/observed.json"
            observed = json.loads(path.read_text())
            observed["config"]["checkpointer"]["exclude_from_loading"] = [
                "optimizer",
                "dataloader",
            ]
            path.write_text(json.dumps(observed))
            with self.assertRaisesRegex(ValueError, "loader policy"):
                trace.report(root)

    def test_single_arm_report_cannot_claim_figure1_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare_report_fixture(root)
            path = root / "campaign.json"
            manifest = json.loads(path.read_text())
            manifest["config"]["arms"] = ["zephon"]
            path.write_text(json.dumps(manifest))
            summary = trace.report(root)
            self.assertTrue(summary["exact"])
            self.assertTrue(summary["expected_outcomes_met"])
            self.assertFalse(summary["paper_protocol_match"])


def observed_config_fixture(config, stage):
    return {
        "model": {
            "dim": 2048,
            "vocab_size": 128256,
            "enable_weight_tying": True,
            "max_context_length": config["sequence_length"],
            "layers": [
                {"attention": {"n_heads": 32, "n_kv_heads": 8}} for _ in range(16)
            ],
        },
        "precision": {
            "dtype": "float32",
            "param_dtype": "bfloat16",
            "reduce_dtype": "float64",
        },
        "loss_reduce_dtype": "float64",
        "dataloader": {
            **config["loader"],
            "seed": config["data_seed"],
            "canonical_replicas": config["canonical_replicas"]
            if protocol.stage_arm(stage) == "zephon"
            else None,
        },
        "checkpointer": {
            "initial_load_model_only": False,
            "exclude_from_loading": ["dataloader"]
            if protocol.rebuild_loader(stage)
            else [],
        },
        "optim": {
            "optimizer": {
                "optimizers": [
                    {
                        "fused": True,
                        **{
                            k: v
                            for k, v in config["optimizer"].items()
                            if k != "max_norm"
                        },
                    }
                ]
            },
            "max_norm": config["optimizer"]["max_norm"],
            "lr_scheduler": config["lr_scheduler"],
        },
        "training": {
            "num_tokens_per_microbatch_per_dp_rank": config["local_batch"]
            * config["sequence_length"],
            "num_tokens_per_train_step": config["sequences_per_step"]
            * config["sequence_length"],
            "steps": config["steps"],
            "disable_accelerator_graphs": True,
        },
        "debug": {"seed": config["seed"], "deterministic": True},
        "create_seed_checkpoint": False,
    }


def prepare_report_fixture(root, config=None, identity="fixture"):
    config = config or protocol.load(smoke=True)
    (root / "campaign.json").write_text(
        json.dumps({"config": config, "identity": identity})
    )
    receipt = {
        f"{folder}/{Path(name).name}": sha
        for key, folder in (("dataset", "data"), ("tokenizer", "tokenizer"))
        for name, sha in config[key]["files"].items()
    }
    (root / "assets.json").write_text(json.dumps(receipt))
    source = b"synthetic source metadata for reporting tests"
    source_sha = __import__("hashlib").sha256(source).hexdigest()
    for stage in protocol.plan(config):
        directory = fixture(root, stage, identity)
        final = b"synthetic final metadata for reporting tests"
        (directory / "final.metadata").write_bytes(final)
        proof = {
            "final": {
                "metadata_sha256": protocol.file_digest(directory / "final.metadata"),
                "shards": {"example.distcp": 1},
            }
        }
        if protocol.source_name(stage) is None:
            (directory / "source.metadata").write_bytes(source)
            proof["source"] = {
                "metadata_sha256": source_sha,
                "shards": {"example.distcp": 1},
            }
        else:
            for path in (directory / "diagnostics").glob("rank-*.jsonl"):
                path.write_text(
                    path.read_text().replace(
                        '"' + "a" * 64 + '"', '"' + source_sha + '"'
                    )
                )
        (directory / "checkpoint-proof.json").write_text(json.dumps(proof))
        (directory / "observed.json").write_text(
            json.dumps(
                {
                    "config": observed_config_fixture(config, stage),
                    "torchtitan_commit": config["sources"]["torchtitan"],
                    "torch": config["sources"]["torch"],
                    "zephon": config["sources"]["zephon"],
                }
            )
        )


try:
    import zephon_modal
except ImportError:
    zephon_modal = None


@unittest.skipIf(
    zephon_modal is None, "Install the training extra for Modal adapter checks"
)
class ModalAdapterTests(unittest.TestCase):
    def test_each_resume_uses_only_its_matching_arm_checkpoint(self):
        from vldb_experiments.training.modal_app import stage_source

        config = protocol.load()
        completed = {
            f"{arm}-baseline": {
                "directory": f"/runs/{arm}-baseline/attempts/original",
                "proof": {"source": {"metadata_sha256": sha * 64}},
            }
            for arm, sha in (("control", "a"), ("zephon", "b"))
        }
        for stage in protocol.plan(config):
            source, sha = stage_source(stage, completed, config)
            if not stage["start_step"]:
                self.assertIsNone(source)
                self.assertIsNone(sha)
            else:
                self.assertEqual(
                    source,
                    f"/runs/{stage['arm']}-baseline/attempts/original/checkpoint/step-150",
                )
                self.assertEqual(sha, ("a" if stage["arm"] == "control" else "b") * 64)
        del completed["control-baseline"]
        control_resume = protocol.plan(config)[1]
        with self.assertRaises(KeyError):
            stage_source(control_resume, completed, config)

    def test_platform_retries_cannot_resume_a_reference_from_its_halfway_checkpoint(
        self,
    ):
        from dataclasses import replace

        from zephon_modal.context import LaunchContext

        from vldb_experiments.training.adapter import ExperimentAdapter

        config = protocol.load(smoke=True)
        adapter = ExperimentAdapter(
            module="vldb_experiments.training.runtime",
            config="get_config",
            checkpoints=None,
            manifest={"config": config, "identity": "fixture"},
            stage=protocol.plan(config)[0],
        )
        ctx = LaunchContext(
            run_id="baseline",
            run_dir="/runs/baseline",
            steps=4,
            seed=42,
            launch_id="first",
            placement="h100x2",
            nnodes=1,
            node_rank=0,
            nproc_per_node=2,
            sample_hash_dir="/tmp/first",
        )
        first = json.loads(adapter.env(ctx)["ZEPHON_REPRO_STAGE"])
        second = json.loads(
            adapter.env(replace(ctx, launch_id="retry"))["ZEPHON_REPRO_STAGE"]
        )
        self.assertIsNone(adapter.checkpoints)
        self.assertIsNone(second["source"])
        self.assertNotEqual(first["directory"], second["directory"])
        self.assertIn(
            second["directory"], adapter.command(replace(ctx, launch_id="retry"))[-1]
        )

    def test_interrupted_attempts_are_not_spliced_into_completed_evidence(self):
        from vldb_experiments.training.modal_app import completed_attempt

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = protocol.plan(protocol.load(smoke=True))[0]
            original = fixture(root, stage)
            run = root / "remote"
            raw = run / "sample-hashes"
            raw.mkdir(parents=True)
            data = [
                (original / f"diagnostics/rank-{rank}.jsonl").read_bytes()
                for rank in range(2)
            ]
            (raw / "attempt-a.rank-0.jsonl").write_bytes(data[0])
            (raw / "attempt-b.rank-1.jsonl").write_bytes(data[1])
            with self.assertRaisesRegex(ValueError, "found 0"):
                completed_attempt(run, stage, "fixture")
            for rank in range(2):
                (raw / f"attempt-c.rank-{rank}.jsonl").write_bytes(data[rank])
            self.assertEqual(
                completed_attempt(run, stage, "fixture"), run / "attempts/attempt-c"
            )
            self.assertEqual((raw / "attempt-a.rank-0.jsonl").read_bytes(), data[0])


class ObservedNumericsTests(unittest.TestCase):
    def test_callable_addresses_are_ignored_but_callable_names_and_learning_rates_are_not(
        self,
    ):
        before = {
            "config": {
                "init": "<function factory.<locals>._init at 0x1234>",
                "optim": {"lr": 0.0003},
                "parallelism": {"data_parallel_replicate_degree": 16},
                "checkpointer": {"folder": "/reference", "initial_load_path": None},
            }
        }
        after = deepcopy(before)
        after["config"]["init"] = "<function factory.<locals>._init at 0x5678>"
        after["config"]["parallelism"]["data_parallel_replicate_degree"] = 8
        after["config"]["checkpointer"] = {
            "folder": "/resume",
            "initial_load_path": "/reference/step-150",
        }
        self.assertEqual(
            trace.observed_contract(before), trace.observed_contract(after)
        )
        after["config"]["optim"]["lr"] *= 2
        self.assertNotEqual(
            trace.observed_contract(before), trace.observed_contract(after)
        )
        after = deepcopy(before)
        after["config"]["init"] = "<function another_factory.<locals>._init at 0x1234>"
        self.assertNotEqual(
            trace.observed_contract(before), trace.observed_contract(after)
        )


@unittest.skipIf(zephon_modal is None or torch is None, "Training dependency checks")
class PublicAssetTests(unittest.TestCase):
    def test_public_staging_verifies_both_data_and_tokenizer_and_uses_no_hf_token(self):
        import hashlib
        from unittest.mock import patch

        import pyarrow as pa
        import pyarrow.parquet as pq

        from vldb_experiments.training.assets import stage_assets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "public.parquet"
            pq.write_table(
                pa.table({"text": ["first public example", "second public example"]}),
                parquet,
            )
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text('{"example":"public tokenizer test"}')
            config = {
                "dataset": {
                    "repository": "example/public",
                    "revision": "a" * 40,
                    "files": {
                        "public.parquet": hashlib.sha256(
                            parquet.read_bytes()
                        ).hexdigest()
                    },
                },
                "tokenizer": {
                    "repository": "example/tokenizer",
                    "revision": "b" * 40,
                    "files": {
                        "tokenizer.json": hashlib.sha256(
                            tokenizer.read_bytes()
                        ).hexdigest()
                    },
                },
            }
            with patch(
                "huggingface_hub.hf_hub_download",
                side_effect=lambda repo, name, **kwargs: root / name,
            ) as download:
                receipt = stage_assets(config, root / "assets")
                self.assertEqual(len(receipt), 2)
                self.assertTrue((root / "assets/data/index.json").is_file())
                self.assertTrue(
                    all(
                        call.kwargs["token"] is False
                        for call in download.call_args_list
                    )
                )
                (root / "assets/tokenizer/tokenizer.json").write_text("changed")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    stage_assets(config, root / "assets")


@unittest.skipIf(zephon_modal is None, "Install the training extra for launch checks")
class SubmissionTests(unittest.TestCase):
    def test_coordinator_continuation_preserves_training_and_refuses_overlap(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from modal.types import InputStatus

        from vldb_experiments.training.cli import continue_campaign, write_json

        class Registry(dict):
            def put(self, key, value, skip_if_exists=False):
                if skip_if_exists and key in self:
                    return False
                self[key] = value
                return True

        registry = Registry()
        coordinator = SimpleNamespace(
            spawn=Mock(return_value=SimpleNamespace(object_id="fc-continued"))
        )
        node = SimpleNamespace(
            function_call_id="fc-previous", status=InputStatus.PENDING
        )
        manifest = {
            "config": protocol.load(smoke=True),
            "code": protocol.code_identity(),
            "identity": "fixture",
            "zephon_modal": {"commit": "fixture"},
        }
        # A verifier-only fix may continue the same actual training recipe.
        manifest["code"]["files"]["src/vldb_experiments/training/trace.py"] = (
            "old-verifier"
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "vldb_experiments.training.cli.status", return_value={"state": "failed"}
            ),
            patch(
                "modal.FunctionCall.from_id",
                return_value=SimpleNamespace(get_call_graph=lambda: [node]),
            ),
            patch(
                "vldb_experiments.training.cli.dependency_identity",
                return_value={"commit": "fixture"},
            ),
            patch(
                "vldb_experiments.training.modal_app.build_app",
                return_value=(
                    SimpleNamespace(run=lambda **kwargs: nullcontext()),
                    coordinator,
                    None,
                    registry,
                    {},
                ),
            ),
        ):
            root = Path(tmp)
            write_json(root / "campaign.json", manifest)
            previous = {
                "identity": "fixture",
                "state": "submitted",
                "call_id": "fc-previous",
            }
            write_json(root / "launch.json", previous)
            with self.assertRaisesRegex(RuntimeError, "active"):
                continue_campaign(root)
            coordinator.spawn.assert_not_called()
            node.status = InputStatus.FAILURE
            changed = deepcopy(manifest)
            changed["code"]["files"]["src/vldb_experiments/training/runtime.py"] = (
                "different-training"
            )
            write_json(root / "campaign.json", changed)
            with self.assertRaisesRegex(ValueError, "Training source changed"):
                continue_campaign(root)
            coordinator.spawn.assert_not_called()
            write_json(root / "campaign.json", manifest)
            continue_campaign(root)
            self.assertEqual(coordinator.spawn.call_count, 1)
            self.assertEqual(coordinator.spawn.call_args.args[0], manifest)
            write_json(root / "launch.json", previous)
            continue_campaign(root)
            self.assertEqual(coordinator.spawn.call_count, 1)
            self.assertEqual(
                json.loads((root / "launch.json").read_text())["call_id"],
                "fc-continued",
            )

    def test_collection_without_a_replacement_coordinator(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        import modal

        from vldb_experiments.training.cli import collect, write_json

        manifest = {
            "identity": "fixture",
            "prefix": "unit-test",
            "config": protocol.load(smoke=True),
        }
        stages = protocol.plan(manifest["config"])
        records = {
            s["name"]: {"directory": f"/runs/{s['name']}/attempt"} for s in stages
        }
        files = {
            "campaign.json": json.dumps(manifest).encode(),
            "result.json": json.dumps(records).encode(),
            "assets.json": b"{}",
        }
        volume = SimpleNamespace(
            read_file=lambda path: [files.get(path, b"preserved evidence")],
            iterdir=Mock(
                side_effect=modal.exception.NotFoundError("No controllers directory")
            ),
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("modal.Volume.from_name", return_value=volume),
            patch(
                "vldb_experiments.training.cli.trace.report",
                return_value={"exact": True},
            ) as report,
        ):
            root = Path(tmp)
            write_json(root / "campaign.json", manifest)
            self.assertEqual(collect(root), {"exact": True})
            report.assert_called_once_with(root)
            for stage in stages:
                self.assertTrue((root / stage["name"] / "final.metadata").is_file())

    def test_repeat_and_uncertain_submissions_never_spawn_another_campaign(self):
        from argparse import Namespace
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from vldb_experiments.training.cli import run_campaign

        class Registry(dict):
            def put(self, key, value, skip_if_exists=False):
                if skip_if_exists and key in self:
                    return False
                self[key] = value
                return True

        registry = Registry()
        coordinator = SimpleNamespace(
            spawn=Mock(return_value=SimpleNamespace(object_id="fc-test"))
        )
        app = SimpleNamespace(run=lambda **kwargs: nullcontext())
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "vldb_experiments.training.modal_app.build_app",
                return_value=(app, coordinator, None, registry, {}),
            ),
            patch(
                "vldb_experiments.training.preparation.require_inputs", return_value={}
            ),
            patch(
                "vldb_experiments.training.cli.dependency_identity",
                return_value={"commit": "fixture"},
            ),
        ):
            args = Namespace(
                output=Path(tmp), config=None, smoke=True, prefix="unit-test"
            )
            run_campaign(args)
            run_campaign(args)
            self.assertEqual(coordinator.spawn.call_count, 1)
            del registry["submission"]["call_id"]
            with self.assertRaisesRegex(RuntimeError, "uncertain"):
                run_campaign(args)
            self.assertEqual(coordinator.spawn.call_count, 1)
