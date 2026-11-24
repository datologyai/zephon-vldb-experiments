from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_sweep_protocol import write_result

from vldb_experiments import core, matrix, provenance
from vldb_experiments.report import export_results


class CampaignTests(unittest.TestCase):
    def test_environment_discovery_ignores_ambient_pythonpath_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "ambient_fake_dependency-999.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: ambient-fake-dependency\nVersion: 999\n"
            )
            with patch.dict(os.environ, {"PYTHONPATH": tmp}):
                result = provenance.environment_states([Path(sys.executable)])
            state = result[sys.executable]
            self.assertNotIn("ambient-fake-dependency", state["packages"])
            self.assertTrue(state["python"])

    def test_report_rejects_pooling_different_vlm_payload_layouts(self):
        selected = [
            c
            for c in matrix.cases("vlm-pipeline")
            if c["loader_name"] == "zephon-fused-dop8-f2"
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, case in enumerate(selected):
                case["config"]["vlm_zephon"]["flatten_payload"] = index == 0
                folder = root / "cases" / case["id"]
                core.write_json(folder / "status.json", {"state": "complete"})
                core.write_json(
                    folder / "result.json",
                    {
                        "csv_row": {
                            "result.samples_per_second": 1,
                            "result.usable_tokens_per_second": 100,
                        }
                    },
                )
            core.write_json(root / "campaign.json", {"cases": selected})
            with self.assertRaisesRegex(ValueError, "payload layouts"):
                export_results(root)

    def test_matrix_is_exact_and_deduplicated(self):
        fnb = matrix.cases("fetch-and-batch")
        vlm = matrix.cases("vlm-pipeline")
        self.assertEqual(len(fnb), 294)
        self.assertEqual(len(vlm), 93)
        self.assertEqual(len({c["id"] for c in fnb + vlm}), 387)
        self.assertEqual({c["batch_size"] for c in fnb}, {32})
        self.assertEqual({c["batch_size"] for c in vlm}, {4})
        self.assertEqual({c["seed"] for c in fnb + vlm}, {42, 67, 1337})
        self.assertEqual(
            {c["system"] for c in fnb if c["python"] == "3.14t"}, {"zephon"}
        )
        self.assertEqual(sum(c["system"] == "zephon" for c in vlm), 81)

    def test_paper_measurement_windows_and_vlm_stage_allocations(self):
        fetch = matrix.plan("fetch-and-batch", "v0.1.0")
        self.assertEqual({c["warmup_batches"] for c in fetch}, {16})
        for dataset, batches in (("text", 7813), ("vlm", 2344)):
            self.assertEqual(
                {c["measure_batches"] for c in fetch if c["dataset"] == dataset},
                {batches},
            )
        vlm = matrix.plan("vlm-pipeline", "v0.1.0")
        self.assertEqual({c["warmup_batches"] for c in vlm}, {50})
        for system, batches in (("grain", 2500), ("zephon", 1500)):
            self.assertEqual(
                {c["measure_batches"] for c in vlm if c["system"] == system}, {batches}
            )
        selected = next(
            c for c in vlm if c["loader_name"] == "zephon-fused-dop64-f8-p2c2"
        )
        self.assertEqual(selected["config"]["fetch_parallelism"], 8)
        self.assertEqual(selected["config"]["vlm_zephon"]["fused_parallelism"], 54)
        self.assertEqual(
            selected["config"]["vlm_zephon"]["shuffle_pack_parallelism"], 2
        )
        self.assertEqual(
            selected["config"]["vlm_zephon"]["batch_construct_parallelism"], 2
        )

    def test_vlm_launches_pin_the_recorded_tokenizer(self):
        selected = matrix.plan("vlm-pipeline", "v0.1.0")
        self.assertEqual(len(selected), 93)
        self.assertEqual(
            {c["config"]["vlm"]["tokenizer_revision"] for c in selected},
            {"c1899de289a04d12100db370d81485cdf75e47ca"},
        )
        self.assertTrue(
            all(
                "tokenizer_revision" in c["config"]["vlm"]
                for c in matrix.cases("vlm-pipeline")
            )
        )

    def test_paper_vlm_uses_flat_tensor_payloads(self):
        selected = matrix.plan("vlm-pipeline", "v0.1.0")
        zephon = [c for c in selected if c["system"] == "zephon"]
        self.assertEqual(len(zephon), 81)
        self.assertTrue(
            all(c["config"]["vlm_zephon"]["flatten_payload"] for c in zephon)
        )
        self.assertTrue(
            all(
                c["config"]["vlm_zephon"]["flatten_payload"]
                for c in matrix.cases("vlm-pipeline")
                if c["system"] == "zephon"
            )
        )

    def test_baseline_keeps_the_full_paper_and_gil_matrix(self):
        selected = matrix.plan("fetch-and-batch", "v0.1.0")
        self.assertEqual(len(selected), 294)
        self.assertEqual(
            {c["python"] for c in selected if c["system"] == "zephon"},
            {"3.13", "3.14t"},
        )
        noshm = [c for c in selected if c["variant"] == "noshm"]
        self.assertEqual(len(noshm), 42)
        self.assertTrue(
            all(c["config"]["pipeline"]["coalesce_tensors"] is False for c in noshm)
        )
        self.assertTrue(
            all(
                "shm_enabled" not in c["config"]["pipeline"]
                for c in selected
                if c["system"] == "zephon"
            )
        )

    def test_unknown_revision_requires_an_explicit_pin(self):
        with self.assertRaisesRegex(ValueError, "Unknown revision"):
            matrix.plan("fetch-and-batch", "unconfigured", systems=["zephon"])

    def test_smoke_does_not_modify_paper_definitions(self):
        selected = matrix.plan(
            "fetch-and-batch",
            "v0.1.0",
            panel="overview",
            systems=["zephon"],
            seeds=[42],
            dops=[8],
            smoke=True,
        )
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(c["measure_batches"] == 8 for c in selected))
        self.assertTrue(
            all(c["measure_batches"] > 8 for c in matrix.cases("fetch-and-batch"))
        )

    def test_dataset_validation_requires_public_source(self):
        for purpose in ("throughput", "ttfb"):
            with (
                self.subTest(purpose=purpose),
                self.assertRaisesRegex(ValueError, "must be 'public'"),
            ):
                provenance.validate_datasets(
                    {
                        purpose + "_data": {"source": "archived"},
                        "paths": {"text": "/unused"},
                    },
                    ["text"],
                    purpose=purpose,
                )

    def test_dataset_validation_rejects_truncated_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core.write_json(
                root / "index.json",
                {"chunks": [{"filename": "a.bin", "chunk_bytes": 16, "chunk_size": 1}]},
            )
            (root / "a.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "Wrong byte count"):
                provenance.dataset_manifest(root)

    def test_dataset_validation_includes_offset_table_and_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core.write_json(
                root / "index.json",
                {"chunks": [{"filename": "a.bin", "chunk_bytes": 16, "chunk_size": 1}]},
            )
            (root / "a.bin").write_bytes(struct.pack("<III", 1, 12, 16) + b"data")
            manifest = provenance.dataset_manifest(root, content_hashes=True)
            self.assertEqual(manifest["bytes"], 16)
            self.assertEqual(manifest["samples"], 1)
            self.assertIn("sha256", manifest["files"][0])

    def test_resume_skips_completed_cases_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "paths": {
                    "environments": str(root / "envs"),
                    "arena": str(root),
                    "plots": str(root),
                }
            }
            counter = root / "executions.txt"
            selected = [matrix.cases("fetch-and-batch")[0]]

            def command(profile, case, revision, attempt):
                write_result(attempt, case)
                script = "from pathlib import Path; import sys; p=Path(sys.argv[2]); p.write_text(p.read_text()+'run\\n' if p.exists() else 'run\\n')"
                return [sys.executable, "-c", script, str(attempt), str(counter)]

            with (
                patch.object(
                    provenance, "source_state", return_value={"revision": "fixture"}
                ),
                patch.object(provenance, "validate_datasets", return_value={}),
                patch.object(provenance, "host_snapshot", return_value={}),
                patch.object(provenance, "environment_states", return_value={}),
                patch.object(core, "sweep_command", side_effect=command),
            ):
                self.assertEqual(
                    core.execute(selected, profile, "test", root / "run", 10, False, 0),
                    0,
                )
                self.assertEqual(
                    core.execute(selected, profile, "test", root / "run", 10, False, 0),
                    0,
                )
                self.assertEqual(counter.read_text().splitlines(), ["run"])
                changed = [selected[0] | {"seed": 999}]
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    core.execute(changed, profile, "test", root / "run", 10, False, 0)


if __name__ == "__main__":
    unittest.main()
