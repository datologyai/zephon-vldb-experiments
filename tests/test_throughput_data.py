"""Throughput inputs need verified provenance and sufficient, correctly shaped records."""

import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vldb_experiments import ROOT, cli, core, prepare, provenance


def settings():
    pins = provenance.throughput_data_settings()
    pins["text"].update(samples=3, minimum_chunks=1, sequence_length=2)
    return pins


def text_fixture(root, pins, *, samples=3, record_length=2):
    root.mkdir()
    pin = pins["text"]
    record = struct.pack("<I", record_length * 4) + struct.pack(
        f"<{record_length}I", *range(record_length)
    )
    first = 4 * (samples + 2)
    offsets = [first + i * len(record) for i in range(samples + 1)]
    (root / "chunk.bin").write_bytes(
        struct.pack(f"<{samples + 2}I", samples, *offsets) + record * samples
    )
    core.write_json(
        root / "index.json",
        {
            "config": {"compression": None, "data_format": ["no_header_numpy:6"]},
            "chunks": [
                {
                    "filename": "chunk.bin",
                    "chunk_size": samples,
                    "chunk_bytes": offsets[-1],
                }
            ],
        },
    )
    metadata = {
        "format_version": 1,
        "mode": "text",
        "schema": "uint32[sequence_length]",
        "source": {
            "repo": pin["repository"],
            "revision": pin["revision"],
            "subset": pin["subset"],
        },
        "tokenizer": {
            "repo": pin["tokenizer"],
            "revision": pin["tokenizer_revision"],
            "add_special_tokens": False,
        },
        "seed": pin["seed"],
        "sequence_length": pin["sequence_length"],
        "chunk_bytes": pin["chunk_mib"] * 1024**2,
        "samples": samples,
        "chunks": 1,
        "sha256": {
            name: provenance.file_sha256(root / name)
            for name in ["chunk.bin", "index.json"]
        },
    }
    core.write_json(root / "manifest.json", metadata)
    return metadata


class ThroughputDataTests(unittest.TestCase):
    def test_recipe_covers_the_configured_windows_and_worker_read_ahead(self):
        spec = json.loads(
            (ROOT / "configs/throughput/fetch-and-batch.json").read_text()
        )
        pins = provenance.throughput_data_settings()
        prefetch = max(
            spec["configs"]["litdata"]["prefetch_factor"],
            spec["configs"]["grain"]["pipeline"]["prefetch"]["buffer_size"],
        )
        queued_and_active = max(spec["dops"]) * (prefetch + 1) * spec["batch_size"]
        for key, batches in spec["measure_batches"].items():
            consumer = (spec["warmup_batches"] + batches) * spec["batch_size"]
            self.assertGreaterEqual(
                pins[key]["samples"], math.ceil(1.2 * (consumer + queued_and_active))
            )

    def test_public_file_byte_count_includes_the_offset_table(self):
        pins = settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "text"
            text_fixture(path, pins)
            with patch.object(
                provenance, "throughput_data_settings", return_value=pins
            ):
                result = provenance.validate_datasets(
                    {
                        "throughput_data": {"source": "public"},
                        "paths": {"text": str(path)},
                    },
                    ["text"],
                    hashes=True,
                )
            self.assertEqual(result["text"]["samples"], 3)
            self.assertTrue(result["text"]["generation_manifest_sha256"])
            self.assertEqual(
                provenance.dataset_manifest(path)["bytes"], result["text"]["bytes"]
            )

    def test_changed_source_revision_is_rejected_even_when_files_match_manifest(self):
        pins = settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "text"
            metadata = text_fixture(path, pins)
            metadata["source"]["revision"] = "different-corpus-revision"
            core.write_json(path / "manifest.json", metadata)
            with (
                patch.object(provenance, "throughput_data_settings", return_value=pins),
                self.assertRaisesRegex(ValueError, "source provenance"),
            ):
                provenance.validate_public_dataset(path, "text", hashes=True)

    def test_vlm_recipe_is_required_even_when_source_and_file_hashes_match(self):
        pins = settings()
        pins["vlm"].update(samples=1, minimum_chunks=1)
        pin = pins["vlm"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vlm"
            path.mkdir()
            leaves = [
                b"encoded image",
                b"conversation",
                b"id",
                b"url",
                struct.pack("<q", 1),
            ]
            record = struct.pack("<5I", *(len(leaf) for leaf in leaves)) + b"".join(
                leaves
            )
            (path / "chunk.bin").write_bytes(
                struct.pack("<3I", 1, 12, 12 + len(record)) + record
            )
            core.write_json(
                path / "index.json",
                {
                    "config": {
                        "compression": None,
                        "data_format": ["bytes", "str", "str", "str", "int"],
                    },
                    "chunks": [
                        {
                            "filename": "chunk.bin",
                            "chunk_size": 1,
                            "chunk_bytes": 12 + len(record),
                        }
                    ],
                },
            )
            metadata = {
                "format_version": 1,
                "mode": "vlm",
                "source": {
                    "repo": pin["repository"],
                    "revision": pin["revision"],
                    "subset": pin["subset"],
                },
                "seed": pin["seed"],
                "chunk_bytes": pin["chunk_mib"] * 1024**2,
                "schema": "encoded_images_conversation_metadata",
                "vlm_loader_settings": {"decoded_image_input": False},
                "shard_order": pin["shard_order"],
                "samples": 1,
                "chunks": 1,
                "sha256": {
                    name: provenance.file_sha256(path / name)
                    for name in ["index.json", "chunk.bin"]
                },
            }
            core.write_json(path / "manifest.json", metadata)
            with patch.object(
                provenance, "throughput_data_settings", return_value=pins
            ):
                self.assertEqual(
                    provenance.validate_public_dataset(path, "vlm", hashes=True)[
                        "samples"
                    ],
                    1,
                )
                metadata["shard_order"] = (
                    "archive prefixes without global record mixing"
                )
                core.write_json(path / "manifest.json", metadata)
                with self.assertRaisesRegex(
                    ValueError, "different sampling or record-order recipe"
                ):
                    provenance.validate_public_dataset(path, "vlm", hashes=True)

    def test_setup_sample_is_rejected_for_the_full_paper_matrix(self):
        pins = settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "text"
            text_fixture(path, pins, samples=2)
            with (
                patch.object(provenance, "throughput_data_settings", return_value=pins),
                self.assertRaisesRegex(ValueError, "too small"),
            ):
                provenance.validate_public_dataset(path, "text")

    def test_actual_record_width_must_match_the_declared_sequence_length(self):
        pins = settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "text"
            text_fixture(path, pins, record_length=3)
            with (
                patch.object(provenance, "throughput_data_settings", return_value=pins),
                self.assertRaisesRegex(ValueError, "sequence length"),
            ):
                provenance.validate_public_dataset(path, "text")

    def test_index_dtype_must_be_uint32_even_when_record_width_matches(self):
        pins = settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "text"
            metadata = text_fixture(path, pins)
            index = core.read_json(path / "index.json")
            index["config"]["data_format"] = ["no_header_numpy:5"]
            core.write_json(path / "index.json", index)
            metadata["sha256"]["index.json"] = provenance.file_sha256(
                path / "index.json"
            )
            core.write_json(path / "manifest.json", metadata)
            with (
                patch.object(provenance, "throughput_data_settings", return_value=pins),
                self.assertRaisesRegex(ValueError, "serialization schema"),
            ):
                provenance.validate_public_dataset(path, "text", hashes=True)

    def test_changed_payload_is_rejected_with_content_hashes(self):
        pins = settings()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "text"
            text_fixture(path, pins)
            payload = bytearray((path / "chunk.bin").read_bytes())
            payload[-1] ^= 1
            (path / "chunk.bin").write_bytes(payload)
            with (
                patch.object(provenance, "throughput_data_settings", return_value=pins),
                self.assertRaisesRegex(ValueError, "chunk differs"),
            ):
                provenance.validate_public_dataset(path, "text", hashes=True)

    def test_public_preparation_calls_the_arena_generator_without_aws(self):
        pins = settings()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = root / "arena/scripts/prepare_data.py"
            generator.parent.mkdir(parents=True)
            generator.write_text("# generator fixture\n")
            profile = {
                "throughput_data": {"source": "public"},
                "paths": {
                    "arena": str(root / "arena"),
                    "environments": str(root / "envs"),
                    "text": str(root / "text"),
                    "vlm": str(root / "vlm"),
                },
            }
            with (
                patch.object(provenance, "throughput_data_settings", return_value=pins),
                patch.object(provenance, "validate_public_dataset") as validate,
                patch.object(prepare, "call") as call,
                patch.object(cli.profiles, "read", return_value={}),
                patch.object(
                    cli.profiles,
                    "load",
                    return_value=profile
                    | {"measurement": {"lock": str(root / "lock")}},
                ),
            ):
                cli.main(["prepare", "throughput-data", "--profile", "unused.toml"])
            self.assertEqual(call.call_count, 2)
            for key, invocation in zip(["text", "vlm"], call.call_args_list):
                args = invocation.args[0]
                self.assertEqual(args[0], root / "envs/cpu-3.13/bin/python")
                self.assertEqual(
                    args[args.index("--samples") + 1], str(pins[key]["samples"])
                )
                self.assertEqual(
                    args[args.index("--revision") + 1], pins[key]["revision"]
                )
                self.assertIn(generator, args)
            self.assertEqual(validate.call_count, 2)

    def test_unknown_throughput_data_source_cannot_silently_fall_back_to_private_download(
        self,
    ):
        with (
            patch.object(prepare, "call") as call,
            self.assertRaisesRegex(ValueError, "throughput_data.source"),
        ):
            prepare.throughput_data({"throughput_data": {"source": "typo"}})
        call.assert_not_called()


def test_generator_fingerprints_are_informational_on_reuse(tmp_path):
    pins = settings()
    path = tmp_path / "text"
    text_fixture(path, pins)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["generator_files"] = {"scripts/data_preparation/mammoth.py": "changed"}
    (path / "manifest.json").write_text(json.dumps(manifest))
    provenance.validate_public_dataset(path, "text", recipe=pins, hashes=True)
