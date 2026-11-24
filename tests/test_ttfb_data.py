"""Input mistakes must be found before any startup timing is collected."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vldb_experiments import ROOT, cli, core, prepare, provenance, ttfb_data
from test_throughput_data import settings, text_fixture


NVME = dict(
    source="/dev/md0",
    fstype="ext4",
    target="/data",
    devices=["nvme1n1", "nvme2n1"],
    nvme=True,
)


class TTFBDataTests(unittest.TestCase):
    def test_checked_in_layouts_match_the_original_scale(self):
        spec = provenance.ttfb_data_settings()
        for key, samples, shards in [("text", 4860078, 622), ("vlm", 203077, 1593)]:
            pin = spec[key]
            counts = json.loads((ROOT / "configs" / pin["shard_samples"]).read_text())
            self.assertEqual(sum(counts), samples)
            self.assertEqual(len(counts), shards)
            self.assertGreater(min(counts), 0)
        self.assertEqual(spec["text"]["sequence_length"], 2048)
        self.assertEqual(
            8200 * spec["text"]["samples"] + 8 * spec["text"]["minimum_chunks"],
            spec["text"]["shard_file_bytes"],
        )
        self.assertEqual(
            spec["vlm"]["shard_order"],
            provenance.throughput_data_settings()["vlm"]["shard_order"],
        )

    def fixture(self, root):
        pins = settings()
        pins["validation"] = dict(
            maximum_size_difference=0.10, size_warning_difference=0.05
        )
        pins["text"].update(shard_samples="shape.json")
        (root / "configs").mkdir()
        (root / "configs/shape.json").write_text("[3]")
        path = root / "text"
        metadata = text_fixture(path, pins)
        metadata["shard_samples"] = [3]
        core.write_json(path / "manifest.json", metadata)
        pins["text"]["shard_file_bytes"] = (path / "chunk.bin").stat().st_size
        return pins, {"paths": {"text": str(path)}, "ttfb_data": {"source": "public"}}

    def test_public_ttfb_recipe_is_validated_instead_of_throughput_minimum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pins, profile = self.fixture(root)
            with (
                patch.object(provenance, "ttfb_data_settings", return_value=pins),
                patch.object(provenance, "ROOT", root),
                patch.object(ttfb_data, "ROOT", root),
                patch.object(ttfb_data, "local_storage", return_value=NVME),
            ):
                report = ttfb_data.check(profile, ["text"], hashes=True)
            self.assertTrue(report["valid"], report)
            self.assertEqual(report["datasets"]["text"]["records"], 3)
            self.assertTrue(report["datasets"]["text"]["content_hashed"])

    def test_wrong_layout_dtype_and_changed_payload_are_rejected(self):
        for problem in ["layout", "dtype", "content"]:
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                pins, profile = self.fixture(root)
                path = root / "text"
                metadata = core.read_json(path / "manifest.json")
                if problem == "layout":
                    (root / "configs/shape.json").write_text("[2, 1]")
                elif problem == "dtype":
                    index = core.read_json(path / "index.json")
                    index["config"]["data_format"] = ["no_header_numpy:5"]
                    core.write_json(path / "index.json", index)
                    metadata["sha256"]["index.json"] = provenance.file_sha256(
                        path / "index.json"
                    )
                    core.write_json(path / "manifest.json", metadata)
                else:
                    data = bytearray((path / "chunk.bin").read_bytes())
                    data[-1] ^= 1
                    (path / "chunk.bin").write_bytes(data)
                with (
                    patch.object(provenance, "ttfb_data_settings", return_value=pins),
                    patch.object(provenance, "ROOT", root),
                    patch.object(ttfb_data, "ROOT", root),
                    patch.object(ttfb_data, "local_storage", return_value=NVME),
                ):
                    report = ttfb_data.check(profile, ["text"], hashes=True)
                self.assertFalse(report["valid"], report)

    def test_remote_missing_and_non_nvme_paths_fail_before_reading_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            for path, storage in [
                ("s3://bucket/text", NVME),
                ("/does/not/exist", NVME),
                (temporary, NVME | dict(fstype="nfs4")),
                (temporary, NVME | dict(fstype="tmpfs")),
                (temporary, NVME | dict(ebs=True)),
                (temporary, NVME | dict(nvme=False, devices=["sda"])),
            ]:
                with (
                    self.subTest(path=path, storage=storage),
                    patch.object(ttfb_data, "local_storage", return_value=storage),
                    patch.object(provenance, "validate_datasets") as validate,
                ):
                    report = ttfb_data.check({"paths": dict(text=path)}, ["text"])
                self.assertFalse(report["valid"], report)
                validate.assert_not_called()

    def test_chunk_symlinks_cannot_bypass_the_storage_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pins, profile = self.fixture(root)
            chunk = root / "text/chunk.bin"
            chunk.rename(root / "outside.bin")
            chunk.symlink_to(root / "outside.bin")
            with (
                patch.object(provenance, "ttfb_data_settings", return_value=pins),
                patch.object(provenance, "ROOT", root),
                patch.object(ttfb_data, "ROOT", root),
                patch.object(ttfb_data, "local_storage", return_value=NVME),
            ):
                report = ttfb_data.check(profile, ["text"])
            self.assertFalse(report["valid"])
            self.assertIn("symlinks", report["errors"][0])

    def test_mount_probe_recognizes_nvme_raid(self):
        mount = dict(
            filesystems=[dict(source="/dev/md0", fstype="xfs", target="/mnt/local")]
        )
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps(mount))
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(ttfb_data.subprocess, "run", return_value=result),
            patch.object(
                ttfb_data, "block_devices", return_value=["nvme1n1", "nvme2n1"]
            ),
        ):
            self.assertTrue(ttfb_data.local_storage(Path(temporary))["nvme"])

    def test_vlm_size_drift_is_reported_and_large_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/shape.json").write_text("[3]")
            recipe = dict(
                vlm=dict(shard_samples="shape.json", shard_file_bytes=1000),
                validation=dict(
                    maximum_size_difference=0.1, size_warning_difference=0.05
                ),
            )
            for size, valid, warns in [
                (1000, True, False),
                (1060, True, True),
                (1200, False, False),
            ]:
                data = dict(
                    samples=3,
                    files=[dict(name="chunk.bin", samples=3)],
                    bytes=size,
                    index_sha256="index",
                )
                with (
                    self.subTest(size=size),
                    patch.object(ttfb_data, "ROOT", root),
                    patch.object(ttfb_data, "local_storage", return_value=NVME),
                    patch.object(provenance, "ttfb_data_settings", return_value=recipe),
                    patch.object(
                        provenance, "validate_datasets", return_value=dict(vlm=data)
                    ),
                ):
                    report = ttfb_data.check({"paths": dict(vlm=temporary)}, ["vlm"])
                self.assertEqual(report["valid"], valid, report)
                self.assertEqual(bool(report["warnings"]), warns, report)

    def test_preparation_uses_reference_boundaries_and_keeps_existing_data(self):
        spec = provenance.ttfb_data_settings()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = dict(
                paths=dict(
                    arena=str(root / "arena"),
                    environments=str(root / "envs"),
                    text=str(root / "text"),
                    vlm=str(root / "vlm"),
                ),
                ttfb_data=dict(source="public"),
            )
            generator = root / "arena/scripts/prepare_data.py"
            generator.parent.mkdir(parents=True)
            generator.write_text("generator")
            with (
                patch.object(provenance, "ttfb_data_settings", return_value=spec),
                patch.object(provenance, "validate_public_dataset") as validate,
                patch.object(prepare, "call") as call,
            ):
                prepare.ttfb_data(profile)
            self.assertEqual(call.call_count, 2)
            for command, key in zip(call.call_args_list, ["text", "vlm"]):
                args = list(map(str, command.args[0]))
                self.assertEqual(
                    args[args.index("--samples") + 1], str(spec[key]["samples"])
                )
                self.assertEqual(
                    args[args.index("--shard-samples") + 1],
                    str(ROOT / "configs" / spec[key]["shard_samples"]),
                )
            self.assertEqual(validate.call_count, 2)
            (root / "text").mkdir()
            with (
                patch.object(prepare, "call") as call,
                self.assertRaisesRegex(ValueError, "new output path"),
            ):
                prepare.ttfb_data(profile)
            call.assert_not_called()

    def test_local_profile_rejects_uris_instead_of_silently_resolving_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.toml"
            path.write_text(
                '[paths]\narena="/arena"\nplots="/plots"\ntext="s3://bucket/data"\nvlm="/vlm"\nenvironments="/envs"\n'
            )
            with self.assertRaisesRegex(ValueError, "local filesystem path"):
                core.load_profile(path)

    def test_check_obeys_the_shared_lock_without_running_a_benchmark(self):
        import fcntl

        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "host.lock"
            profile = dict(
                paths=dict(environments=temporary), measurement=dict(lock=str(lock))
            )
            with (
                lock.open("a") as held,
                patch.object(cli.profiles, "read", return_value={}),
                patch.object(cli.profiles, "load", return_value=profile),
                patch.object(ttfb_data, "check") as check,
            ):
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "measuring"):
                    cli.main(["check", "ttfb", "--profile", "unused.toml"])
                check.assert_not_called()
