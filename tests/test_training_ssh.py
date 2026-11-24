"""SSH execution must preserve the protocol, provenance and duplicate guards."""

import io
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

from test_training import prepare_report_fixture

from vldb_experiments.training import protocol, ssh, ssh_runner


def inventory(count=8, capability="9.0", driver="580.65.06"):
    return "".join(
        f"GPU-{rank}, NVIDIA H100, {capability}, {driver}\n" for rank in range(count)
    ).encode()


def fake_host(root, *, failure=None, count=8, capability="9.0", driver="580.65.06"):
    """Execute the real POSIX probes with isolated host-tool stand-ins, no SSH/GPU."""
    tools = root / "bin"
    tools.mkdir()
    for name in (
        "tar",
        "gzip",
        "mktemp",
        "mkdir",
        "rmdir",
        "rm",
        "cat",
        "id",
        "dirname",
    ):
        (tools / name).symlink_to(shutil.which(name))

    def script(name, content):
        path = tools / name
        path.write_text(f"#!{sys.executable}\n{content}")
        path.chmod(0o755)

    script("uname", f"print({'Darwin' if failure == 'os' else 'Linux'!r})")
    script(
        "nvidia-smi",
        f"import sys\nsys.stdout.buffer.write({inventory(count, capability, driver)!r})",
    )
    if failure != "docker-missing":
        script(
            "docker",
            f"""import json, sys
from pathlib import Path
args = sys.argv[1:]
with Path({str(root / "docker-calls.jsonl")!r}).open('a') as stream:
    stream.write(json.dumps(args) + '\\n')
failure = {failure!r}
if args[0] == 'info':
    if failure == 'daemon':
        sys.stderr.write('permission denied accessing Docker daemon')
        sys.exit(1)
    print('linux')
elif args[:2] == ['build', '--help']:
    sys.exit(1 if failure == 'build' else 0)
elif args[0] == 'run':
    if failure == 'runtime':
        sys.stderr.write('could not select device driver with capabilities: [[gpu]]')
        sys.exit(1)
    mount = args[args.index('--mount') + 1]
    probe = Path(mount.split('src=', 1)[1].split(',dst=', 1)[0])
    assert (probe / 'host-check').read_text() == 'ok\\n'
    if failure != 'bind':
        (probe / 'container-check').write_text('ok\\n')
    sys.stdout.buffer.write({inventory(4 if failure == "container-count" else count, capability, driver)!r})
else:
    raise AssertionError(args)
""",
        )
    if failure == "output":
        (tools / "mktemp").unlink()
        script("mktemp", "import sys\nsys.exit(1)")
    calls = []

    def remote(host, args, **kwargs):
        calls.append(args)
        if args[:3] != ["sh", "-s", "--"]:
            raise AssertionError(f"Unexpected mutation during preflight: {args}")
        return subprocess.run(
            ["/bin/sh", *args[1:]], check=True, env={"PATH": str(tools)}, **kwargs
        )

    return remote, calls


class SSHTrainingTests(unittest.TestCase):
    def test_host_prerequisites_fail_before_upload_build_or_campaign_creation(self):
        cases = [
            ({"failure": "docker-missing"}, "Missing prerequisite: docker"),
            ({"failure": "os"}, "must run Linux"),
            ({"failure": "daemon"}, "Cannot access the Docker daemon"),
            ({"failure": "build"}, "image building is unavailable"),
            ({"count": 4}, "needs 8 GPUs; detected 4"),
            ({"capability": "7.5"}, "Ampere"),
            ({"driver": "575.51.03"}, "driver >= 580"),
            ({"failure": "output"}, "not writable"),
            ({"failure": "runtime"}, "CUDA container probe failed"),
            ({"failure": "bind"}, "cannot write"),
            ({"failure": "container-count"}, "needs 8 GPUs; detected 4"),
        ]
        for options, expected in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                remote, calls = fake_host(root, **options)
                args = Namespace(
                    config=None,
                    small=True,
                    smoke=False,
                    prefix="test-preflight",
                    ssh="gpu-box",
                    remote_root=str(root / "new output"),
                    output=root / "local",
                )
                with (
                    patch.object(ssh, "remote", side_effect=remote),
                    patch.object(ssh, "source_archive") as upload,
                    self.assertRaisesRegex((RuntimeError, ValueError), expected),
                ):
                    ssh.run_campaign(args)
                upload.assert_not_called()
                self.assertFalse(args.output.exists())
                self.assertFalse(Path(args.remote_root).exists())
                self.assertFalse(list(root.glob(".zephon-preflight.*")))
                if options.get("count") == 4:
                    self.assertEqual(len(calls), 1)
                    docker = (root / "docker-calls.jsonl").read_text()
                    self.assertNotIn('"run"', docker)

    def test_successful_preflight_probes_exact_allocation_mount_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote, _ = fake_host(root)
            # Shell metacharacters in a user path remain literal positional args.
            output = root / "space $(do-not-execute)" / "new"
            with patch.object(ssh, "remote", side_effect=remote):
                proof = ssh.preflight("gpu-box", PurePosixPath(output), 8)
            self.assertEqual(len(proof["host_gpus"]), 8)
            self.assertEqual(len(proof["container_gpus"]), 8)
            self.assertTrue(proof["writable_bind_mount"])
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".zephon-preflight.*")))
            calls = [
                json.loads(line)
                for line in (root / "docker-calls.jsonl").read_text().splitlines()
            ]
            probe = next(args for args in calls if args[0] == "run")
            self.assertIn("--rm", probe)
            self.assertEqual(probe[probe.index("--gpus") + 1], "8")
            self.assertEqual(probe[probe.index("--network") + 1], "none")
            self.assertIn(ssh.PROBE_IMAGE, probe)

    def test_ssh_preflight_timeout_is_actionable(self):
        with (
            patch.object(
                ssh, "remote", side_effect=subprocess.TimeoutExpired("ssh", 60)
            ),
            self.assertRaisesRegex(RuntimeError, "SSH preflight could not finish"),
        ):
            ssh.preflight("gpu-box", PurePosixPath("/scratch"), 8)

    def test_small_torchrun_commands_use_exact_gpu_counts_and_reject_multinode(self):
        for stage in protocol.plan(protocol.load(small=True)):
            args = ssh_runner.command(stage)
            self.assertIn(f"--nproc-per-node={stage['dp']}", args)
            self.assertIn("--standalone", args)
            self.assertIn("--nnodes=1", args)
        with self.assertRaisesRegex(ValueError, "one node"):
            ssh_runner.command(protocol.plan(protocol.load())[0])

    def test_ssh_preserves_argument_boundaries_and_rejects_host_options(self):
        args = ["mkdir", "--", "/scratch/a path/$(do-not-run)"]
        command = ssh.ssh_command("alice@gpu-box", args)
        self.assertEqual(shlex.split(command[-1]), args)
        for host in ["-oProxyCommand=anything", "host;command", "host extra"]:
            with self.assertRaises(ValueError):
                ssh.ssh_command(host, ["true"])

    def test_upload_contains_only_fingerprinted_training_sources_and_public_pins(self):
        manifest = {
            "config": protocol.load(small=True),
            "code": protocol.code_identity(),
        }
        with tarfile.open(fileobj=io.BytesIO(ssh.source_archive(manifest))) as archive:
            self.assertEqual(
                set(archive.getnames()), set(manifest["code"]["files"]) | {"Dockerfile"}
            )
            self.assertTrue(all(member.isfile() for member in archive.getmembers()))
            dockerfile = archive.extractfile("Dockerfile").read().decode()
        self.assertIn(manifest["config"]["sources"]["torchtitan"], dockerfile)
        self.assertIn("HF_HUB_DISABLE_IMPLICIT_TOKEN=1", dockerfile)
        self.assertNotIn("zephon_modal", dockerfile)

    def test_repeated_ssh_submission_attaches_without_another_build_or_container(self):
        calls = []
        info = {"id": "a" * 64, "image": "sha256:pinned", "state": {"Running": True}}

        def remote(host, args, **kwargs):
            calls.append(args)
            if args[:3] == ["sh", "-s", "--"]:
                return SimpleNamespace(stdout=inventory())
            if args[:2] == ["docker", "run"] and "--gpus" not in args:
                return SimpleNamespace(stdout=b"{}")
            if args[:2] == ["docker", "inspect"]:
                return SimpleNamespace(stdout=json.dumps(info).encode())
            return SimpleNamespace(stdout=("a" * 64 + "\n").encode())

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(ssh, "remote", side_effect=remote),
        ):
            args = Namespace(
                config=None,
                small=True,
                smoke=False,
                prefix="test-small",
                ssh="gpu-box",
                remote_root="/scratch/experiments",
                output=Path(tmp),
            )
            ssh.run_campaign(args)
            ssh.run_campaign(args)
            self.assertEqual(sum(a[:2] == ["docker", "build"] for a in calls), 0)
            checks = [
                a for a in calls if a[:2] == ["docker", "run"] and "--gpus" not in a
            ]
            self.assertEqual(len(checks), 1)
            self.assertEqual(checks[0][-1], "check")
            self.assertTrue(
                checks[0][checks[0].index("--mount") + 1].endswith(",readonly")
            )
            launches = [
                a for a in calls if a[:2] == ["docker", "run"] and "--gpus" in a
            ]
            self.assertEqual(len(launches), 1)
            self.assertEqual(launches[0][launches[0].index("--gpus") + 1], "8")
            receipt = json.loads((Path(tmp) / "launch.json").read_text())
            self.assertEqual(receipt["container_id"], info["id"])
            self.assertEqual(sum(a[:3] == ["sh", "-s", "--"] for a in calls), 2)
            self.assertEqual([a[:3] for a in calls[:2]], [["sh", "-s", "--"]] * 2)
            self.assertTrue((Path(tmp) / "preflight.json").exists())

    def test_active_or_uncertain_container_is_never_restarted(self):
        with (
            patch.object(
                ssh,
                "target",
                return_value=({}, {"id": "a", "state": {"Running": True}}),
            ),
            patch.object(ssh, "remote") as remote,
        ):
            ssh.continue_campaign(Path("unused"))
            remote.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "campaign.json").write_text(
                json.dumps({"backend": {"kind": "ssh"}})
            )
            (root / "launch.json").write_text(json.dumps({"state": "prepared"}))
            with self.assertRaisesRegex(RuntimeError, "Uncertain"):
                ssh.continue_campaign(root)

    def test_lost_launch_reply_reattaches_to_original_container(self):
        calls = []
        created = False
        info = {"id": "b" * 64, "image": "sha256:original", "state": {"Running": True}}

        def remote(host, args, **kwargs):
            nonlocal created
            calls.append(args)
            if args[:3] == ["sh", "-s", "--"]:
                return SimpleNamespace(stdout=inventory())
            if args[:2] == ["docker", "run"]:
                if "--gpus" not in args:
                    return SimpleNamespace(stdout=b"{}")
                created = True
                raise subprocess.CalledProcessError(255, args)
            if args[:2] == ["docker", "inspect"] and created:
                return SimpleNamespace(stdout=json.dumps(info).encode())
            return SimpleNamespace(stdout=b"")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(ssh, "remote", side_effect=remote),
        ):
            args = Namespace(
                config=None,
                small=True,
                smoke=False,
                prefix="test-lost-reply",
                ssh="gpu-box",
                remote_root="/scratch/experiments",
                output=Path(tmp),
            )
            with self.assertRaises(subprocess.CalledProcessError):
                ssh.run_campaign(args)
            self.assertEqual(
                json.loads((Path(tmp) / "launch.json").read_text())["state"], "prepared"
            )
            ssh.run_campaign(args)
            self.assertEqual(
                sum(
                    args[:2] == ["docker", "run"] and "--gpus" in args for args in calls
                ),
                1,
            )
            self.assertEqual(
                json.loads((Path(tmp) / "launch.json").read_text())["container_id"],
                info["id"],
            )

    def test_archive_rejects_path_traversal_and_links(self):
        for name, kind in [("../escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
            raw = io.BytesIO()
            with tarfile.open(fileobj=raw, mode="w") as archive:
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
                archive.addfile(member)
            with (
                tempfile.TemporaryDirectory() as tmp,
                self.assertRaisesRegex(ValueError, "regular files"),
            ):
                ssh.unpack_evidence(raw.getvalue(), Path(tmp), [name])

    def test_collection_strictly_validates_before_publishing_any_raw_evidence(self):
        for corrupt in [False, True]:
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, local = root / "remote", root / "local"
                source.mkdir()
                local.mkdir()
                config = protocol.load(smoke=True)
                prepare_report_fixture(source, config, "fixture")
                manifest = json.loads((source / "campaign.json").read_text())
                manifest["backend"] = {
                    "kind": "ssh",
                    "host": "gpu-box",
                    "directory": "/scratch/campaign",
                }
                content = json.dumps(manifest)
                (source / "campaign.json").write_text(content)
                (local / "campaign.json").write_text(content)
                for stage in protocol.plan(config):
                    (source / stage["name"] / "launch.json").write_text("{}")
                (source / "result.json").write_text("{}")
                if corrupt:
                    (
                        source
                        / protocol.plan(config)[1]["name"]
                        / "diagnostics/rank-0.jsonl"
                    ).write_text("incomplete\n")
                raw = io.BytesIO()
                with tarfile.open(fileobj=raw, mode="w") as archive:
                    for name in ssh.evidence_files(config):
                        archive.add(source / name, arcname=name)
                with (
                    patch.object(ssh, "target", return_value=(manifest, {})),
                    patch.object(ssh, "status", return_value={"state": "complete"}),
                    patch.object(
                        ssh,
                        "remote",
                        return_value=SimpleNamespace(stdout=raw.getvalue()),
                    ),
                ):
                    if corrupt:
                        with self.assertRaises(ValueError):
                            ssh.collect(local)
                        self.assertFalse(any(local.glob("*/diagnostics/*")))
                    else:
                        self.assertTrue(ssh.collect(local)["expected_outcomes_met"])

    def test_completed_remote_stages_are_validated_and_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = protocol.load(smoke=True)
            prepare_report_fixture(root, config, "fixture")
            manifest = json.loads((root / "campaign.json").read_text())
            manifest["code"] = protocol.code_identity()
            (root / "campaign.json").write_text(json.dumps(manifest))
            for stage in protocol.plan(config):
                (root / stage["name"] / "launch.json").write_text(
                    json.dumps(
                        {"state": "complete", "directory": str(root / stage["name"])}
                    )
                )

            def publish(directory, destination, stage, config):
                return json.loads((destination / "checkpoint-proof.json").read_text())

            assets = json.loads((root / "assets.json").read_text())
            with (
                patch.object(ssh_runner, "preflight", return_value={}),
                patch.object(ssh_runner, "validate_assets", return_value=assets),
                patch.object(ssh_runner, "publish", side_effect=publish),
                patch.object(ssh_runner.subprocess, "run") as run,
            ):
                self.assertTrue(
                    ssh_runner.coordinate(root, root / "assets")[
                        "expected_outcomes_met"
                    ]
                )
                run.assert_not_called()

    def test_complete_unpublished_attempt_is_reused_after_controller_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = protocol.load(smoke=True)
            prepare_report_fixture(root, config, "fixture")
            manifest = json.loads((root / "campaign.json").read_text())
            manifest["code"] = protocol.code_identity()
            (root / "campaign.json").write_text(json.dumps(manifest))
            stages = protocol.plan(config)
            attempt = "a" * 32
            originals = {}
            for index, stage in enumerate(stages):
                directory = root / stage["name"]
                record = {"state": "complete", "directory": str(directory)}
                if index == 0:
                    record.update(state="running", attempt=attempt)
                    for rank in range(stage["dp"]):
                        source = directory / "diagnostics" / f"rank-{rank}.jsonl"
                        raw = source.with_name(f"{attempt}.rank-{rank}.jsonl")
                        source.rename(raw)
                        originals[raw] = raw.read_bytes()
                (directory / "launch.json").write_text(json.dumps(record))
            request_id = "b" * 32
            (root / "continue.json").write_text(
                json.dumps({"identity": "fixture", "request_id": request_id})
            )

            def publish(directory, destination, stage, config):
                return json.loads((destination / "checkpoint-proof.json").read_text())

            assets = json.loads((root / "assets.json").read_text())
            with (
                patch.object(ssh_runner, "preflight", return_value={}),
                patch.object(ssh_runner, "validate_assets", return_value=assets),
                patch.object(ssh_runner, "publish", side_effect=publish),
                patch.object(
                    ssh_runner.trace,
                    "checkpoint_proof",
                    return_value={"shards": {"native.distcp": 10}},
                ) as checkpoint,
                patch.object(ssh_runner.subprocess, "run") as run,
            ):
                self.assertTrue(
                    ssh_runner.coordinate(root, root / "assets")[
                        "expected_outcomes_met"
                    ]
                )
                run.assert_not_called()
                self.assertEqual(checkpoint.call_count, 2)
            self.assertEqual({path: path.read_bytes() for path in originals}, originals)
            self.assertTrue((root / "continuations" / f"{request_id}.json").exists())


if __name__ == "__main__":
    unittest.main()
