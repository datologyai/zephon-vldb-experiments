"""Input preparation uses the training target, preserves bytes and needs no GPUs."""

import json
import subprocess
import sys
from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vldb_experiments import cli
from vldb_experiments.training import assets, preparation, protocol, ssh


def arguments(**overrides):
    return Namespace(
        **(
            dict(
                config=None,
                small=True,
                smoke=False,
                prefix="test-inputs",
                ssh="gpu-box",
                remote_root="/scratch/vldb",
            )
            | overrides
        )
    )


def test_preparation_preview_needs_no_training_dependencies_or_target(capsys):
    with (
        patch.dict(sys.modules, modal=None, zephon=None, torch=None),
        patch.object(ssh, "remote", side_effect=AssertionError("target access")),
    ):
        assert (
            cli.main(["prepare", "training-data", "--prefix", "test-inputs", "--plan"])
            == 0
        )
        result = json.loads(capsys.readouterr().out)
        assert result["gpus"] == 0 and len(result["files"]) == 2
        assert "plots" not in result
        assert result["destination"] == "test-inputs-training-assets (Modal volume)"
        assert (
            cli.main(
                [
                    "prepare",
                    "training-data",
                    "--prefix",
                    "test-inputs",
                    "--ssh",
                    "gpu-box",
                    "--remote-root",
                    "/scratch/vldb",
                    "--plan",
                ]
            )
            == 0
        )
        assert (
            json.loads(capsys.readouterr().out)["destination"]
            == "/scratch/vldb/test-inputs-training-assets"
        )


def test_check_dispatches_to_the_same_training_target_without_preparing(
    capsys, tmp_path
):
    receipt = tmp_path / "check.json"
    with (
        patch.object(
            preparation, "ssh_inputs", return_value={"data/test.parquet": "sha"}
        ) as check,
    ):
        assert (
            cli.main(
                [
                    "check",
                    "training",
                    "--small",
                    "--prefix",
                    "test-inputs",
                    "--ssh",
                    "gpu-box",
                    "--remote-root",
                    "/scratch/vldb",
                    "--output",
                    str(receipt),
                ]
            )
            == 0
        )
    assert check.call_args.args[2] == "check"
    assert json.loads(receipt.read_text())["host"] == "gpu-box"
    assert "checks passed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command,component,action",
    [
        ("prepare", "training-data", "prepare"),
        ("check", "training", "check"),
    ],
)
def test_training_inputs_need_no_plotting_setup(command, component, action):
    from vldb_experiments import prepare

    flags = [
        "--small",
        "--prefix",
        "test-inputs",
        "--ssh",
        "gpu-box",
        "--remote-root",
        "/scratch/vldb",
    ]
    with (
        patch.object(prepare, "plotting", side_effect=AssertionError("setup")),
        patch(
            "vldb_experiments.report.plotting_python",
            side_effect=AssertionError("plotting check"),
        ),
        patch.object(preparation, "ssh_inputs", return_value={}) as inputs,
    ):
        assert cli.main([command, component, *flags]) == 0
    assert inputs.call_args.args[2] == action


@pytest.mark.parametrize(
    "command,component",
    [
        ("prepare", "training-data"),
        ("check", "training"),
    ],
)
def test_training_input_commands_do_not_accept_plotting_options(command, component):
    with pytest.raises(SystemExit):
        cli.parser().parse_args(
            [command, component, "--prefix", "test-inputs", "--plots", "/plots"]
        )


def test_invalid_training_target_fails_before_remote_access():
    for flags in (
        ("--remote-root", "/scratch"),
        ("--ssh", "gpu-box"),
        ("--ssh", "-bad", "--remote-root", "/scratch"),
    ):
        with patch.object(ssh, "remote") as remote:
            with pytest.raises((ValueError, SystemExit)):
                cli.main(
                    [
                        "prepare",
                        "training-data",
                        "--prefix",
                        "test-inputs",
                        "--plan",
                        *flags,
                    ]
                )
            remote.assert_not_called()


def test_ssh_preparation_uses_shared_image_and_a_cpu_container():
    calls = []

    def remote(host, command, **kwargs):
        calls.append(command)
        output = (
            b"/tmp/zephon-training-data.12345678\n" if command[0] == "mktemp" else b"{}"
        )
        return SimpleNamespace(stdout=output)

    config = protocol.load(small=True)
    args = arguments()
    with patch.object(ssh, "remote", side_effect=remote):
        preparation.ssh_inputs(args, config, "prepare")
    run = next(c for c in calls if c[:2] == ["docker", "run"])
    build = next(c for c in calls if c[:2] == ["docker", "build"])
    assert "--gpus" not in run and "--detach" not in run
    assert build[build.index("--tag") + 1] in run
    assert "type=bind,src=/scratch/vldb/test-inputs-training-assets,dst=/assets" in run
    assert not any("/scratch/vldb/test-inputs" == item for c in calls for item in c)
    assert calls[-1][-1] == "prepare"


def test_ssh_check_mounts_inputs_read_only_and_does_not_build_or_download():
    with patch.object(
        ssh, "remote", return_value=SimpleNamespace(stdout=b"{}")
    ) as remote:
        preparation.ssh_inputs(arguments(), protocol.load(small=True), "check")
    remote.assert_called_once()
    command = remote.call_args.args[1]
    assert "--gpus" not in command and "--network" in command and "none" in command
    assert command[-1] == "check"
    assert command[command.index("--mount") + 1].endswith(",readonly")


def test_unprepared_inputs_fail_before_campaign_creation_or_gpu_work(tmp_path):
    calls = []

    def remote(host, command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=b"{}")

    args = arguments(output=tmp_path)
    with (
        patch.object(ssh, "preflight", return_value={}),
        patch.object(ssh, "remote", side_effect=remote),
        patch.object(
            ssh,
            "input_command",
            side_effect=subprocess.CalledProcessError(1, ["check"]),
        ),
        pytest.raises(RuntimeError, match="reproduce prepare training-data"),
    ):
        ssh.run_campaign(args)
    assert not any(c[:2] == ["docker", "run"] and "--gpus" in c for c in calls)
    assert not (tmp_path / "launch.json").exists()
    assert not (tmp_path / "campaign.json").exists()
    assert not any(c[:2] == ["docker", "build"] for c in calls)
    assert not any(c[0] == "mkdir" for c in calls)


@pytest.mark.parametrize(
    "error", [FileNotFoundError("missing inputs"), ValueError("checksum mismatch")]
)
def test_modal_run_requires_inputs_before_claiming_or_spawning_a_campaign(
    tmp_path, error
):
    from vldb_experiments.training import cli as training_cli

    with (
        patch.dict(sys.modules, modal=SimpleNamespace()),
        patch.object(
            training_cli, "dependency_identity", return_value={"commit": "fixture"}
        ),
        patch.object(preparation, "modal_inputs", side_effect=error),
        patch("vldb_experiments.training.modal_app.build_app") as build,
        patch.object(assets, "stage_assets", side_effect=AssertionError("download")),
        pytest.raises(RuntimeError, match="reproduce prepare training-data"),
    ):
        training_cli.run_campaign(
            arguments(ssh=None, remote_root=None, output=tmp_path)
        )
    build.assert_not_called()
    assert not (tmp_path / "campaign.json").exists()
    assert not (tmp_path / "launch.json").exists()


def test_input_validation_is_read_only_even_when_preparation_is_missing(tmp_path):
    config = protocol.load()
    root = tmp_path / "missing-assets"
    with (
        patch.object(assets, "stage_assets", side_effect=AssertionError("download")),
        pytest.raises(ValueError, match="reproduce prepare training-data"),
    ):
        assets.validate_assets(config, root)
    assert not root.exists()


def test_ssh_runner_never_downloads_or_starts_ranks_on_missing_inputs(tmp_path):
    from vldb_experiments.training import ssh_runner

    (tmp_path / "campaign.json").write_text(
        json.dumps(
            {
                "config": protocol.load(smoke=True),
                "code": protocol.code_identity(),
                "identity": "fixture",
            }
        )
    )
    with (
        patch.object(ssh_runner, "preflight", return_value={}),
        patch.object(ssh_runner.subprocess, "run") as ranks,
        patch.object(assets, "stage_assets", side_effect=AssertionError("download")),
        pytest.raises(ValueError, match="reproduce prepare training-data"),
    ):
        ssh_runner.coordinate(tmp_path, tmp_path / "missing-assets")
    ranks.assert_not_called()
    assert not (tmp_path / "missing-assets").exists()


@pytest.mark.parametrize("action", ["prepare", "check"])
def test_modal_preparation_uses_same_volume_and_image_without_gpu_requests(action):
    task = Mock()
    task.remote.return_value = {"data/test.parquet": "sha"}
    app = Mock()
    app.function.return_value = lambda function: task
    app.run.return_value = nullcontext()
    volume = Mock()
    volume.with_mount_options.return_value = volume
    modal = SimpleNamespace(
        App=Mock(return_value=app),
        Volume=SimpleNamespace(from_name=Mock(return_value=volume)),
        enable_output=nullcontext,
    )
    config = protocol.load()
    with (
        patch.dict(sys.modules, modal=modal),
        patch(
            "vldb_experiments.training.modal_app.training_image",
            return_value="shared-image",
        ),
    ):
        preparation.modal_inputs(arguments(ssh=None, remote_root=None), config, action)
    options = app.function.call_args.kwargs
    assert "gpu" not in options and options["image"] == "shared-image"
    modal.Volume.from_name.assert_called_once_with(
        "test-inputs-training-assets", create_if_missing=action == "prepare", version=2
    )
    if action == "check":
        volume.with_mount_options.assert_called_once_with(read_only=True)
    task.remote.assert_called_once_with(config, "test-inputs", action)


def test_real_parquet_preparation_preserves_rows_and_reuses_files_and_index(tmp_path):
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("zephon.build_index")
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard = tmp_path / "source.parquet"
    table = pa.table({"text": ["first document", "second document", "third document"]})
    pq.write_table(table, shard)
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"test":"tokenizer"}')
    config = {
        kind: dict(
            repository="example/public",
            revision="a" * 40,
            files={path.name: protocol.file_digest(path)},
        )
        for kind, path in (("dataset", shard), ("tokenizer", tokenizer))
    }
    root = tmp_path / "assets"
    with patch(
        "huggingface_hub.hf_hub_download",
        side_effect=lambda repo, name, **kw: tmp_path / name,
    ) as download:
        receipt = assets.stage_assets(config, root)
    assert all(c.kwargs["token"] is False for c in download.call_args_list)
    assert pq.read_table(root / "data/source.parquet").equals(table)
    assert (root / "data/source.parquet").read_bytes() == shard.read_bytes()
    before = {
        p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }
    with (
        patch(
            "huggingface_hub.hf_hub_download", side_effect=AssertionError("download")
        ),
        patch("zephon.build_index.build_index", side_effect=AssertionError("reindex")),
    ):
        assert assets.stage_assets(config, root) == receipt
        assert assets.validate_assets(config, root) == receipt
    assert {
        p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    } == before
    index_path = root / "data/index.json"
    index = json.loads(index_path.read_text())
    index["shards"][0]["basename"] = "wrong.parquet"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index differs"):
        assets.validate_assets(config, root)
    index["shards"][0]["basename"] = "source.parquet"
    index_path.write_text(json.dumps(index))
    (root / "data/extra.parquet").write_bytes(shard.read_bytes())
    with pytest.raises(ValueError, match="Unexpected Parquet"):
        assets.validate_assets(config, root)
    (root / "data/extra.parquet").unlink()
    (root / "tokenizer/tokenizer.json").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        assets.validate_assets(config, root)
