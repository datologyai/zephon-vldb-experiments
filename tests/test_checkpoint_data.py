"""The paper launcher rejects incorrect storage/scale before measuring."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from vldb_experiments import checkpoint_data, cli


def profile():
    return {
        "arena": "/public/arena",
        "python": "/envs/cpu-3.13/bin/python",
        "dataset": "s3://public/data",
        "data_preparation": {
            "source": "public",
            "scratch": "/scratch/checkpoint",
            "journal": "/persistent/checkpoint",
        },
    }


def fake_command(command, **kwargs):
    assert kwargs == {"check": True}
    output = Path(command[command.index("--receipt") + 1])
    output.write_text(json.dumps({"valid": True, "index_sha256": "test-index"}))


def test_full_recipe_is_paper_scale_and_keeps_fixed_source_pin():
    settings = checkpoint_data.settings()
    assert settings["samples"] == 9976290
    assert settings["chunk_bytes"] == 64 * 1024**2
    assert settings["scratch_bytes"] == 128 * 1024**3
    assert settings["minimum_shards"] < 472260 < settings["maximum_shards"]
    assert settings["minimum_records"] > 262144
    assert len(settings["revision"]) == 40


@pytest.mark.parametrize("flag", ["--plan", "--yes", "--abandon"])
def test_prepare_cli_uses_shared_machine_profile(tmp_path, flag):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            dict(
                machine=dict(kind="local", root=str(tmp_path)),
                checkpoint=dict(dataset="s3://bucket/data"),
            )
        )
    )
    with patch.object(checkpoint_data, "execute") as execute:
        cli.main(["prepare", "checkpoint-data", "--profile", str(path), flag])
    if flag == "--plan":
        execute.assert_not_called()
    else:
        execute.assert_called_once_with(
            checkpoint_data.load_profile(path),
            yes=flag == "--yes",
            abandon=flag == "--abandon",
        )


def test_plan_needs_no_aws_or_generator_dependencies():
    with (
        patch.object(
            checkpoint_data, "generator", return_value=Path("/arena/generator.py")
        ),
        patch.object(subprocess, "run", side_effect=fake_command) as run,
    ):
        checkpoint_data.execute(profile(), plan=True)
    command = run.call_args.args[0]
    assert "uv" not in command and "--plan" in command
    assert "--yes" not in command


def test_preparation_and_check_preserve_required_flags():
    with (
        patch.object(
            checkpoint_data, "generator", return_value=Path("/arena/generator.py")
        ),
        patch.object(subprocess, "run", side_effect=fake_command) as run,
    ):
        checkpoint_data.execute(profile(), yes=True)
        command = run.call_args.args[0]
        assert command[:2] == [profile()["python"], "/arena/generator.py"]
        assert "uv" not in command
        assert "--yes" in command and "--scratch" in command
        checkpoint_data.execute(profile(), yes=True, abandon=True)
        assert "--abandon" in run.call_args.args[0]
        checkpoint_data.execute(profile(), check=True, all_objects=True)
        assert "--check-only" in run.call_args.args[0]
        assert "--all-objects" in run.call_args.args[0]


def test_existing_export_is_explicit_and_is_not_regenerated():
    p = profile()
    p["data_preparation"]["source"] = "existing"
    with (
        patch.object(
            checkpoint_data, "generator", return_value=Path("/arena/generator.py")
        ),
        patch.object(subprocess, "run", side_effect=fake_command) as run,
    ):
        with pytest.raises(ValueError, match="Existing export"):
            checkpoint_data.execute(p, yes=True)
        run.assert_not_called()
        checkpoint_data.execute(p, check=True)
        assert "--existing" in run.call_args.args[0]


def test_checkpoint_check_is_a_real_cli_command():
    args = cli.parser().parse_args(
        ["check", "checkpoint", "--profile", "checkpoint.toml", "--all-objects"]
    )
    assert (
        args.command == "check" and args.experiment == "checkpoint" and args.all_objects
    )


@pytest.mark.parametrize(
    "uri", ["/data/local", "file:///data/local", "s3:///data", "s3://bucket"]
)
def test_bad_dataset_storage_is_rejected_before_subprocess(uri):
    p = profile() | {"dataset": uri}
    with (
        patch.object(subprocess, "run") as run,
        pytest.raises(ValueError, match="s3://"),
    ):
        checkpoint_data.execute(p, plan=True)
    run.assert_not_called()


def test_missing_generator_does_not_fall_back_to_other_preparation(tmp_path):
    with pytest.raises(ValueError, match="script is missing"):
        checkpoint_data.generator(profile() | {"arena": str(tmp_path)})


def test_relative_preparation_paths_are_anchored_to_profile(tmp_path):
    profile_path = tmp_path / "profiles/checkpoint.toml"
    profile_path.parent.mkdir()
    profile_path.write_text(
        '[paths]\narena="../arena"\n[checkpoint]\ndataset="s3://bucket/data"\n[checkpoint_data]\nscratch="../scratch"\njournal="../journal"\ncache_dir="../cache"\nshuffle_storage="../shuffle"\n'
    )
    p = checkpoint_data.load_profile(profile_path)
    assert p["arena"] == str(tmp_path / "arena")
    assert {k: v for k, v in p["data_preparation"].items() if k != "source"} == {
        key: str(tmp_path / value)
        for key, value in (
            ("scratch", "scratch"),
            ("journal", "journal"),
            ("cache_dir", "cache"),
            ("shuffle_storage", "shuffle"),
        )
    }


def test_plan_and_generation_use_same_storage_options():
    p = profile()
    with (
        patch.object(
            checkpoint_data, "generator", return_value=Path("/arena/prepare_data.py")
        ),
        patch.object(subprocess, "run", side_effect=fake_command) as run,
    ):
        checkpoint_data.execute(p, plan=True)
        plan = run.call_args.args[0]
        checkpoint_data.execute(p, yes=True)
        generate = run.call_args.args[0]
    for option in (
        "--output",
        "--scratch",
        "--journal",
        "--shuffle-storage",
        "--cache-dir",
    ):
        assert plan[plan.index(option) + 1] == generate[generate.index(option) + 1]
    assert plan[plan.index("--shuffle-storage") + 1] == p["dataset"] + "/_shuffle"
