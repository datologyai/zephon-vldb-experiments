"""Machine setup and cloud launch requests are safe to preview and repeat."""

import fcntl
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vldb_experiments import cli, core, machines, prepare, profiles


def aws_profile(tmp_path):
    raw = dict(
        machine=dict(kind="aws", root=str(tmp_path / "target"), connect="ssm"),
        aws=dict(
            name="experiment",
            security_group="reviewer-sg",
            key_pair_name="reviewer-key",
            iam_instance_profile="reviewer-role",
        ),
        controller=dict(arena=str(tmp_path / "arena")),
    )
    path = tmp_path / "aws.json"
    path.write_text(json.dumps(raw))
    return path, raw


def test_plan_and_prepare_preview_do_not_connect_or_install(tmp_path):
    path, raw = aws_profile(tmp_path)
    with (
        patch("vldb_experiments.remote.execute", side_effect=AssertionError("SSH")),
        patch.object(prepare, "call", side_effect=AssertionError("installation")),
        patch.object(machines, "provisioner", side_effect=AssertionError("AWS")),
    ):
        assert cli.main(["machine", "create", "--profile", str(path), "--plan"]) == 0
        assert cli.main(["machine", "setup", "--profile", str(path), "--plan"]) == 0
        assert (
            cli.main(["prepare", "checkpoint-data", "--profile", str(path), "--plan"])
            == 0
        )


def test_same_primary_environment_is_selected_by_all_loaders_and_checkpoint(tmp_path):
    raw = dict(machine=dict(root=str(tmp_path)))
    resolved = profiles.resolve(raw)
    expected = tmp_path / "environments/cpu-3.13/bin/python"
    assert Path(resolved["checkpoint"]["python"]) == expected
    for system in ("zephon", "grain", "litdata"):
        assert (
            core.environment_path(
                resolved, dict(system=system, python="3.13"), "v0.1.0"
            )
            == expected
        )
    assert (
        profiles.resolve(raw, "ttfb")["paths"]["vlm"]
        != profiles.resolve(raw, "throughput")["paths"]["vlm"]
    )


def test_separate_checkpoint_environment_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="shared CPU setup"):
        profiles.resolve(
            dict(
                machine=dict(root=str(tmp_path)),
                checkpoint=dict(python="/old-checkpoint-env/bin/python"),
            )
        )


@pytest.mark.parametrize(
    "section,key",
    [
        ("machine", "root"),
        ("paths", "arena"),
        ("ttfb_data", "vlm"),
        ("checkpoint_data", "journal"),
    ],
)
def test_remote_paths_cannot_expand_on_controller(tmp_path, section, key):
    raw = dict(machine=dict(kind="ssh", root="/target", host="host"))
    raw.setdefault(section, {})[key] = "~/relative"
    with pytest.raises(ValueError, match="absolute"):
        profiles.local_target(raw)


def test_setup_refuses_to_mutate_an_active_measurement(tmp_path):
    profile = profiles.resolve(dict(machine=dict(root=str(tmp_path))))
    with Path(profile["measurement"]["lock"]).open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (
            patch.object(machines, "check", return_value={"ready": True}),
            patch.object(prepare, "sources") as sources,
        ):
            with pytest.raises(RuntimeError, match="measuring"):
                machines.setup(profile)
        sources.assert_not_called()


def test_aws_calls_arena_once_and_preserves_launch_identity(tmp_path):
    path, raw = aws_profile(tmp_path)
    arena = Path(raw["controller"]["arena"])
    arena.mkdir()
    (arena / "README.md").write_text("fixture")
    (arena / "SOURCE.json").write_text(
        json.dumps(
            dict(
                revision="4ead2f082cac6ae3e902638151cedd4b291cb6fb",
                files={"README.md": hashlib.sha256(b"fixture").hexdigest()},
            )
        )
    )
    module = SimpleNamespace(
        LOGGER=SimpleNamespace(handlers=[]),
        UBUNTU_OWNER_ID="099720109477",
        load_user_data=lambda _: (
            "#!/bin/bash\nclone_repo_to_nvme() { true; }\nif true; then\n    clone_repo_to_nvme\nfi\n"
        ),
        inject_cstate_setup=lambda value: value,
        inject_open_file_limit=lambda value, _: value,
        inject_perf_stabilize=lambda value: value,
        LaunchConfig=lambda **kw: kw,
        create_instance=Mock(return_value=("i-owned-test", "10.0.0.1")),
    )
    with patch.object(machines, "provisioner", return_value=module):
        first = machines.create(path)
        assert machines.create(path) == first
    module.create_instance.assert_called_once()
    config = module.create_instance.call_args.args[0]
    assert config["user_data"].startswith("#!/bin/bash\n")
    assert (
        "https://github.com/datologyai/dataloader-arena-public.git"
        in config["user_data"]
    )
    assert config["tags"]["vldb-owner"] == first["owner"]
    assert config["shutdown_behavior"] == "stop"
    assert first["bootstrap_repo_clone_skipped"] is True
    assert "\n    clone_repo_to_nvme\n" not in config["user_data"]
    subprocess.run(["bash", "-n"], input=config["user_data"], text=True, check=True)
    old = core.read_json(machines.receipt_path(path, raw))
    old["request"]["arena_revision"] = old["request"].pop("default_arena_revision")
    core.write_json(machines.receipt_path(path, raw), old)
    with patch.object(
        machines, "provisioner", side_effect=AssertionError("duplicate AWS launch")
    ):
        assert machines.create(path) == old
    raw["aws"]["root_volume_size"] = 300
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="request changed"):
        machines.create(path)


def test_uncertain_aws_launch_cannot_create_another_instance(tmp_path):
    path, raw = aws_profile(tmp_path)
    core.write_json(
        machines.receipt_path(path, raw),
        dict(request=machines.aws_plan(raw), state="launching", owner="test"),
    )
    with (
        patch.object(machines, "provisioner") as load,
        pytest.raises(RuntimeError, match="unresolved"),
    ):
        machines.create(path)
    load.assert_not_called()


def test_environment_check_rejects_missing_packages_and_wrong_python():
    from copy import deepcopy

    state = dict(
        python="3.13.15 (main build)",
        free_threaded=False,
        packages={
            name: "installed"
            for name in (
                "torch",
                "transformers",
                "pillow",
                "obstore",
                "psutil",
                "pandas",
                "grain",
                "pyarrow",
                "ijson",
                "huggingface-hub",
                "tokenizers",
                "boto3",
                "zstd",
            )
        },
        editable_sources={
            "zephon": {"revision": "16a71891c7751169fa096c352a7f90fd0ac8be53"}
        },
    )
    state["packages"].update(zephon="0.1.0", litdata="0.2.61")
    path = "/environments/cpu-3.13/bin/python"
    prepare.validate_environments({path: state})
    for key, value, expected in [
        ("python", "3.12.3", "Python"),
        ("free_threaded", True, "GIL"),
    ]:
        broken = deepcopy(state)
        broken[key] = value
        with pytest.raises(ValueError, match=expected):
            prepare.validate_environments({path: broken})
    del state["packages"]["grain"]
    with pytest.raises(ValueError, match="Missing shared CPU package grain"):
        prepare.validate_environments({path: state})
