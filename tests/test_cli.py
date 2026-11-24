"""The public CLI previews recipes and runs only validated selections."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from vldb_experiments import cli, core, prepare, provenance


def machine_profile(tmp_path, **extra):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(dict(machine=dict(kind="local", root=str(tmp_path)), **extra))
    )
    return path


def test_plans_need_no_profile_external_repositories_or_network(tmp_path):
    for experiment, count in [("throughput", 387), ("ttfb", 345), ("checkpoint", 435)]:
        output = tmp_path / (experiment + ".json")
        with patch.object(
            cli.machines, "check", side_effect=AssertionError("target access")
        ):
            assert cli.main(["plan", experiment, "--output", str(output)]) == 0
        assert core.read_json(output)["count"] == count


def test_run_launches_complete_figures_in_separate_directories(tmp_path):
    profile = machine_profile(tmp_path)
    with (
        patch.object(cli, "plotting_python"),
        patch.object(core, "execute", return_value=0) as execute,
        patch.object(cli.summary, "finish") as finish,
    ):
        assert (
            cli.main(
                [
                    "run",
                    "throughput",
                    "--profile",
                    str(profile),
                    "--output",
                    str(tmp_path / "results"),
                ]
            )
            == 0
        )
    assert [len(call.args[0]) for call in execute.call_args_list] == [294, 93]
    assert [call.args[3] for call in execute.call_args_list] == [
        tmp_path / "results" / experiment for experiment in cli.EXPERIMENTS
    ]
    finish.assert_called_once_with(
        tmp_path / "results", plots=tmp_path / "repositories/plots"
    )


def test_unprepared_plot_environment_fails_before_measurement(tmp_path):
    profile = machine_profile(tmp_path)
    with (
        patch.object(core, "execute") as execute,
        pytest.raises(RuntimeError, match="reproduce prepare plotting"),
    ):
        cli.main(
            [
                "run",
                "throughput",
                "--profile",
                str(profile),
                "--output",
                str(tmp_path / "run"),
            ]
        )
    execute.assert_not_called()


def test_smoke_never_attempts_paper_figures(tmp_path):
    profile = machine_profile(tmp_path)
    with (
        patch.object(core, "execute", return_value=0),
        patch.object(cli.summary, "finish") as finish,
    ):
        cli.main(
            [
                "run",
                "throughput",
                "--profile",
                str(profile),
                "--output",
                str(tmp_path / "run"),
                "--smoke",
            ]
        )
    finish.assert_not_called()


def test_sources_include_all_public_dependencies(tmp_path):
    profile = cli.profiles.load(machine_profile(tmp_path))
    with patch.object(prepare, "export_source") as export:
        prepare.sources(profile, "v0.1.0")
    assert {call.args[0]["repository"] for call in export.call_args_list} == {
        "datologyai/zephon",
        "datologyai/dataloader-arena-public",
        "datologyai/zephon-vldb-plots",
    }


def test_supplied_revision_is_installed_without_replacing_sources(tmp_path):
    profile = cli.profiles.load(machine_profile(tmp_path))
    (tmp_path / "environments/cpu-3.13").mkdir(parents=True)
    with (
        patch.object(provenance, "source_state", return_value={"revision": "custom"}),
        patch.object(prepare, "call") as call,
        patch.object(prepare, "pip_install") as install,
        patch.object(
            prepare.subprocess, "check_output", return_value="zephon==1.2.3\n"
        ),
    ):
        prepare.environments(profile, "v0.1.0", ["cpu-3.13"])
    command = call.call_args.args[0]
    assert command[command.index("--no-install-package") + 1] == "zephon"
    assert install.call_args.args[-1] == Path(profile["paths"]["zephon"])


def test_shared_environment_uses_lock_and_exact_paper_extras(tmp_path):
    profile = cli.profiles.load(machine_profile(tmp_path))
    env = tmp_path / "environments/cpu-3.13"
    env.mkdir(parents=True)
    with (
        patch.object(
            provenance,
            "source_state",
            side_effect=[
                {"revision": "4ead2f082cac6ae3e902638151cedd4b291cb6fb"},
                {"revision": "16a71891c7751169fa096c352a7f90fd0ac8be53"},
            ],
        ),
        patch.object(prepare, "call") as call,
        patch.object(prepare, "pip_install") as install,
        patch.object(
            prepare.subprocess, "check_output", return_value="zephon==0.1.0\n"
        ),
    ):
        prepare.environments(profile, "v0.1.0", ["cpu-3.13"])
    command = call.call_args.args[0]
    assert "--locked" in command
    assert {command[i + 1] for i, arg in enumerate(command) if arg == "--extra"} == {
        "zephon",
        "grain",
        "litdata",
        "vlm-pipeline",
        "sweep",
        "checkpoint",
        "data-preparation",
    }
    assert call.call_args.kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(env)
    install.assert_called_once()
    assert install.call_args.args[-1] == tmp_path / "repositories/zephon"
