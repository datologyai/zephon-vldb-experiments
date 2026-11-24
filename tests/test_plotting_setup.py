"""Plotting bootstraps from public pinned sources or reuses supplied checkouts."""

import io
import hashlib
import json
import tarfile
import tomllib
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from vldb_experiments import ROOT, cli, core, prepare, provenance, report, summary

PIN = tomllib.loads((ROOT / "configs/shared/sources.toml").read_text())["plots"][
    "revision"
]


def test_plotting_preparation_is_shared_and_needs_no_machine_profile(tmp_path):
    plots = tmp_path / "existing plots"
    with patch.object(prepare, "plotting") as setup:
        assert cli.main(["prepare", "plotting"]) == 0
        setup.assert_called_once_with(
            prepare.controller_plots({}).resolve(), existing=False
        )
        setup.reset_mock()
        assert cli.main(["prepare", "plotting", "--plots", str(plots)]) == 0
        setup.assert_called_once_with(plots, existing=True)


def test_plotting_preparation_preview_does_not_install_or_fetch(tmp_path, capsys):
    plots = tmp_path / "plots"
    with patch.object(prepare, "plotting") as setup:
        assert cli.main(["prepare", "plotting", "--plots", str(plots), "--plan"]) == 0
    setup.assert_not_called()
    assert json.loads(capsys.readouterr().out) == {"plots": str(plots)}
    assert not plots.exists()


def test_missing_plotting_environment_points_to_shared_setup(tmp_path):
    plots = tmp_path / "missing plots"
    with pytest.raises(RuntimeError, match="reproduce prepare plotting") as failure:
        report.plotting_python(plots)
    assert "--plots" in str(failure.value) and str(plots) in str(failure.value)
    assert not plots.exists()


def source_archive():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, value in {
            "pyproject.toml": b"[project]\nname='plotting'\n",
            "plotting/__init__.py": b"",
            "uv.lock": b"plotting repository lock",
        }.items():
            member = tarfile.TarInfo("github-export/" + name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    return stream.getvalue()


def test_first_setup_fetches_public_pin_and_repeat_reuses_snapshot(tmp_path):
    destination = tmp_path / "plots"
    with (
        patch.object(
            prepare.urllib.request, "urlopen", return_value=io.BytesIO(source_archive())
        ) as download,
        patch.object(prepare, "call") as install,
        patch("vldb_experiments.report.plotting_python") as check,
    ):
        assert prepare.plotting(destination) == destination
        assert (
            download.call_args.args[0].full_url
            == f"https://api.github.com/repos/datologyai/zephon-vldb-plots/tarball/{PIN}"
        )
        before = {
            str(p.relative_to(destination)): p.read_bytes()
            for p in destination.rglob("*")
            if p.is_file()
        }
        prepare.plotting(destination)
        assert before == {
            str(p.relative_to(destination)): p.read_bytes()
            for p in destination.rglob("*")
            if p.is_file()
        }
    download.assert_called_once()
    assert install.call_count == 2
    assert install.call_args.args[0] == [
        "uv",
        "sync",
        "--project",
        ROOT / "environments/plotting",
        "--python",
        "3.13",
        "--locked",
    ]
    assert check.call_count == 2
    assert provenance.source_state(destination)["revision"] == PIN
    assert install.call_args.kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(
        destination / ".venv"
    )
    assert (destination / "uv.lock").read_bytes() == b"plotting repository lock"


def test_supplied_custom_checkout_is_used_without_download_or_changes(tmp_path):
    (tmp_path / "SOURCE.json").write_text(
        json.dumps(
            dict(
                revision="other",
                files={"my-work.py": hashlib.sha256(b"local work").hexdigest()},
            )
        )
    )
    (tmp_path / "my-work.py").write_text("local work")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with (
        patch.object(prepare.urllib.request, "urlopen") as download,
        patch.object(prepare, "call") as install,
        patch("vldb_experiments.report.plotting_python"),
    ):
        prepare.plotting(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    download.assert_not_called()
    install.assert_called_once()


def test_private_source_uses_existing_github_cli_login(tmp_path):
    source = {
        "repository": "datologyai/zephon-vldb-plots",
        "revision": PIN,
        "directory": "plots",
    }
    denied = urllib.error.HTTPError("https://api.github.com", 404, "private", {}, None)
    with (
        patch.object(prepare.urllib.request, "urlopen", side_effect=denied),
        patch.object(
            prepare.subprocess, "check_output", return_value=source_archive()
        ) as fetch,
    ):
        prepare.export_source(source, tmp_path)
    assert fetch.call_args.args[0] == [
        "gh",
        "api",
        f"repos/datologyai/zephon-vldb-plots/tarball/{PIN}",
    ]
    assert provenance.source_state(tmp_path / "plots")["revision"] == PIN


def test_private_source_auth_failure_is_actionable_and_leaves_no_snapshot(tmp_path):
    source = {
        "repository": "datologyai/zephon-vldb-plots",
        "revision": PIN,
        "directory": "plots",
    }
    denied = urllib.error.HTTPError("https://api.github.com", 404, "private", {}, None)
    with (
        patch.object(prepare.urllib.request, "urlopen", side_effect=denied),
        patch.object(prepare.subprocess, "check_output", side_effect=FileNotFoundError),
        pytest.raises(RuntimeError, match="gh auth login"),
    ):
        prepare.export_source(source, tmp_path)
    assert not (tmp_path / "plots").exists()


def test_custom_controller_path_is_separate_from_target_paths(tmp_path):
    raw = dict(
        controller=dict(plots=str(tmp_path / "laptop plots")),
        paths=dict(plots="/target/plots"),
    )
    assert prepare.controller_plots(raw) == tmp_path / "laptop plots"
    assert prepare.controller_plots({}) == ROOT.parent / "zephon-vldb-plots"


def test_setup_plan_includes_controller_without_fetching_or_connecting(
    tmp_path, capsys
):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            dict(
                machine=dict(kind="ssh", host="host", root="/target"),
                controller=dict(plots=str(tmp_path / "plots")),
            )
        )
    )
    with (
        patch.object(prepare, "plotting") as setup,
        patch("vldb_experiments.remote.execute") as execute,
    ):
        assert cli.main(["machine", "setup", "--profile", str(path), "--plan"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["controller"]["plots"] == str(tmp_path / "plots")
    assert preview["paths"]["plots"] == "/target/repositories/plots"
    setup.assert_not_called()
    execute.assert_not_called()


def test_remote_summary_uses_recorded_controller_path(tmp_path):
    core.write_json(tmp_path / "remote.json", dict(plots="/laptop/custom-plots"))
    with (
        patch.object(summary, "kind", return_value="throughput"),
        patch(
            "vldb_experiments.report.summary", return_value=dict(status="Complete")
        ) as analyze,
    ):
        summary.summarize(tmp_path, format="json")
        assert analyze.call_args.args[1] == Path("/laptop/custom-plots")
        summary.summarize(tmp_path, format="json", plots=Path("/explicit/plots"))
        assert analyze.call_args.args[1] == Path("/explicit/plots")


def test_remote_plot_uses_recorded_controller_path(tmp_path):
    core.write_json(tmp_path / "remote.json", dict(plots="/laptop/custom-plots"))
    (tmp_path / "report/figures").mkdir(parents=True)
    with (
        patch.object(summary, "kind", return_value="throughput"),
        patch("vldb_experiments.report.analyze") as analyze,
    ):
        summary.plot(tmp_path)
    analyze.assert_called_once_with(tmp_path, Path("/laptop/custom-plots"), render=True)


def test_existing_dependency_lock_is_never_overwritten(tmp_path):
    (tmp_path / "SOURCE.json").write_text(
        json.dumps(
            dict(
                revision=PIN,
                files={"uv.lock": hashlib.sha256(b"user's lock").hexdigest()},
            )
        )
    )
    (tmp_path / "uv.lock").write_text("user's lock")
    with (
        patch.object(prepare, "call") as install,
        patch("vldb_experiments.report.plotting_python"),
    ):
        prepare.plotting(tmp_path)
    assert (tmp_path / "uv.lock").read_text() == "user's lock"
    install.assert_called_once()
