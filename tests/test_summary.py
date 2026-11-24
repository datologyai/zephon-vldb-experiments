"""Terminal reporting reuses saved evidence without changing it or launching jobs."""

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from test_checkpoint_report import completed_case
from test_training import prepare_report_fixture
from test_ttfb import artifacts

from vldb_experiments import cli, core, presentation, summary, ttfb
from vldb_experiments.training import cli as training_cli
from vldb_experiments.training import protocol, trace


def snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_checkpoint_summary_is_read_only_and_paper_order_precedes_rss(tmp_path):
    completed_case(tmp_path, "save")
    before = snapshot(tmp_path)
    output = io.StringIO()
    with redirect_stdout(output):
        cli.main(["summarize", str(tmp_path)])
    assert snapshot(tmp_path) == before
    text = output.getvalue()
    assert (
        text.index("In-flight state")
        < text.index("Save cost")
        < text.index("Restore cost")
        < text.index("Lane scaling")
    )
    assert (
        text.index("Paper metrics")
        < text.index("Additional measurements")
        < text.index("RSS")
    )
    assert "Partial/diagnostic" in text
    assert not (tmp_path / "report").exists()
    output = io.StringIO()
    with redirect_stdout(output):
        cli.main(["summarize", str(tmp_path), "--format", "json"])
    assert json.loads(output.getvalue())["paper"]
    assert snapshot(tmp_path) == before


def test_training_summary_and_plot_inputs_need_only_raw_evidence(tmp_path):
    from vldb_experiments.training.plotting import paper_data

    prepare_report_fixture(tmp_path, protocol.load(small=True), "raw-only")
    before = snapshot(tmp_path)
    result = trace.summary(tmp_path)
    rows, checkpoint = paper_data(tmp_path)
    assert checkpoint == 150
    assert len(rows) == 1800
    assert result["paper"][0]["rows"][0][0] == "Standard loader"
    assert result["paper"][0]["rows"][-1][0] == "Zephon"
    assert snapshot(tmp_path) == before
    assert not (tmp_path / "report").exists()


def test_ttfb_summary_preserves_raw_trials_and_missing_baselines(tmp_path):
    selected = ttfb.plan(dops=[64])
    for name, cases in ttfb.jobs(selected).items():
        attempt = tmp_path / "jobs" / name
        artifacts(attempt, cases)
        ttfb.collect(cases, attempt, tmp_path)
    data = dict(samples=100, files=[])
    core.write_json(
        tmp_path / "campaign.json",
        dict(
            cases=selected,
            datasets=dict(text=data, vlm=data),
            profile={"ttfb_data": {"source": "public"}},
        ),
    )
    before = snapshot(tmp_path)
    output = io.StringIO()
    with redirect_stdout(output):
        result = summary.summarize(tmp_path)
    assert result["paper"][0]["rows"][0][0] == "Zephon threads (GIL)"
    assert result["paper"][0]["rows"][4][1:] == ["—", "—"]
    assert "Partial/diagnostic" in result["status"]
    assert snapshot(tmp_path) == before
    assert not (tmp_path / "report").exists()


def test_terminal_tables_wrap_and_handle_no_measurements():
    report = presentation.document(
        "Example",
        [],
        [presentation.table("Empty", ["Metric", "Value"], [])],
        [
            presentation.table(
                "Long", ["Metric", "Value"], [["A long measurement name " * 5, "1.25"]]
            )
        ],
    )
    text = presentation.render(report, width=70)
    assert max(map(len, text.splitlines())) <= 70
    assert "1.25" in text
    assert text.index("Paper metrics") < text.index("Additional measurements")


def test_training_run_waits_collects_and_finishes_without_extra_flag(tmp_path):
    with (
        patch.object(training_cli, "run_campaign") as launch,
        patch.object(
            training_cli,
            "wait_and_collect",
            return_value={"exact": True, "expected_outcomes_met": True},
        ) as finish,
        patch("vldb_experiments.report.plotting_python") as plotting_check,
        patch(
            "vldb_experiments.prepare.plotting",
            side_effect=AssertionError("implicit setup"),
        ),
    ):
        assert (
            cli.main(
                [
                    "run",
                    "training",
                    "--prefix",
                    "test-run",
                    "--output",
                    str(tmp_path),
                    "--plots",
                    "/plots",
                ]
            )
            == 0
        )
    launch.assert_called_once()
    plotting_check.assert_called_once_with(Path("/plots"))
    finish.assert_called_once_with(tmp_path, Path("/plots"))


def test_training_detach_does_not_wait_or_collect(tmp_path):
    with (
        patch.object(training_cli, "run_campaign"),
        patch.object(training_cli, "wait_and_collect") as wait,
    ):
        cli.main(
            [
                "run",
                "training",
                "--prefix",
                "test-run",
                "--output",
                str(tmp_path),
                "--detach",
            ]
        )
    wait.assert_not_called()


def test_waiter_attaches_to_existing_owner_then_collects_and_finishes(tmp_path):
    with (
        patch.object(
            training_cli,
            "status",
            side_effect=[{"state": "running"}, {"state": "complete"}],
        ),
        patch.object(training_cli.time, "sleep") as sleep,
        patch.object(training_cli, "collect", return_value={"exact": True}) as collect,
        patch.object(summary, "finish") as finish,
    ):
        assert training_cli.wait_and_collect(tmp_path, Path("/plots")) == {
            "exact": True
        }
    sleep.assert_called_once_with(30)
    collect.assert_called_once_with(tmp_path)
    finish.assert_called_once_with(tmp_path, plots=Path("/plots"))


def test_failed_owner_does_not_collect_or_restart(tmp_path):
    import pytest

    with (
        patch.object(
            training_cli,
            "status",
            return_value={"state": "failed", "error": "trainer failed"},
        ),
        patch.object(training_cli, "collect") as collect,
        pytest.raises(RuntimeError, match="preserved"),
    ):
        training_cli.wait_and_collect(tmp_path)
    collect.assert_not_called()


def test_completed_run_finishes_with_summary_then_paper_plots(tmp_path):
    calls = []
    with (
        patch.object(
            summary,
            "summarize",
            side_effect=lambda *a, **k: (
                calls.append("summary") or {"status": "Complete"}
            ),
        ),
        patch.object(
            summary, "plot", side_effect=lambda *a, **k: calls.append("plot") or []
        ),
    ):
        summary.finish(tmp_path)
    assert calls == ["summary", "plot"]


def test_text_only_results_do_not_claim_the_paper_contains_tables(tmp_path, capsys):
    with patch.object(summary, "plot", return_value=[]):
        assert cli.main(["plot", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "No figure to generate" in text
    assert "tables" not in text


def test_uploaded_training_sources_import_without_local_presentation_modules(tmp_path):
    import subprocess
    import sys
    import tarfile

    from vldb_experiments.training import ssh

    manifest = {"code": protocol.code_identity(), "config": protocol.load()}
    raw = ssh.source_archive(manifest)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        archive.extractall(tmp_path, filter="data")
    assert not (tmp_path / "src/vldb_experiments/presentation.py").exists()
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys;sys.path.insert(0,sys.argv[1]);import vldb_experiments.training.trace, vldb_experiments.training.ssh_runner",
            str(tmp_path / "src"),
        ],
        check=True,
        capture_output=True,
    )
