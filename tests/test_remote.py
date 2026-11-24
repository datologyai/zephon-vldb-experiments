"""Exercise real detached owners and transfers without a cloud VM or benchmark."""

import io
import json
import sys
import tarfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from vldb_experiments import cli, core, remote, remote_worker, summary

SLEEP = time.sleep


def test_connection_rejects_options_as_hostnames():
    with pytest.raises(ValueError, match="SSH host"):
        remote.connection(
            Path("profile.toml"),
            dict(machine=dict(kind="ssh", host="-oProxyCommand=bad")),
        )


def test_ssh_shell_arguments_are_quoted_once(tmp_path):
    value = "spaces 'quoted' $HOME `false` ; $(false)"
    result = remote.call(
        ["sh", "-c", 'eval "$1"', "unused"],
        [sys.executable, "-c", "import sys; print(sys.argv[1])", value],
    )
    assert result.stdout.decode().strip() == value


def test_source_upload_omits_profiles_data_and_build_metadata():
    identity, data = remote.source_archive()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        names = archive.getnames()
        assert "SOURCE.json" in names
        assert "configs/checkpoint/experiment.json" in names
        assert all(
            not name.startswith(("profiles/", "runs/", "data/")) for name in names
        )
        assert all(
            ".egg-info" not in name and "__pycache__" not in name for name in names
        )
        assert "snapshot.json" not in names
        assert (
            core.digest(json.load(archive.extractfile("SOURCE.json"))["files"])
            == identity
        )


def test_launcher_upload_identity_ignores_finder_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    for name in ("src/vldb_experiments", "configs", "environments"):
        directory = tmp_path / name
        directory.mkdir(parents=True)
        (directory / "source.txt").write_text("source")
    before = remote.source_archive()[0]
    (tmp_path / "configs/.DS_Store").write_bytes(b"Finder metadata")
    assert remote.source_archive()[0] == before


def fake_target(tmp_path, monkeypatch):
    # This SSH stand-in executes the quoted remote command on this test host.
    ssh = tmp_path / "ssh-standin"
    ssh.write_text(
        "#!"
        + sys.executable
        + "\nimport os,sys\nos.execv('/bin/sh',['sh','-c',sys.argv[-1]])\n"
    )
    ssh.chmod(0o755)
    monkeypatch.setattr(remote, "connection", lambda *_: [str(ssh)])
    monkeypatch.setattr(remote.time, "sleep", lambda _: SLEEP(0.01))
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            dict(
                machine=dict(
                    kind="ssh",
                    host="test",
                    root=str(tmp_path / "target with spaces"),
                    python=sys.executable,
                )
            )
        )
    )
    return path


def terminal_state(run):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = remote.status(run)
        if state["state"] in ("complete", "failed", "orphaned"):
            return state
        time.sleep(0.03)
    pytest.fail("Detached fixture did not terminate")


def test_detach_reconnect_collect_and_conflict_protection(tmp_path, monkeypatch):
    path = fake_target(tmp_path, monkeypatch)
    counter = tmp_path / "invocations"

    def command(raw, source, argv):
        output = argv[argv.index("--output") + 1]
        code = "from pathlib import Path; import sys,json; c=Path(sys.argv[1]); c.write_text(str(int(c.read_text())+1) if c.exists() else '1'); p=Path(sys.argv[2]); p.mkdir(parents=True,exist_ok=True); (p/'result.json').write_text(json.dumps({'observed':42}))"
        return [sys.executable, "-c", code, str(counter), output]

    monkeypatch.setattr(remote, "command", command)
    run = tmp_path / "controller results"
    argv = [
        "run",
        "throughput",
        "--profile",
        str(path),
        "--output",
        str(run),
        "--detach",
    ]
    assert cli.main(argv) == 0
    assert terminal_state(run)["state"] == "complete"
    assert cli.main(argv) == 0
    assert counter.read_text() == "1"
    remote.collect(run, finish=False)
    assert core.read_json(run / "result.json") == {"observed": 42}
    (run / "result.json").write_text("changed locally")
    with pytest.raises(ValueError, match="conflicts"):
        remote.collect(run, finish=False)
    assert (run / "result.json").read_text() == "changed locally"
    receipt = core.read_json(run / "remote.json")
    frozen = Path(receipt["source"]) / "configs/checkpoint/experiment.json"
    frozen.write_text("tampered")
    with pytest.raises(Exception):
        remote.deploy(path, cli.profiles.read(path))


def test_failed_owner_requires_resume_and_keeps_failed_attempt(tmp_path, monkeypatch):
    path = fake_target(tmp_path, monkeypatch)
    counter = tmp_path / "attempts"

    def command(raw, source, argv):
        output = argv[argv.index("--output") + 1]
        code = "from pathlib import Path; import sys; c=Path(sys.argv[1]); n=int(c.read_text())+1 if c.exists() else 1; c.write_text(str(n)); p=Path(sys.argv[2]); p.mkdir(parents=True,exist_ok=True); (p/'result.json').write_text('{}'); sys.exit(1 if n==1 else 0)"
        return [sys.executable, "-c", code, str(counter), output]

    monkeypatch.setattr(remote, "command", command)
    run = tmp_path / "run"
    argv = ["run", "ttfb", "--profile", str(path), "--output", str(run), "--detach"]
    cli.main(argv)
    assert terminal_state(run)["state"] == "failed"
    cli.main(argv)
    assert counter.read_text() == "1"
    with patch.object(summary, "finish"):
        remote.resume(run)
    assert counter.read_text() == "2"
    receipt = core.read_json(run / "remote.json")
    assert (
        core.read_json(Path(receipt["job"]) / "attempt-1-state.json")["returncode"] == 1
    )
    assert remote.status(run)["state"] == "complete"


def test_partial_summary_does_not_attempt_complete_paper_figures(tmp_path):
    with (
        patch.object(
            summary, "summarize", return_value={"status": "Partial selection"}
        ),
        patch.object(summary, "plot") as plot,
    ):
        assert summary.finish(tmp_path) == []
    plot.assert_not_called()


def test_dead_owner_is_not_relaunched_automatically(tmp_path):
    core.write_json(
        tmp_path / "state.json",
        dict(state="running", pid=987654321, process_start="absent"),
    )
    assert remote_worker.status(tmp_path)["state"] == "orphaned"


def test_machine_setup_prepares_controller_before_remote_owner(tmp_path, monkeypatch):
    path = fake_target(tmp_path, monkeypatch)
    raw = cli.profiles.read(path)
    raw["controller"] = dict(plots=str(tmp_path / "controller-plots"))
    path.write_text(json.dumps(raw))
    calls = []

    def setup(plots, *, existing=False):
        assert existing
        plots.mkdir()
        (plots / "renderer.py").write_text("# local renderer\n")
        calls.append(("local", plots))

    def command(raw, source, argv):
        calls.append(("remote", argv[:2]))
        return [sys.executable, "-c", "print('Target fixture setup completed')"]

    monkeypatch.setattr(remote.prepare, "plotting", setup)
    monkeypatch.setattr(remote, "command", command)
    assert cli.main(["machine", "setup", "--profile", str(path)]) == 0
    assert calls == [
        ("local", tmp_path / "controller-plots"),
        ("remote", ["machine", "setup"]),
    ]
    receipt = next((path.parent / ".jobs").glob("*/remote.json"))
    assert core.read_json(receipt)["plots"] == str(tmp_path / "controller-plots")


def test_plot_setup_failure_never_starts_remote_setup(tmp_path, monkeypatch):
    path = fake_target(tmp_path, monkeypatch)
    with (
        patch.object(
            remote.prepare,
            "plotting",
            side_effect=RuntimeError("plotting setup failed"),
        ),
        patch.object(remote, "deploy") as deploy,
    ):
        with pytest.raises(RuntimeError, match="plotting setup failed"):
            cli.main(["machine", "setup", "--profile", str(path)])
    deploy.assert_not_called()
    assert not (path.parent / ".jobs").exists()
