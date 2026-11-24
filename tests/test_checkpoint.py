import json

import pytest
from vldb_experiments import checkpoint, checkpoint_report


@pytest.mark.parametrize("field", ["dataset", "artifacts", "aggregation"])
@pytest.mark.parametrize(
    "location",
    [
        "/nvme/data",
        "./data",
        "file:///nvme/data",
        "https://bucket/data",
        "s3://",
        "s3:///nvme/data",
        "s3://bucket/data?query=1",
    ],
)
def test_paper_launcher_rejects_local_or_malformed_storage_before_starting(
    tmp_path, field, location
):
    profile = dict(
        dataset="s3://bucket/data",
        artifacts="s3://bucket/checkpoints",
        aggregation="s3://bucket/aggregation",
    )
    profile[field] = location
    output = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match=f"checkpoint.{field} must be an s3://"):
        checkpoint.run(profile, checkpoint.plan()[:1], output)
    assert not output.exists()


def test_paper_matrix_deduplicates_shared_points():
    cases = checkpoint.plan()
    assert len(cases) == 435
    assert len({c["id"] for c in cases}) == 435
    for rep in [1, 2, 3]:
        shared = [
            c
            for c in cases
            if c["repetition"] == rep
            and c["k"] == 8
            and c["chunk_size"] == 1024
            and c["seed"] == 42
            and c["lanes"] == 8
            and c["optimizer_steps"] == 100
        ]
        assert {c["mode"] for c in shared} == {"fresh", "save", "restore"}
        assert all({"main", "lanes"} <= set(c["analysis_tags"]) for c in shared)
        assert all(
            "progress" in c["analysis_tags"] for c in shared if c["mode"] != "fresh"
        )
    assert len(checkpoint.plan("main")) == 405
    assert len(checkpoint.plan("progress")) == 18
    assert len(checkpoint.plan("lanes")) == 27


def test_restore_rejects_wrong_continuation():
    case = dict(checkpoint.plan()[0], mode="restore")
    result = dict(
        mode="restore",
        world_size=8,
        ranks=[dict(rank=r, continuation={"matches": True}) for r in range(8)],
    )
    checkpoint.validate_result(case, result)
    result["ranks"][4]["continuation"]["matches"] = False
    with pytest.raises(ValueError, match="continuation"):
        checkpoint.validate_result(case, result)


def test_command_preserves_progress_and_s3_protocol(tmp_path):
    case = next(c for c in checkpoint.plan() if c["optimizer_steps"] == 10000)
    profile = dict(
        python="/env/bin/python",
        cache="/nvme/cache",
        dataset="s3://bucket/data",
        artifacts="s3://bucket/artifacts",
        aggregation="s3://bucket/aggregation",
        tokenizer_revision="a" * 40,
        image_processor_revision="b" * 40,
    )
    argv = checkpoint.command(case, profile, tmp_path, "study")
    assert argv[argv.index("--optimizer-steps") + 1] == "10000"
    assert argv[argv.index("--cache-policy") + 1] == "cold"
    assert argv[argv.index("--nproc-per-node") + 1] == "8"
    data = json.loads((tmp_path / "loaders.json").read_text())["loaders"][0]["config"]
    assert data["vlm_zephon"]["shuffle_pack_mode"] == "per_lane"
    assert data["vlm_zephon"]["batch_metadata"] == "preserve_contributors"
    assert data["pipeline"]["canonical_replicas"] == case["lanes"]
    assert data["vlm"]["tokenizer_revision"] == "a" * 40


def test_cache_cleanup_rejects_other_invocations(tmp_path):
    profile = {"cache": str(tmp_path)}
    own = tmp_path / ".arena-checkpoint-cache" / "own"
    (own / "rank-0").mkdir(parents=True)
    other = tmp_path / ".arena-checkpoint-cache" / "other"
    (other / "rank-0").mkdir(parents=True)
    result = {"ranks": [{"rank": 0, "cache": {"path": str(other / "rank-0")}}]}
    with pytest.raises(ValueError, match="outside"):
        checkpoint.cleanup_cache(profile, ["--cache-namespace", "own"], result)
    assert (other / "rank-0").exists()
    result["ranks"][0]["cache"]["path"] = str(own / "rank-0")
    checkpoint.cleanup_cache(profile, ["--cache-namespace", "own"], result)
    assert not own.exists()
    assert other.exists()


def test_prose_distinguishes_ratio_of_medians_from_paired_ratio():
    rows = []
    for point, fresh, restore in [("a", 1, 2), ("b", 2, 8), ("c", 10, 15)]:
        common = dict(
            point=point,
            analysis_tags="main",
            optimizer_steps=100,
            lanes=8,
            chunk_size=1024,
        )
        rows.extend(
            [
                common | dict(mode="fresh", fresh_to_first_batch=fresh),
                common | dict(mode="restore", restore_to_first_batch=restore),
            ]
        )
    values = {x["metric"]: x for x in checkpoint_report.prose_values(rows)}
    assert values["restore_to_fresh_paired_ratio"]["median"] == 2
    assert values["restore_to_fresh_ratio_of_medians"]["median"] == 4
    assert values["restore_to_fresh_paired_ratio"]["n"] == 3


def test_machine_profile_preserves_virtual_environment_interpreter_symlink(tmp_path):
    import sys

    python = tmp_path / "environments" / "cpu-3.13" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    values = dict(
        arena=str(tmp_path / "arena"),
        python=str(python),
        cache=str(tmp_path / "cache"),
        measurement_lock=str(tmp_path / "lock"),
        dataset="s3://bucket/data",
        artifacts="s3://bucket/artifacts",
        aggregation="s3://bucket/aggregation",
        tokenizer_revision="a" * 40,
        image_processor_revision="b" * 40,
    )
    profile = tmp_path / "profile.toml"
    profile.write_text(
        "[paths]\narena="
        + json.dumps(values["arena"])
        + "\nenvironments="
        + json.dumps(str(tmp_path / "environments"))
        + "\n[checkpoint]\n"
        + "".join(f"{k} = {json.dumps(v)}\n" for k, v in values.items())
    )
    loaded = checkpoint.load_profile(profile)
    assert loaded["python"] == str(python)
    assert loaded["python"] != str(python.resolve())


def test_torchrun_does_not_consume_benchmark_local_option(tmp_path):
    from torch.distributed.run import parse_args

    profile = dict(
        python="/env/bin/python",
        cache="/nvme/cache",
        dataset="s3://bucket/data",
        artifacts="s3://bucket/artifacts",
        aggregation="s3://bucket/aggregation",
        tokenizer_revision="a" * 40,
        image_processor_revision="b" * 40,
    )
    command = checkpoint.command(checkpoint.plan()[0], profile, tmp_path, "study")
    parsed = parse_args(command[3:])
    assert parsed.training_script == "dataloader_arena.checkpoint"
    assert "--local" in parsed.training_script_args
    assert (
        parsed.training_script_args[
            parsed.training_script_args.index("--cache-scope") + 1
        ]
        == "host"
    )


def test_successful_host_shared_cache_cleanup_preserves_other_namespaces(tmp_path):
    own = tmp_path / ".arena-checkpoint-cache" / "own" / "host"
    own.mkdir(parents=True)
    other = tmp_path / ".arena-checkpoint-cache" / "other" / "host"
    other.mkdir(parents=True)
    result = {
        "ranks": [
            {"rank": rank, "cache": {"path": str(own), "scope": "host"}}
            for rank in range(8)
        ]
    }
    checkpoint.cleanup_cache(
        {"cache": str(tmp_path)},
        ["--cache-namespace", "own", "--cache-scope", "host"],
        result,
    )
    assert not own.exists()
    assert other.exists()


def test_completed_benchmark_waits_for_orphaned_download_workers(tmp_path):
    import os
    import sys

    from vldb_experiments import core

    # The benchmark exits, leaving a worker that ignores TERM and keeps writing.
    script = """
import os, signal, time
from pathlib import Path
pid = os.fork()
if pid == 0:
    os.setsid()  # torchrun ranks also start their own sessions.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path('ready').touch()
    while True:
        with Path('writes').open('a') as handle:
            handle.write('x')
        time.sleep(0.01)
while not Path('ready').exists():
    time.sleep(0.01)
time.sleep(0.5)
"""
    assert (
        core.run_process_group(
            [sys.executable, "-c", script],
            str(tmp_path),
            dict(os.environ),
            tmp_path / "run.log",
            10,
        )
        == 0
    )
    # A live orphan would recreate these after the command returned.
    (tmp_path / "ready").unlink()
    (tmp_path / "writes").unlink()
    import time

    time.sleep(0.1)
    assert not (tmp_path / "writes").exists()


def test_cache_cleanup_handles_a_cancelled_download_removing_its_partial_file(
    tmp_path, monkeypatch
):
    import os

    own = tmp_path / ".arena-checkpoint-cache" / "own" / "host"
    own.mkdir(parents=True)
    (own / "chunk.bin.part").touch()
    unlink = os.unlink

    def cancelled_download(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
        raise FileNotFoundError(path)

    monkeypatch.setattr(os, "unlink", cancelled_download)
    checkpoint.cleanup_cache(
        {"cache": str(tmp_path)},
        ["--cache-namespace", "own", "--cache-scope", "host"],
        {"ranks": [{"rank": 0, "cache": {"path": str(own)}}]},
    )
    assert not own.exists()
