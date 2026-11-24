import pytest

from vldb_experiments import checkpoint, checkpoint_report, checkpoint_summary, core
from vldb_experiments.presentation import render


def completed_case(output, mode):
    case = next(c for c in checkpoint.plan() if c["mode"] == mode)
    ranks = []
    for rank in range(8):
        ranks.append(
            dict(
                rank=rank,
                memory=dict(
                    window={
                        "fresh": "iterator startup through first fresh batch",
                        "save": "checkpoint capture through serialization, final write and synchronization",
                        "restore": "checkpoint read through first restored batch",
                    }[mode],
                    poll_interval_ms=10,
                    rss_baseline_bytes=(12 if rank == 0 else 2) * 2**30,
                    rss_peak_bytes=(20 if rank == 0 else 26) * 2**30,
                    rss_incremental_bytes=(8 if rank == 0 else 24) * 2**30,
                ),
                continuation={"matches": True},
                checkpoint_bytes=2**20,
                checkpoint_source_state=dict(
                    total=dict(chunks=36, source_records=36864),
                    per_dp_rank={
                        str(r): dict(chunks=r + 1, source_records=(r + 1) * 1024)
                        for r in range(8)
                    },
                    per_lane={
                        str(r): dict(chunks=r + 1, source_records=(r + 1) * 1024)
                        for r in range(8)
                    },
                ),
                batches_before_save_per_rank=400,
                continuation_hashes=["same"] * 4,
            )
        )
    latency = dict(
        fresh="fresh_to_first_batch",
        save="save_end_to_end",
        restore="restore_to_first_batch",
    )[mode]
    result = dict(mode=mode, world_size=8, ranks=ranks, max_across_ranks_s={latency: 1})
    directory = output / "cases" / case["id"]
    core.write_json(directory / "status.json", {"state": "complete"})
    core.write_json(directory / "result.json", result)
    core.write_json(output / "campaign.json", {"cases": [case]})
    return directory, result


def test_state_report_distinguishes_rank_mean_and_maximum(tmp_path):
    completed_case(tmp_path, "save")
    (row,) = checkpoint_report.report(tmp_path)
    assert row["inflight_source_records_mean_per_rank"] == 4608
    assert row["inflight_source_records_max_per_rank"] == 8192
    values = {r["metric"]: r for r in checkpoint_report.prose_values([row])}
    assert values["inflight_source_records_mean_per_rank"]["median"] == 4608
    assert values["inflight_source_records_max_per_rank"]["median"] == 8192


@pytest.mark.parametrize("mode", ["fresh", "save", "restore"])
def test_rss_report_retains_rank_windows_and_independent_maxima(tmp_path, mode):
    completed_case(tmp_path, mode)
    (row,) = checkpoint_report.report(tmp_path)
    # The largest baseline is rank0; the largest increase belongs to another
    # rank. Subtracting independent baseline/peak maxima gives the wrong answer.
    assert row["rss_baseline_max_per_rank_gib"] == 12
    assert row["rss_peak_max_per_rank_gib"] == 26
    assert row["rss_incremental_max_per_rank_gib"] == 24
    assert not (tmp_path / "report").exists()
    values = checkpoint_report.prose_values([row])
    rss = next(r for r in values if r["metric"] == "rss_incremental_max_per_rank_gib")
    assert rss["unit"] == "GiB" and rss["median"] == 24
    assert not any("completion" in r["metric"] for r in values)


def test_mixed_rank_rss_windows_are_not_aggregated(tmp_path):
    directory, result = completed_case(tmp_path, "restore")
    result["ranks"][3]["memory"]["window"] = "through full replay completion"
    core.write_json(directory / "result.json", result)
    with pytest.raises(ValueError, match="different RSS measurement windows"):
        checkpoint_report.report(tmp_path)


def test_paper_and_additional_tables_keep_memory_out_of_paper_metrics(tmp_path):
    completed_case(tmp_path, "save")
    values = checkpoint_report.prose_values(checkpoint_report.report(tmp_path))
    paper = [r for r in values if r["category"] == "paper"]
    additional = [r for r in values if r["category"] == "additional"]
    assert "inflight_source_records_mean_per_rank" in {r["metric"] for r in paper}
    assert not any(r["metric"].startswith("rss_") for r in paper)
    assert "rss_incremental_max_per_rank_gib" in {r["metric"] for r in additional}
    assert not (tmp_path / "report").exists()
    text = render(checkpoint_summary.summary(tmp_path))
    assert text.index("Paper metrics") < text.index("Additional measurements")
    assert "Process-tree RSS" not in text.split("Additional measurements")[0]


def test_partial_summary_does_not_fill_pending_cells_with_reference_values(tmp_path):
    completed_case(tmp_path, "fresh")
    selected = [
        c for c in checkpoint.plan() if c["point"] == checkpoint.plan()[0]["point"]
    ]
    core.write_json(tmp_path / "campaign.json", {"cases": selected})
    result = checkpoint_summary.summary(tmp_path)
    text = render(result)
    assert "Partial/diagnostic campaign: 1/3 jobs" in text
    progress = next(
        table
        for table in result["paper"]
        if table["title"] == "Checkpoint size and save cost versus progress"
    )
    assert progress["rows"][-1] == [10000, "—", "—"]
    assert not (tmp_path / "report").exists()
    assert "Packing waste" not in text
    assert "48.5" not in text
