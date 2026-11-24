"""The training command adapts its own evidence to the existing paper renderer."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_training import prepare_report_fixture

from vldb_experiments.training import protocol, trace
from vldb_experiments.training.plotting import paper_data, render


def full_report(root, config=None):
    config = config or protocol.load()
    prepare_report_fixture(root, config, "fixture")
    trace.report(root)


class TrainingPlottingTests(unittest.TestCase):
    def test_small_scale_reports_measured_widths_without_claiming_full_figure1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = protocol.load(small=True)
            prepare_report_fixture(root, config, "small-identity")
            summary = trace.report(root)
            self.assertTrue(summary["expected_outcomes_met"])
            self.assertFalse(summary["paper_protocol_match"])
            self.assertEqual(summary["experiment_scale"], "single-node")
            rows, checkpoint = paper_data(root)
            self.assertEqual(checkpoint, 150)
            self.assertEqual({r["reference_dp"] for r in rows}, {2})
            self.assertEqual({r["resume_dp"] for r in rows}, {1, 4, 8})
            self.assertEqual(sum(r["origin"] == "resume" for r in rows), 900)

    def test_complementary_campaign_plot_uses_both_original_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for arm in ("zephon", "control"):
                run = root / arm
                run.mkdir()
                config = protocol.load()
                config["arms"] = [arm]
                prepare_report_fixture(run, config, arm + "-identity")
                runs.append(run)
            output = root / "combined"
            summary = trace.combine(runs, output)
            self.assertTrue(summary["paper_protocol_match"])
            rows, checkpoint = paper_data(output)
            self.assertEqual(checkpoint, 150)
            self.assertEqual({r["arm"] for r in rows}, {"elastic", "control"})
            self.assertEqual(
                {r["campaign_identity"] for r in rows},
                {"zephon-identity", "control-identity"},
            )
            self.assertEqual(len(rows), 1800)

    def test_only_measured_resumes_get_comparison_rows_and_incomplete_data_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_report(root)
            rows, checkpoint = paper_data(root)
            self.assertEqual(checkpoint, 150)
            self.assertEqual({r["arm"] for r in rows}, {"control", "elastic"})
            self.assertEqual({r["resume_dp"] for r in rows}, {8, 32, 64})
            self.assertEqual(sum(r["origin"] == "resume" for r in rows), 900)
            self.assertTrue(
                all(r["full_hash_match"] for r in rows if r["arm"] == "elastic")
            )
            self.assertTrue(
                all(
                    not r["full_hash_match"]
                    for r in rows
                    if r["arm"] == "control" and r["is_post_fault"]
                )
            )
            self.assertTrue(
                all(r["origin"] == "shared_reference" for r in rows if r["step"] <= 150)
            )
            path = root / "report/summary.json"
            path.parent.mkdir()
            path.write_text("untrusted derived data")
            # Figures validate the original rank traces.
            self.assertEqual(paper_data(root)[0], rows)
            raw = root / "control-resume-dp64/diagnostics/rank-0.jsonl"
            raw.write_text("\n".join(raw.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaises(ValueError):
                paper_data(root)

    @unittest.skipUnless(
        importlib.util.find_spec("plotting"), "Needs the plotting checkout environment"
    )
    def test_render_calls_existing_paper_function_and_records_its_source(self):
        from plotting.system import plot_fig_1_fault_resume as paper

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_report(root)
            with patch.object(paper, "_plot_arm", wraps=paper._plot_arm) as existing:
                paths = render(root)
            self.assertEqual(existing.call_count, 2)
            self.assertEqual(
                [call.args[2] for call in existing.call_args_list],
                ["control", "elastic"],
            )
            self.assertEqual({p.suffix for p in paths}, {".png", ".pdf", ".svg"})
            self.assertTrue(all(p.stat().st_size > 1000 for p in paths))
            svg = (root / "report/figures/figure-1-control.svg").read_text()
            self.assertIn(r"\z fault", svg)
            self.assertNotIn("checkpoint</text>", svg)
            provenance = json.loads(
                (root / "report/figures/provenance.json").read_text()
            )
            self.assertEqual(
                provenance["renderer"],
                "plotting.system.plot_fig_1_fault_resume.plot_checkpoint_resume",
            )
            self.assertIn(
                "plotting/system/plot_fig_1_fault_resume.py", provenance["code_sha256"]
            )
            self.assertEqual(provenance["variant"], "small")
            self.assertEqual(provenance["arms"], ["control", "elastic"])
            self.assertEqual(len(paths), 6)

    @unittest.skipUnless(
        importlib.util.find_spec("plotting"), "Needs the plotting checkout environment"
    )
    def test_small_renderer_uses_measured_widths_and_restores_paper_defaults(self):
        from plotting.system import plot_fig_1_fault_resume as paper

        original = paper._plot_arm
        widths = []
        defaults = (
            paper.REFERENCE_WORLD_SIZE,
            paper.RESUME_WIDTHS,
        )

        def measured(*args, **kwargs):
            widths.append(
                (
                    kwargs["geometry"].reference,
                    list(kwargs["geometry"].resumes),
                    kwargs["geometry"].resumes[0],
                )
            )
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_report(root, protocol.load(small=True))
            with patch.object(paper, "_plot_arm", side_effect=measured):
                paths = render(root)
            self.assertEqual(widths, [(2, [1, 4, 8], 1)] * 2)
            self.assertEqual(
                (paper.REFERENCE_WORLD_SIZE, paper.RESUME_WIDTHS),
                defaults,
            )
            self.assertTrue(all(p.stat().st_size > 1000 for p in paths))
