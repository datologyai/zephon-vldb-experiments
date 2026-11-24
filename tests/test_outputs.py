"""The paper plotting checkout owns aggregation, allocation selection and rendering."""

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vldb_experiments import core, matrix
from vldb_experiments.plotting import generate
from vldb_experiments.prose import prose_values
from vldb_experiments.report import export_results


class OutputTests(unittest.TestCase):
    def test_export_uses_measured_rows_without_archived_values(self):
        selected = [
            c
            for c in matrix.cases("fetch-and-batch")
            if c["dataset"] == "text"
            and c["system"] == "zephon"
            and c["python"] == "3.13"
            and c["dop"] == 1
            and c["config"]["runner"] == "process"
        ]
        self.assertEqual(len(selected), 6)
        self.assertTrue(all("reference" not in c for c in selected))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core.write_json(root / "campaign.json", {"cases": selected})
            for case in selected:
                folder = root / "cases" / case["id"]
                core.write_json(folder / "status.json", {"state": "complete"})
                core.write_json(
                    folder / "result.json",
                    {
                        "csv_row": {
                            "loader_name": case["loader_name"],
                            "run.seed": case["seed"],
                            "result.samples_per_second": 123,
                        }
                    },
                )
            self.assertEqual(export_results(root), 6)
            files = sorted((root / "report/plot-data").glob("*.csv"))
            self.assertEqual(len(files), 2)
            for file in files:
                with file.open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual({int(r["run.seed"]) for r in rows}, {42, 67, 1337})
                self.assertEqual(
                    {r["result.samples_per_second"] for r in rows}, {"123"}
                )
                self.assertEqual({r["sweep_python"] for r in rows}, {"3.13"})

    def test_figure_and_tables_use_the_same_external_objects(self):
        class Table:
            def __init__(self, rows):
                self.rows = rows

            def write_csv(self, path):
                Path(path).write_text("external table\n")

            def to_dicts(self):
                return self.rows

        modules = {"plotting": types.ModuleType("plotting")}
        for experiment in ("fetch-and-batch", "vlm-pipeline"):
            with (
                self.subTest(experiment=experiment),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                (root / "report").mkdir()
                core.write_json(
                    root / "campaign.json", {"cases": [{"experiment": experiment}]}
                )
                aggregate = Table([{"sentinel": "external seed averages"}])
                selected = Table([{"sentinel": "external allocation choice"}])
                module = types.SimpleNamespace(
                    load_and_process=Mock(return_value=aggregate),
                    select_best_configs=Mock(return_value=selected),
                    plot_fig_6=Mock(),
                    plot_fig_7=Mock(),
                )
                with (
                    patch.dict(sys.modules, modules),
                    patch(
                        "vldb_experiments.plotting.importlib.import_module",
                        return_value=module,
                    ),
                ):
                    output = generate(root, root, render=True)
                module.load_and_process.assert_called_once_with(
                    root / "report/plot-data"
                )
                self.assertEqual(output["points"], aggregate.rows)
                self.assertFalse(hasattr(module, "DATA_DIR"))
                if experiment == "vlm-pipeline":
                    module.select_best_configs.assert_called_once_with(aggregate)
                    module.plot_fig_7.assert_called_once_with(
                        selected, root / "report/figures", filename_prefix="figure-7-"
                    )
                    self.assertEqual(output["selected_allocations"], selected.rows)
                else:
                    module.select_best_configs.assert_not_called()
                    module.plot_fig_6.assert_called_once_with(
                        aggregate, root / "report/figures", filename_prefix="figure-6-"
                    )

    def aggregates(self):
        conditions = [
            ("zephon", "3.13", "processes", "baseline", 100),
            ("zephon", "3.13", "threads", "baseline", 50),
            ("zephon", "3.14t", "processes", "baseline", 120),
            ("zephon", "3.14t", "threads", "baseline", 200),
            ("zephon", "3.13", "processes", "noshm", 145),
            ("grain", "3.13", None, "baseline", 300),
            ("litdata", "3.13", None, "baseline", 200),
        ]
        fetch = {
            "points": [
                {
                    "data_type": dataset,
                    "system": system,
                    "py": python,
                    "runner": runner,
                    "variant": variant,
                    "parallelism": dop,
                    "sps_mean": value * dop / 64,
                }
                for dataset in ("text", "vlm-raw")
                for system, python, runner, variant, value in conditions
                for dop in (1, 64)
            ]
        }
        vlm = {
            "selected_allocations": [
                {
                    "system": system,
                    "dop": dop,
                    "loader_name": f"{system}-selected-{dop}",
                    "utps_mean": rate * dop,
                }
                for system, rate in (
                    ("zephon-fused", 100),
                    ("grain", 120),
                )
                for dop in (8, 16, 32, 64)
            ]
        }
        return fetch, vlm

    def test_prose_shows_only_measured_paper_quantities(self):
        fetch, vlm = self.aggregates()
        rows = prose_values(fetch, vlm)
        self.assertEqual(len(rows), 27)
        self.assertTrue(
            all(
                set(row) == {"metric", "value", "unit", "configuration"} for row in rows
            )
        )
        by_name = {row["metric"]: row for row in rows}
        self.assertAlmostEqual(
            by_name["text: throughput change when disabling SHM"]["value"], 45
        )
        self.assertAlmostEqual(
            by_name["VLM pipeline: Grain lead over Zephon at DOP 64"]["value"], 20
        )
        self.assertFalse(any("InternalDs" in row["metric"] for row in rows))

    def test_summary_preserves_paper_order_and_normalizes_selected_vlm_rates(self):
        from vldb_experiments.report import summary

        fetch, vlm = self.aggregates()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for experiment in ("fetch-and-batch", "vlm-pipeline"):
                cases = matrix.plan(experiment, "v0.1.0")
                core.write_json(root / experiment / "campaign.json", {"cases": cases})
                for case in cases:
                    folder = root / experiment / "cases" / case["id"]
                    core.write_json(folder / "status.json", {"state": "complete"})
                    core.write_json(folder / "result.json", {})
            result = summary(
                root, results={"fetch-and-batch": fetch, "vlm-pipeline": vlm}
            )
        sections = result["paper"]
        self.assertEqual(
            [section["title"] for section in sections],
            [
                "§8.1.1 Fetch-and-batch",
                "Effect of GIL removal",
                "SHM coalescing",
                "§8.1.2 VLM pipeline — Figure 7",
                "VLM pipeline comparisons",
            ],
        )
        self.assertEqual(sections[3]["rows"][0], ["grain", 8, "120.00"])
        self.assertEqual(sections[4]["rows"][0][1:], ["20.00", "%"])
