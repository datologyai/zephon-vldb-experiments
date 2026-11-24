"""TTFB protocol checks use synthetic arena artifacts; no loader or VM is run."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vldb_experiments import cli, core, ttfb, ttfb_report


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artifacts(attempt, cases, problem=None):
    rows, observations, trials = [], [], []
    for case in cases:
        identity = dict(
            loader_name=case["loader_name"],
            **{
                "run.batch_size": case["batch_size"],
                "run.num_workers": case["num_workers"],
                "run.seed": case["seed"],
            },
        )
        trial = dict(
            factory_time=0.1,
            iter_time=0.2,
            first_next_time=0.3,
            iter_to_batch=0.5,
            total_ttfb=0.6,
        )
        result = dict(
            name=case["loader_name"],
            batch_size=case["batch_size"],
            num_workers=case["num_workers"],
            seed=case["seed"],
            repetitions=1,
            trials=[trial],
            ttfb_median=0.6,
            factory_time_median=0.1,
            iter_time_median=0.2,
            first_next_time_median=0.3,
            iter_to_batch_median=0.5,
        )
        row = dict(
            identity,
            **{
                "benchmark.repetitions": 1,
                "config.plaintext": json.dumps(case["config"]),
            },
        )
        for metric in ttfb_report.TIMINGS:
            column = ttfb_report.TIMINGS[metric]
            row[column] = result[column.removeprefix("result.").removesuffix("_s")]
        rows.append(row)
        observations.append(
            dict(
                result=result,
                resolved_config={},
                cache_policy="best-effort page-cache drop",
                page_cache_dropped=[True],
                environment=dict(
                    gil_enabled=not core.is_free_threaded(case),
                    free_threaded_build=core.is_free_threaded(case),
                ),
            )
        )
        trials.append(
            dict(
                identity, trial=1, **{"result." + k + "_s": v for k, v in trial.items()}
            )
        )
    payload = dict(schema_version=1, metric="ttfb", failures=[], results=observations)
    if problem:
        problem(rows, payload, trials)
    root = attempt / "sweep/csv"
    csv_file(root / "result.csv", rows)
    csv_file(root / "result-trials.csv", trials)
    core.write_json(root / "result.json", payload)


class TTFBProtocolTests(unittest.TestCase):
    def test_plan_covers_paper_scenarios_gil_variants_and_scaling(self):
        cases = ttfb.plan()
        self.assertEqual(len(cases), 345)
        self.assertEqual(len(ttfb.jobs(cases)), 8)
        self.assertEqual({c["seed"] for c in cases}, {42, 67, 1337})
        self.assertEqual({c["variant"] for c in cases}, {"baseline"})
        self.assertEqual({c["system"] for c in cases}, {"grain", "litdata", "zephon"})
        self.assertTrue(
            all(c["repetitions"] == 1 and "warmup_batches" not in c for c in cases)
        )
        litdata = [
            c
            for c in cases
            if c["scenario"] == "fetch-text" and c["system"] == "litdata"
        ]
        self.assertEqual({c["dop"] for c in litdata}, {1, 2, 4, 8, 16, 32, 64})
        full = [
            c
            for c in cases
            if c["scenario"] == "vlm-pipeline" and c["system"] == "zephon"
        ]
        self.assertEqual(len(full), 81)
        self.assertTrue(
            all(
                c["batch_size"] == 4 and c["config"]["vlm_zephon"]["flatten_payload"]
                for c in full
            )
        )
        self.assertEqual(
            cli.main(["plan", "ttfb", "--scenario", "fetch-text", "--dop", "64"]), 0
        )

    def test_sweep_keeps_all_configurations_together_and_preserves_trial_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = {
                "paths": dict(
                    arena="/arena", environments="/envs", text="/text", vlm="/vlm"
                )
            }
            for name, cases in ttfb.jobs(ttfb.plan()).items():
                attempt = root / name
                command = ttfb.sweep_command(cases, profile, "v0.1.0", attempt)
                config = core.read_json(attempt / "sweep.json")
                self.assertEqual(command[1], "/arena/sweep.py")
                self.assertEqual(config["bench_command"], "dataloader-ttfb")
                self.assertFalse(config["preinstall"])
                self.assertFalse(
                    {
                        "--warmup",
                        "--steps",
                        "--packing-stats",
                        "--json-output",
                        "--per-trial-csv",
                    }
                    & set(config["bench_args"])
                )
                self.assertEqual(
                    "--require-no-gil" in config["bench_args"],
                    cases[0]["python"] == "3.14t",
                )
                if cases[0]["system"] == "zephon":
                    self.assertGreater(
                        len(core.read_json(attempt / "benchmark.json")["loaders"]), 1
                    )

    def test_collect_retains_json_trials_and_accepts_both_arena_csv_files(self):
        cases = ttfb.plan(
            scenarios=["fetch-text"], systems=["zephon"], dops=[64], seeds=[42]
        )
        cases = next(iter(ttfb.jobs(cases).values()))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "jobs/check/attempt-1"
            artifacts(attempt, cases)
            ttfb.collect(cases, attempt, root)
            for case in cases:
                result = core.read_json(root / "cases" / case["id"] / "result.json")
                self.assertEqual(result["result"]["trials"][0]["total_ttfb"], 0.6)
                self.assertEqual(result["page_cache_dropped"], [True])
                self.assertEqual(
                    result["execution"]["entrypoint"], "sweep.py -> dataloader-ttfb"
                )

    def test_invalid_observations_are_never_partially_promoted(self):
        cases = ttfb.plan(
            scenarios=["fetch-text"], systems=["grain"], dops=[64], seeds=[42]
        )

        def modify(problem):
            def change(rows, payload, trials):
                if problem == "seed":
                    rows[0]["run.seed"] = 123
                elif problem == "gil":
                    payload["results"][0]["environment"]["gil_enabled"] = False
                elif problem == "config":
                    rows[0]["config.plaintext"] = "{}"
                elif problem == "timing":
                    payload["results"][0]["result"]["trials"][0]["total_ttfb"] = 1
                elif problem == "csv":
                    rows[0]["result.ttfb_median_s"] = 1
                elif problem == "trial":
                    trials[0]["result.total_ttfb_s"] = 1
                elif problem == "failed":
                    payload["failures"] = [{"error": "failed"}]
                elif problem == "metric":
                    payload["metric"] = "throughput"
                elif problem == "extra":
                    payload["results"] *= 2

            return change

        for problem in [
            "seed",
            "gil",
            "config",
            "timing",
            "csv",
            "trial",
            "failed",
            "metric",
            "extra",
        ]:
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                attempt = root / "jobs/check/attempt-1"
                artifacts(attempt, cases, modify(problem))
                with self.assertRaises((ValueError, KeyError)):
                    ttfb.collect(cases, attempt, root)
                self.assertFalse((root / "cases").exists())

    def test_cache_confirmation_is_required_for_every_trial_before_promotion(self):
        cases = ttfb.plan(
            scenarios=["fetch-text"], systems=["grain"], dops=[64], seeds=[42, 67]
        )
        for drops in (None, [], [False], [None], [1], ["true"], [True, True]):
            with self.subTest(drops=drops), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                attempt = root / "jobs/check/attempt-1"

                def corrupt(rows, payload, trials):
                    # The first case is valid; the second must stop the whole sweep.
                    observation = payload["results"][1]
                    if drops is None:
                        observation.pop("page_cache_dropped")
                    else:
                        observation["page_cache_dropped"] = drops

                artifacts(attempt, cases, corrupt)
                before = {p: p.read_bytes() for p in attempt.rglob("*") if p.is_file()}
                with self.assertRaisesRegex(ValueError, "page-cache"):
                    ttfb.collect(cases, attempt, root)
                self.assertFalse((root / "cases").exists())
                self.assertEqual(
                    before,
                    {p: p.read_bytes() for p in attempt.rglob("*") if p.is_file()},
                )

    def test_one_successful_drop_does_not_validate_other_repetitions(self):
        with self.assertRaisesRegex(ValueError, "page-cache"):
            ttfb.require_cold_cache({"page_cache_dropped": [True, False]}, 2)
        ttfb.require_cold_cache({"page_cache_dropped": [True, True]}, 2)

    def test_resume_skips_completed_sweep_and_rejects_changed_input_contents(self):
        cases = ttfb.plan(
            scenarios=["fetch-text"], systems=["grain"], dops=[64], seeds=[42]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            profile = {
                "paths": dict(
                    environments=str(root / "envs"),
                    arena=str(root),
                    text="/text",
                    vlm="/vlm",
                )
            }

            def manifest(selected, profile, revision, output):
                core.write_json(output / "campaign.json", dict(cases=selected))

            def process(command, *args):
                attempt = Path(command[command.index("--config") + 1]).parent
                artifacts(attempt, cases)
                return 0

            with (
                patch.object(
                    ttfb.ttfb_data,
                    "check",
                    return_value=dict(valid=True, errors=[], warnings=[]),
                ),
                patch.object(core, "record_campaign_inputs", side_effect=manifest),
                patch.object(core, "run_process_group", side_effect=process) as launch,
                patch.object(
                    ttfb.provenance,
                    "validate_datasets",
                    return_value={"text": "same-hash"},
                ) as hashes,
            ):
                self.assertEqual(ttfb.run(cases, profile, "v0.1.0", output, 60), 0)
                self.assertEqual(ttfb.run(cases, profile, "v0.1.0", output, 60), 0)
                self.assertEqual(launch.call_count, 1)

                result_path = output / "cases" / cases[0]["id"] / "result.json"
                original = result_path.read_bytes()
                result = core.read_json(result_path)
                result["page_cache_dropped"] = [False]
                core.write_json(result_path, result)
                with self.assertRaisesRegex(ValueError, "page-cache"):
                    ttfb.run(cases, profile, "v0.1.0", output, 60)
                self.assertEqual(launch.call_count, 1)
                result_path.write_bytes(original)

                hashes.return_value = {"text": "changed-hash"}
                with self.assertRaisesRegex(ValueError, "contents changed"):
                    ttfb.run(cases, profile, "v0.1.0", output, 60)
                self.assertEqual(launch.call_count, 1)

    def test_busy_measurement_lock_prevents_setup_or_measurement(self):
        import fcntl

        cases = ttfb.plan(
            scenarios=["fetch-text"], systems=["grain"], dops=[64], seeds=[42]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "host.lock"
            profile = {
                "paths": dict(environments=str(root / "envs")),
                "measurement": {"lock": str(lock)},
            }
            with (
                lock.open("a") as held,
                patch.object(core, "record_campaign_inputs") as record,
            ):
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    ttfb.run(cases, profile, "v0.1.0", root / "run", 60)
                record.assert_not_called()

    def test_failed_input_preflight_prevents_fingerprinting_hashing_or_measurement(
        self,
    ):
        cases = ttfb.plan(
            scenarios=["fetch-text"], systems=["grain"], dops=[64], seeds=[42]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = {"paths": dict(environments=str(root / "envs"))}
            bad = dict(valid=False, errors=["throughput-sized text"], warnings=[])
            with (
                patch.object(ttfb.ttfb_data, "check", return_value=bad),
                patch.object(core, "record_campaign_inputs") as record,
                patch.object(core, "run_process_group") as launch,
                patch.object(ttfb.provenance, "validate_datasets") as hashes,
                self.assertRaisesRegex(ValueError, "throughput-sized"),
            ):
                ttfb.run(cases, profile, "v0.1.0", root / "run", 60)
            record.assert_not_called()
            launch.assert_not_called()
            hashes.assert_not_called()
            self.assertEqual(core.read_json(root / "run/input-preflight.json"), bad)


class TTFBReportTests(unittest.TestCase):
    def test_summary_rejects_unconfirmed_cache_drops_without_changing_results(self):
        cases = ttfb.plan(
            scenarios=["fetch-text"], systems=["grain"], dops=[64], seeds=[42]
        )
        for drops in (None, [False], [None]):
            with self.subTest(drops=drops), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary)
                attempt = run / "jobs/check/attempt-1"
                artifacts(attempt, cases)
                ttfb.collect(cases, attempt, run)
                core.write_json(
                    run / "campaign.json", dict(cases=cases, datasets={}, profile={})
                )
                path = run / "cases" / cases[0]["id"] / "result.json"
                result = core.read_json(path)
                if drops is None:
                    result.pop("page_cache_dropped")
                else:
                    result["page_cache_dropped"] = drops
                core.write_json(path, result)
                before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
                with self.assertRaisesRegex(ValueError, "page-cache"):
                    ttfb_report.observations(run)
                self.assertEqual(
                    before, {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
                )

    def test_mean_and_sample_deviation_across_seeds(self):
        rows = []
        for seed, value in zip([42, 67, 1337], [1, 2, 6]):
            row = dict(
                loader_name="litdata-shuffled",
                sweep_python="3.13",
                scenario="fetch-text",
                source="archive",
            )
            row.update(
                {
                    "run.num_workers": 64,
                    "run.batch_size": 32,
                    "run.seed": seed,
                    "benchmark.repetitions": 1,
                }
            )
            row.update({column: value for column in ttfb_report.TIMINGS.values()})
            rows.append(row)
        point = ttfb_report.aggregate(rows)[0]
        self.assertEqual(point["ttfb_s"], 3)
        self.assertAlmostEqual(point["ttfb_std_s"], math.sqrt(7))
        with self.assertRaisesRegex(ValueError, "three paper seeds"):
            ttfb_report.aggregate(rows + [rows[0]])

    def test_report_uses_fixed_paper_allocation_and_preserves_raw_inputs(self):
        cases = ttfb.plan(dops=[64])
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for name, selected in ttfb.jobs(cases).items():
                attempt = run / "jobs" / name / "attempt-1"
                artifacts(attempt, selected)
                ttfb.collect(selected, attempt, run)
            for case in cases:
                if case["loader_name"] == "zephon-fused-dop64-f8-p2c2":
                    path = run / "cases" / case["id"] / "result.json"
                    result = core.read_json(path)
                    result["csv_row"]["result.ttfb_median_s"] = 9.0
                    core.write_json(path, result)
            data = {"samples": 100, "files": []}
            core.write_json(
                run / "campaign.json",
                dict(cases=cases, datasets={"vlm": data, "text": data}, profile={}),
            )
            before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
            summary = ttfb_report.report(run)
            self.assertEqual(
                summary["vlm_selection"]["loader_name"], "zephon-fused-dop64-f8-p2c2"
            )
            row = next(
                r
                for r in summary["table"]
                if r["metric"] == "vlm-pipeline: Zephon processes (GIL)"
            )
            self.assertEqual(row["ttfb_s"], 9.0)
            self.assertEqual(summary["measured_metrics"], 15)
            self.assertEqual(summary["metric_count"], 19)
            self.assertIn("Inputs: public", summary["input_note"])
            self.assertEqual(
                before, {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
            )

    def test_reporting_requires_a_campaign_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                ttfb_report.report(Path(temporary))


if __name__ == "__main__":
    unittest.main()
