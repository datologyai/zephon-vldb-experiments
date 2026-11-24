import csv
import tempfile
import unittest
from pathlib import Path

from vldb_experiments import core, matrix


def observation(case):
    row = {
        "loader_name": case["loader_name"],
        "run.batch_size": case["batch_size"],
        "run.num_workers": case["num_workers"],
        "run.seed": case["seed"],
        "benchmark.warmup_steps": case["warmup_batches"],
        "benchmark.measure_steps": case["measure_batches"],
        "result.steps": case["measure_batches"],
    }
    return {
        "result": {
            "name": case["loader_name"],
            "steps": case["measure_batches"],
            "total_time": 1.0,
        },
        "csv_row": row,
        "resolved_config": {},
        "environment": {"free_threaded_build": True, "gil_enabled": False},
    }


def write_result(attempt, case, observations=None, failures=None):
    value = observation(case)
    core.write_json(
        attempt / "sweep/csv/result.json",
        {
            "schema_version": 1,
            "results": observations if observations is not None else [value],
            "failures": failures or [],
        },
    )
    csv_path = attempt / "sweep/csv/result.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(value["csv_row"]))
        writer.writeheader()
        writer.writerow(value["csv_row"])


class SweepProtocolTests(unittest.TestCase):
    def test_generated_configs_use_sweep_and_exact_batch_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = {
                "paths": {
                    "environments": str(root / "envs"),
                    "arena": str(root / "arena"),
                    "text": str(root / "data"),
                    "vlm": str(root / "data"),
                }
            }
            for case in [
                matrix.cases("fetch-and-batch")[0],
                next(
                    c
                    for c in matrix.cases("fetch-and-batch")
                    if c["python"].endswith("t")
                ),
                matrix.cases("vlm-pipeline")[0],
            ]:
                attempt = root / case["id"]
                command = core.sweep_command(profile, case, "v0.1.0", attempt)
                self.assertEqual(command[1], str(root / "arena/sweep.py"))
                self.assertIn("--no-sync", command)
                self.assertNotIn("dataloader_arena.single", command)
                cfg = core.read_json(attempt / "sweep.json")
                args = cfg["bench_args"]
                self.assertFalse(
                    {"--json-output", "--csv-output", "--config"} & set(args)
                )
                batch_size = case["batch_size"] or 1
                self.assertEqual(
                    int(args[args.index("--steps") + 1]),
                    case["measure_batches"] * batch_size,
                )
                self.assertEqual(
                    int(args[args.index("--warmup") + 1]),
                    case["warmup_batches"] * batch_size,
                )
                self.assertEqual(
                    "--require-no-gil" in args, case["python"].endswith("t")
                )
                self.assertEqual(
                    "--packing-stats" in args, bool(case.get("packing_stats"))
                )
                self.assertEqual(cfg["python_versions"], [case["python"]])
                self.assertFalse(cfg["preinstall"])
                self.assertEqual(
                    core.read_json(attempt / "benchmark.json")["loaders"][0]["config"],
                    case["config"],
                )

    def test_collect_preserves_actual_csv_and_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            attempt = directory / "attempt-1"
            case = matrix.cases("fetch-and-batch")[0]
            write_result(attempt, case)
            core.collect_sweep_result(case, attempt, directory)
            self.assertEqual(
                (directory / "result.csv").read_bytes(),
                (attempt / "sweep/csv/result.csv").read_bytes(),
            )
            result = core.read_json(directory / "result.json")
            self.assertEqual(result["result"], observation(case)["result"])
            self.assertEqual(
                result["execution"]["entrypoint"], "sweep.py -> dataloader-bench"
            )

    def test_collect_rejects_wrong_count_window_identity_gil_and_csv(self):
        case = next(
            c for c in matrix.cases("fetch-and-batch") if c["python"].endswith("t")
        )
        for problem in ("empty", "extra", "failed", "window", "seed", "gil", "csv"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                attempt = directory / "attempt-1"
                value = observation(case)
                values = [value]
                failures = []
                if problem == "empty":
                    values = []
                elif problem == "extra":
                    values *= 2
                elif problem == "failed":
                    failures = [{"error": "failed"}]
                elif problem == "window":
                    value["csv_row"]["benchmark.measure_steps"] += 1
                elif problem == "seed":
                    value["csv_row"]["run.seed"] = 999
                elif problem == "gil":
                    value["environment"]["gil_enabled"] = True
                write_result(
                    attempt,
                    case,
                    observations=values,
                    failures=failures,
                )
                if problem == "csv":
                    (attempt / "sweep/csv/result.csv").write_text(
                        "loader_name\nwrong\n"
                    )
                with self.assertRaises(ValueError):
                    core.collect_sweep_result(case, attempt, directory)
                self.assertFalse((directory / "result.json").exists())
