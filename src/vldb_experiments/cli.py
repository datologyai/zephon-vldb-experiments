"""Public CLI for the VLDB experiments and their CPU execution targets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

from . import core, machines, matrix, prepare, profiles, provenance, summary
from .report import plotting_python

EXPERIMENTS = ("fetch-and-batch", "vlm-pipeline")


def selections(parser):
    group = parser.add_argument_group(
        "Optional subset (not a complete paper reproduction)"
    )
    group.add_argument("--system", action="append", default=[])
    group.add_argument("--seed", action="append", type=int, default=[])
    group.add_argument("--dop", action="append", type=int, default=[])
    return group


def training_options(parser, *, target=False, experiment=True):
    parser.add_argument(
        "--small", action="store_true", help="Single-node experiment, DP2 -> DP1/4/8"
    )
    for name in ("zephon", "torchtitan", "zephon-modal"):
        parser.add_argument(
            "--" + name, type=Path, help=f"Use this local {name} source directory"
        )
    if experiment:
        parser.add_argument(
            "--smoke", action="store_true", help="Short setup check; no paper figure"
        )
    else:
        parser.set_defaults(smoke=False)
    if target:
        parser.add_argument("--prefix", required=True)
        parser.add_argument(
            "--ssh", metavar="HOST", help="Use a supplied GPU host instead of Modal"
        )
        parser.add_argument(
            "--remote-root", help="Absolute storage root on the supplied SSH host"
        )


def parser():
    result = argparse.ArgumentParser(
        prog="reproduce", description="Reproduce the Zephon VLDB experiments."
    )
    commands = result.add_subparsers(dest="command", required=True)
    machine = commands.add_parser(
        "machine", help="Create an AWS CPU host or prepare a local/SSH CPU target"
    )
    actions = machine.add_subparsers(dest="machine_command", required=True)
    for action in ("create", "setup"):
        p = actions.add_parser(action)
        p.add_argument("--profile", type=Path, required=True)
        if action in ("create", "setup"):
            p.add_argument(
                "--plan",
                action="store_true",
                help="Preview without creating or changing anything",
            )
    p = commands.add_parser(
        "prepare",
        help="Prepare sources, environments, assets or experiment inputs",
    )
    components = p.add_subparsers(dest="component", required=True)
    for name in (
        "sources",
        "environments",
        "plotting",
        "assets",
        "throughput-data",
        "ttfb-data",
        "checkpoint-data",
        "training-data",
    ):
        p = components.add_parser(name)
        p.set_defaults(config=None)
        if name == "training-data":
            training_options(p, target=True, experiment=False)
        elif name == "plotting":
            p.add_argument(
                "--plots", type=Path, help="Local zephon-vldb-plots checkout"
            )
        else:
            p.add_argument("--profile", type=Path, required=True)
        p.add_argument(
            "--plan",
            action="store_true",
            help="Show paths/resources without transfers or installation",
        )
        if name == "checkpoint-data":
            p.add_argument(
                "--yes",
                action="store_true",
                help="Explicitly accept the multi-terabyte preparation estimate",
            )
            p.add_argument(
                "--abandon",
                action="store_true",
                help="Discard this unfinished preparation using its journal",
            )
        if name == "environments":
            p.add_argument(
                "--only", action="append", choices=("cpu-3.13", "cpu-3.14t"), default=[]
            )
    for verb in ("plan", "check", "run"):
        p = commands.add_parser(verb)
        experiments = p.add_subparsers(dest="experiment", required=True)
        names = (
            ("throughput", "ttfb", "checkpoint", "training")
            if verb != "check"
            else ("machine", "throughput", "ttfb", "checkpoint", "training")
        )
        for experiment in names:
            p = experiments.add_parser(experiment)
            p.set_defaults(revision="v0.1.0", config=None)
            if experiment != "training" and verb != "plan":
                p.add_argument("--profile", type=Path, required=True)
            if verb == "run":
                p.add_argument("--output", type=Path, required=True)
                p.add_argument(
                    "--detach",
                    action="store_true",
                    help="Submit a remote run and collect its results later",
                )
            elif verb == "plan":
                p.add_argument("--output", type=Path)
            else:
                p.add_argument(
                    "--output", type=Path, help="Optionally save the validation receipt"
                )
                if experiment in ("throughput", "ttfb"):
                    p.add_argument(
                        "--hash-data",
                        action="store_true",
                        help="Read every input chunk to verify its checksum",
                    )
                if experiment == "checkpoint":
                    p.add_argument(
                        "--all-objects",
                        action="store_true",
                        help="Validate every S3 object's server checksum",
                    )
                if experiment == "training":
                    training_options(p, target=True, experiment=False)
                continue
            if experiment == "throughput":
                p.add_argument(
                    "--figure",
                    choices=("6", "7"),
                    help="Default: both throughput figures",
                )
                group = selections(p)
                group.add_argument(
                    "--panel", choices=("all", "overview", "gil", "shm"), default="all"
                )
                group.add_argument(
                    "--runner", choices=("process", "threads", "workers")
                )
                group.add_argument(
                    "--smoke",
                    action="store_true",
                    help="Short setup check; no paper figures",
                )
                if verb == "run":
                    p.add_argument("--timeout", type=float, default=1200)
                    p.add_argument("--cooldown", type=float, default=2)
                    p.add_argument("--keep-going", action="store_true")
            elif experiment == "ttfb":
                group = selections(p)
                group.add_argument(
                    "--scenario",
                    action="append",
                    choices=("fetch-text", "fetch-vlm", "vlm-pipeline"),
                    default=[],
                )
                if verb == "run":
                    p.add_argument("--timeout", type=float, default=7200)
            elif experiment == "checkpoint":
                p.add_argument(
                    "--suite",
                    choices=("all", "main", "progress", "lanes"),
                    default="all",
                )
                p.add_argument("--seed", type=int, action="append", default=[])
                p.add_argument("--repetition", type=int, action="append", default=[])
                if verb == "run":
                    p.add_argument("--timeout", type=float, default=7200)
                    p.add_argument("--retry-failed", action="store_true")
            elif experiment == "training":
                training_options(p, target=verb == "run")
                if verb == "run":
                    p.add_argument(
                        "--plots", type=Path, help="Local zephon-vldb-plots checkout"
                    )
    for verb in ("status", "collect", "resume", "summarize", "plot"):
        p = commands.add_parser(verb)
        p.add_argument("run", type=Path)
        if verb in ("collect", "summarize", "plot"):
            p.add_argument(
                "--plots", type=Path, help="Local zephon-vldb-plots checkout"
            )
        if verb == "summarize":
            p.add_argument("--format", choices=("terminal", "json"), default="terminal")
    return result


def planned_cases(args):
    experiments = (
        EXPERIMENTS if args.figure is None else (EXPERIMENTS[int(args.figure) - 6],)
    )
    return {
        name: matrix.plan(
            name,
            args.revision,
            panel=args.panel,
            systems=args.system,
            runner=args.runner,
            seeds=args.seed,
            dops=args.dop,
            smoke=args.smoke,
        )
        for name in experiments
    }


def show_plan(args):
    from . import checkpoint, ttfb

    if args.experiment == "throughput":
        selected = planned_cases(args)
        result = dict(
            experiment="throughput",
            count=sum(map(len, selected.values())),
            cases=selected,
        )
    elif args.experiment == "ttfb":
        selected = ttfb.plan(
            scenarios=args.scenario, systems=args.system, dops=args.dop, seeds=args.seed
        )
        result = dict(experiment="ttfb", count=len(selected), cases=selected)
    else:
        selected = checkpoint.plan(args.suite, args.seed, args.repetition)
        result = dict(experiment="checkpoint", count=len(selected), cases=selected)
    if args.output:
        core.write_json(args.output, result)
    print(f"{result['experiment']}: {result['count']} observations/jobs")
    cases = (
        [case for group in selected.values() for case in group]
        if isinstance(selected, dict)
        else selected
    )
    if args.experiment == "checkpoint":
        recipe = checkpoint.settings()
        print(
            f"  {recipe['world_size']} CPU ranks; global batch {recipe['global_batch_size']}; microbatch {recipe['micro_batch_size']}"
        )
        for label, field in (
            ("K", "k"),
            ("C", "chunk_size"),
            ("L", "lanes"),
            ("Steps", "optimizer_steps"),
        ):
            print(
                f"  {label}: "
                + ", ".join(map(str, sorted({case[field] for case in cases})))
            )
        print(
            "  Repetitions: "
            + ", ".join(map(str, sorted({case["repetition"] for case in cases})))
        )
        print("  Separate fresh, save and restore processes")
    else:
        if isinstance(selected, dict):
            for name, group in selected.items():
                print(f"  {name}: {len(group)} observations")
        print("  DOP: " + ", ".join(map(str, sorted({case["dop"] for case in cases}))))
        print("  Loaders: " + ", ".join(sorted({case["system"] for case in cases})))
    print("  Seeds: " + ", ".join(map(str, sorted({case["seed"] for case in cases}))))
    print("  No setup, data generation or measurements performed.")


def run_experiments(args):
    profile = profiles.load(args.profile)
    output = args.output.expanduser().absolute()
    cases = planned_cases(args)
    if not args.smoke:
        plotting_python(Path(profile["paths"]["plots"]))
    code = 0
    for name, selected in cases.items():
        destination = output / name if len(cases) > 1 else output
        with redirect_stdout(sys.stderr):
            code |= core.execute(
                selected,
                profile,
                args.revision,
                destination,
                args.timeout,
                args.keep_going,
                args.cooldown,
            )
        if code and not args.keep_going:
            break
    if code == 0 and not os.environ.get("VLDB_REMOTE_WORKER"):
        if args.smoke:
            print(f"Setup-check measurements saved: {output}")
        elif all(
            matrix.is_full_matrix(name, selected) for name, selected in cases.items()
        ):
            summary.finish(output, plots=Path(profile["paths"]["plots"]))
        else:
            summary.summarize(output, plots=Path(profile["paths"]["plots"]))
    return code


def run_prepare(args):
    if args.component == "plotting":
        path = (args.plots or prepare.controller_plots({})).expanduser().resolve()
        if args.plan:
            print(json.dumps({"plots": str(path)}, indent=2))
        else:
            prepare.plotting(path, existing=args.plots is not None)
        return
    purpose = (
        args.component.removesuffix("-data")
        if args.component.endswith("-data")
        else "throughput"
    )
    profile = profiles.load(args.profile, purpose)
    if args.plan:
        print(json.dumps(prepare.plan(profile, args.component), indent=2))
        return
    with machines.setup_lock(profile):
        if args.component == "sources":
            prepare.sources(profile, "v0.1.0")
        elif args.component == "environments":
            prepare.environments(profile, "v0.1.0", args.only)
        elif args.component == "assets":
            prepare.assets(profile, "v0.1.0")
        elif args.component == "throughput-data":
            prepare.throughput_data(profile)
        elif args.component == "ttfb-data":
            prepare.ttfb_data(profile)
        else:
            from . import checkpoint_data

            checkpoint_data.execute(
                checkpoint_data.load_profile(args.profile),
                yes=args.yes,
                abandon=args.abandon,
            )


def check(args):
    if args.experiment == "machine":
        result = machines.check()
    else:
        purpose = args.experiment
        profile = profiles.load(args.profile, purpose)
        with machines.setup_lock(profile):
            if purpose == "checkpoint":
                from . import checkpoint, checkpoint_data

                data = checkpoint.load_profile(args.profile)
                result = dict(
                    environment=checkpoint.source_digest(data),
                    inputs=checkpoint_data.execute(
                        data, check=True, all_objects=args.all_objects
                    ),
                )
            else:
                prepare.verify_assets(profile)
                env = prepare.check_environments(profile)
                if purpose == "ttfb":
                    from . import ttfb_data

                    result = ttfb_data.check(
                        profile, ["text", "vlm"], hashes=args.hash_data
                    )
                    ttfb_data.require_valid(result)
                else:
                    result = dict(
                        datasets=provenance.validate_datasets(
                            profile, ["text", "vlm"], hashes=args.hash_data
                        )
                    )
                result["environment"] = env
    if args.output:
        core.write_json(args.output, result)
    print(
        f"{args.experiment}: "
        + ("checks passed" if result.get("ready", True) else "setup required")
    )
    if args.experiment == "machine":
        print(
            f"  {result['system']} {result['architecture']}; {result['host']['cpu_count']} CPUs; uv: {result['uv'] or 'missing'}"
        )
    else:
        print(
            "  Sources, shared environment, preprocessing assets and input identity validated."
        )
        for name, data in result.get("datasets", {}).items():
            records = data.get("records", data.get("samples"))
            shards = data.get("shards", len(data.get("files", [])))
            print(f"  {name}: {records:,} records; {shards:,} shards")
        for warning in result.get("warnings", []):
            print("  Warning: " + warning)
    if args.output:
        print(f"  Validation receipt: {args.output}")
    if args.experiment == "machine" and not result["ready"]:
        return 1
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    if (args.command == "prepare" and args.component == "training-data") or (
        args.command == "check" and args.experiment == "training"
    ):
        from .training import preparation

        return preparation.main(args)
    if args.command == "machine" and args.machine_command == "create":
        print(json.dumps(machines.create(args.profile, plan=args.plan), indent=2))
        return 0
    if getattr(args, "profile", None):
        raw = profiles.read(args.profile)
        if raw.get("machine", {}).get("kind", "local") != "local":
            if getattr(args, "plan", False):
                local = profiles.resolve(
                    profiles.local_target(raw),
                    getattr(args, "component", "").removesuffix("-data"),
                )
                preview = prepare.plan(local, getattr(args, "component", "setup"))
                if args.command == "machine" and args.machine_command == "setup":
                    preview["controller"] = dict(
                        plots=str(prepare.controller_plots(raw))
                    )
                print(json.dumps(preview, indent=2))
                return 0
            from . import remote

            return remote.execute(args, argv)
    if args.command == "machine":
        profile = profiles.load(args.profile)
        if args.plan:
            print(json.dumps(prepare.plan(profile, "setup"), indent=2))
        else:
            machines.setup(profile)
    elif args.command == "prepare":
        run_prepare(args)
    elif args.command in ("plan", "run") and args.experiment == "training":
        from .training.cli import main as training_main

        args.training_command = args.command
        return training_main(args)
    elif args.command == "plan":
        show_plan(args)
    elif args.command == "check":
        return check(args)
    elif args.command == "run":
        if args.detach:
            raise ValueError("--detach needs an SSH/AWS target or training backend")
        if args.experiment == "throughput":
            return run_experiments(args)
        if args.experiment == "ttfb":
            from . import ttfb

            selected = ttfb.plan(
                scenarios=args.scenario,
                systems=args.system,
                dops=args.dop,
                seeds=args.seed,
            )
            with redirect_stdout(sys.stderr):
                ttfb.run(
                    selected,
                    profiles.load(args.profile, "ttfb"),
                    args.revision,
                    args.output,
                    args.timeout,
                )
        else:
            from . import checkpoint

            with redirect_stdout(sys.stderr):
                code = checkpoint.run(
                    checkpoint.load_profile(args.profile),
                    checkpoint.plan(args.suite, args.seed, args.repetition),
                    args.output,
                    timeout=args.timeout,
                    retry_failed=args.retry_failed,
                )
            if code:
                return code
        if not os.environ.get("VLDB_REMOTE_WORKER"):
            summary.finish(args.output)
    elif args.command in ("status", "collect", "resume"):
        if (args.run / "remote.json").exists():
            from . import remote

            if args.command == "status":
                print(json.dumps(remote.status(args.run), indent=2))
            elif args.command == "collect":
                remote.collect(args.run, plots=args.plots)
            else:
                remote.resume(args.run)
        elif summary.kind(args.run) == "training":
            from .training.cli import main as training_main

            args.training_command = (
                "continue" if args.command == "resume" else args.command
            )
            return training_main(args)
        elif args.command == "status":
            print(json.dumps(core.status(args.run), indent=2))
        else:
            raise ValueError(
                "Local CPU campaigns resume by repeating their run command; results already live locally"
            )
    elif args.command == "summarize":
        summary.summarize(args.run, plots=args.plots, format=args.format)
    elif args.command == "plot":
        paths = summary.plot(args.run, plots=args.plots)
        if paths:
            for path in paths:
                print(path)
        else:
            print(
                "No figure to generate for this run. Print its results with reproduce summarize RUN."
            )
    return 0


def entrypoint():
    try:
        return main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"reproduce: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(entrypoint())
