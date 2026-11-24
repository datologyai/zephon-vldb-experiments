"""Training commands; cloud dependencies are loaded only when needed."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

from . import protocol, sources, trace


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_manifest(run: Path) -> dict:
    return json.loads((run / "campaign.json").read_text())


def print_recipe(config: dict, *, file=None) -> None:
    scope = protocol.recipe_status(config)
    print(
        f"Recipe: {config['model']}; local/global batch {config['local_batch']}/{config['sequences_per_step']}; sequence {config['sequence_length']}",
        file=file,
    )
    print(
        f"Model/data seeds {config['seed']}/{config['data_seed']}; LR {config['optimizer']['lr']}; warmup {config['lr_scheduler']['warmup_steps']}; schedule horizon {config['lr_scheduler']['total_steps']}",
        file=file,
    )
    print(f"Scope: {scope['scope']}", file=file)
    for note in scope["scope_notes"]:
        print(f"  {note}", file=file)


def dependency_identity(config: dict) -> dict:
    """Record the zephon-modal implementation uploaded to Modal."""
    sources.use_modal(config)
    import modal
    import zephon_modal

    if modal.__version__ != config["sources"]["modal"]:
        raise ValueError(
            f"Install modal=={config['sources']['modal']} for this campaign"
        )
    from importlib.metadata import PackageNotFoundError, distribution

    package = Path(zephon_modal.__file__).parent
    try:
        direct = json.loads(
            distribution("zephon-modal").read_text("direct_url.json") or "{}"
        )
    except PackageNotFoundError:
        if "zephon_modal" not in config.get("local_sources", {}):
            raise
        direct = {}
    commit = direct.get("vcs_info", {}).get("commit_id")
    if (package.parent / ".git").exists():
        commit = subprocess.check_output(
            ["git", "-C", str(package.parent), "rev-parse", "HEAD"], text=True
        ).strip()
    if "zephon_modal" not in config.get("local_sources", {}):
        if commit != config["sources"]["zephon_modal"]:
            raise ValueError(
                "Install the pinned zephon-modal revision or select --zephon-modal PATH"
            )
        if (package.parent / ".git").exists():
            dirty = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(package.parent),
                    "status",
                    "--porcelain",
                    "--",
                    str(package),
                ],
                text=True,
            )
            if dirty.strip():
                raise ValueError(
                    "Use a clean pinned zephon-modal or select --zephon-modal PATH"
                )
    files = {
        str(p.relative_to(package)): protocol.file_digest(p)
        for p in sorted(package.rglob("*.py"))
        if p.name != "_version.py"
    }
    return {
        "commit": config.get("local_sources", {})
        .get("zephon_modal", {})
        .get("state", {})
        .get("revision", commit),
        "files_sha256": protocol.digest(files),
    }


def run_campaign(args) -> None:
    if getattr(args, "ssh", None):
        from .ssh import run_campaign as run_ssh

        return run_ssh(args)
    if getattr(args, "remote_root", None):
        raise ValueError("--remote-root requires --ssh")
    import modal

    from .modal_app import build_app

    output = args.output.expanduser().resolve()
    config = sources.configure(
        args, protocol.load(args.config, args.smoke, getattr(args, "small", False))
    )
    prefix = protocol.validate_prefix(args.prefix)
    source = protocol.code_identity()
    dependencies = dependency_identity(config)
    manifest = {
        "schema": 1,
        "config": config,
        "prefix": prefix,
        "code": source,
        "zephon_modal": dependencies,
    }
    manifest["identity"] = protocol.digest(manifest)
    from .preparation import require_inputs

    require_inputs(args, config)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "campaign.json").exists() and read_manifest(output) != manifest:
        raise ValueError(
            "Local campaign identity changed; use the original checkout or a new output/prefix"
        )
    write_json(output / "campaign.json", manifest)
    app, coordinator, _, registry, launch_arguments = build_app(manifest)
    existing = registry.get("submission")
    if existing is not None:
        if existing["identity"] != manifest["identity"]:
            raise ValueError(
                "Modal prefix belongs to another campaign; choose a new prefix"
            )
        if not existing.get("call_id"):
            raise RuntimeError(
                "Prepared submission has uncertain state; inspect Modal, do not resubmit"
            )
        write_json(output / "launch.json", existing)
        print(f"Already submitted: {existing['call_id']}")
        return
    if not registry.put(
        "submission",
        {"state": "prepared", "identity": manifest["identity"]},
        skip_if_exists=True,
    ):
        raise RuntimeError(
            "Another client claimed this campaign; rerun to attach to its submission"
        )
    # Persist the uncertain state before the first cloud mutation too.
    write_json(
        output / "launch.json", {"state": "prepared", "identity": manifest["identity"]}
    )
    spawn_started = False
    try:
        with modal.enable_output(), app.run(detach=True):
            spawn_started = True
            call = coordinator.spawn(manifest, **launch_arguments)
            submission = {
                "state": "submitted",
                "identity": manifest["identity"],
                "call_id": call.object_id,
            }
            registry.put("submission", submission)
            write_json(output / "launch.json", submission)
    except Exception:  # Release only a claim that definitely never submitted a call
        if not spawn_started:
            registry.pop("submission")
            write_json(
                output / "launch.json",
                {"state": "build_failed", "identity": manifest["identity"]},
            )
        raise
    print(f"Submitted {call.object_id}; inspect with: reproduce status {output}")


def status(run: Path) -> dict:
    if read_manifest(run).get("backend", {}).get("kind") == "ssh":
        from .ssh import status as ssh_status

        return ssh_status(run)
    import modal

    launch = json.loads((run / "launch.json").read_text())
    if not launch.get("call_id"):
        return launch
    try:
        result = modal.FunctionCall.from_id(launch["call_id"]).get(timeout=0)
    except TimeoutError:
        return launch | {"state": "running"}
    except modal.exception.InputCancellation:
        return launch | {"state": "failed", "error": "Coordinator was cancelled"}
    except (
        modal.exception.ConnectionError,
        modal.exception.ServiceError,
        modal.exception.AuthError,
        modal.exception.NotFoundError,
    ) as error:
        return launch | {"state": "unknown", "error": str(error)}
    except Exception as error:  # noqa: BLE001 - remote exceptions retain their original Python type
        return launch | {"state": "failed", "error": str(error)}
    exact = all(
        record.get("comparison", {}).get("exact", True) for record in result.values()
    )
    expected = all(
        record.get("comparison", {}).get(
            "expected_outcome_met", record.get("comparison", {}).get("exact", True)
        )
        for record in result.values()
    )
    config = read_manifest(run)["config"]
    stages = protocol.plan(config)
    if set(result) != {stage["name"] for stage in stages}:
        return launch | {
            "state": "failed",
            "error": "Incomplete campaign stage coverage",
        }
    arms = list(dict.fromkeys(protocol.stage_arm(stage) for stage in stages))
    return launch | {
        "state": "complete",
        "exact": exact,
        "expected_outcomes_met": expected,
        "arms": arms,
        "experiment_scale": protocol.experiment_scale(config),
        "paper_protocol_match": protocol.figure1_geometry(config)
        and protocol.recipe_status(config)["paper_training_settings_match"]
        and set(arms) == {"control", "zephon"},
        "reproduction": protocol.recipe_status(config),
        "stages": list(result),
    }


def continue_campaign(run: Path) -> None:
    """Recover a terminal coordinator while attaching to its recorded GPU calls.

    Only analysis/orchestration code may change. The recipe, adapter, protocol,
    dependency source and image requirements must retain their original hashes.
    """
    if read_manifest(run).get("backend", {}).get("kind") == "ssh":
        from .ssh import continue_campaign as continue_ssh

        return continue_ssh(run)
    import modal
    from modal.types import InputStatus

    from .modal_app import build_app

    run = run.expanduser().resolve()
    manifest = read_manifest(run)
    current = protocol.code_identity()
    controller_files = {
        f"src/vldb_experiments/training/{name}.py"
        for name in ("cli", "modal_app", "trace", "plotting")
    }
    for name in manifest["code"]["files"].keys() | current["files"].keys():
        if name not in controller_files and (
            manifest["code"]["files"].get(name) != current["files"].get(name)
        ):
            raise ValueError(
                f"Training source changed: {name}; cannot continue this campaign"
            )
    if dependency_identity(manifest["config"]) != manifest["zephon_modal"]:
        raise ValueError("Training dependency changed; cannot continue this campaign")
    previous = json.loads((run / "launch.json").read_text())
    call_id = previous.get("call_id")
    if not call_id:
        raise RuntimeError("Uncertain submission: inspect Modal before continuing")
    previous_state = status(run)
    if previous_state["state"] == "complete":
        print(f"Already complete: {call_id}")
        return
    if previous_state["state"] != "failed":
        raise RuntimeError(
            "Coordinator is active or its state is uncertain; no duplicate submitted"
        )
    # The result is authoritative; additionally require the recorded input's
    # terminal state before replacing the coordinator. A missing graph is unsafe.
    graph = modal.FunctionCall.from_id(call_id).get_call_graph()
    own = [node for node in graph if node.function_call_id == call_id]
    if len(own) != 1 or own[0].status == InputStatus.PENDING:
        raise RuntimeError(
            "Coordinator is active or its state is uncertain; no duplicate submitted"
        )
    if own[0].status == InputStatus.SUCCESS:
        print(f"Already complete: {call_id}")
        return
    app, coordinator, _, registry, arguments = build_app(manifest)
    key = f"continue:{call_id}"
    existing = registry.get(key)
    if existing is not None:
        if not existing.get("call_id"):
            raise RuntimeError(
                "Continuation submission is uncertain; inspect Modal, do not resubmit"
            )
        write_json(run / "launch.json", existing)
        print(f"Already submitted: {existing['call_id']}")
        return
    if not registry.put(
        key,
        {"state": "prepared", "identity": manifest["identity"]},
        skip_if_exists=True,
    ):
        raise RuntimeError(
            "Another client claimed this continuation; no duplicate submitted"
        )
    controller = {"code": current, "previous_call_id": call_id}
    write_json(
        run / "controllers" / f"{call_id}.json", {"previous": previous, **controller}
    )
    spawn_started = False
    try:
        with modal.enable_output(), app.run(detach=True):
            spawn_started = True
            call = coordinator.spawn(manifest, controller=controller, **arguments)
            submission = {
                "state": "submitted",
                "identity": manifest["identity"],
                "call_id": call.object_id,
            }
            registry.put(key, submission)
            registry.put("submission", submission)
            write_json(run / "launch.json", submission)
    except Exception:
        if not spawn_started:
            registry.pop(key)
        raise
    print(
        f"Continued {call.object_id}; existing GPU call IDs and successful stages are reused"
    )


def collect(run: Path) -> dict:
    if read_manifest(run).get("backend", {}).get("kind") == "ssh":
        from .ssh import collect as collect_ssh

        return collect_ssh(run)
    import modal

    manifest = read_manifest(run)
    volume = modal.Volume.from_name(f"{manifest['prefix']}-training-runs")

    def download(remote: str, local: Path):
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_suffix(local.suffix + ".tmp")
        with temporary.open("wb") as stream:
            for chunk in volume.read_file(remote.lstrip("/")):
                stream.write(chunk)
        temporary.replace(local)

    remote_manifest = json.loads(b"".join(volume.read_file("campaign.json")))
    if remote_manifest != manifest:
        raise ValueError("Remote and local campaign identities differ")
    assets = modal.Volume.from_name(f"{manifest['prefix']}-training-assets")
    receipt = json.loads(b"".join(assets.read_file("assets.json")))
    write_json(run / "assets.json", receipt)
    result = json.loads(b"".join(volume.read_file("result.json")))
    for stage in protocol.plan(manifest["config"]):
        record = result[stage["name"]]
        remote = record["directory"].removeprefix("/runs/")
        local = run / stage["name"]
        for rank in range(stage["dp"]):
            download(
                f"{remote}/diagnostics/rank-{rank}.jsonl",
                local / f"diagnostics/rank-{rank}.jsonl",
            )
        for name in ("observed.json", "checkpoint-proof.json"):
            download(f"{remote}/{name}", local / name)
        download(
            f"{remote}/checkpoint/step-{stage['end_step']}/.metadata",
            local / "final.metadata",
        )
        if protocol.source_name(stage) is None:
            download(
                f"{remote}/checkpoint/step-{manifest['config']['checkpoint_step']}/.metadata",
                local / "source.metadata",
            )
        write_json(local / "launch.json", record)
    write_json(run / "result.json", result)
    try:
        controllers = list(volume.iterdir("controllers", recursive=False))
    except (FileNotFoundError, modal.exception.NotFoundError):
        controllers = []
    for entry in controllers:
        download(entry.path, run / "controllers" / Path(entry.path).name)
    return trace.report(run)


def wait_and_collect(run: Path, plots: Path | None = None) -> dict:
    """The remote owner remains detached; this client waits and publishes results."""
    from ..summary import finish

    previous = None
    while True:
        state = status(run)
        if state["state"] != previous:
            print("Training: " + state["state"], file=sys.stderr, flush=True)
            previous = state["state"]
        if state["state"] == "complete":
            break
        if state["state"] in ("failed", "unknown", "cancelled"):
            raise RuntimeError(
                f"Training {state['state']}: {state.get('error', '')}. Remote evidence is preserved at {run}."
            )
        time.sleep(30)
    with redirect_stdout(sys.stderr):
        result = collect(run)
    finish(run, plots=plots)
    return result


def main(args) -> int:
    command = args.training_command
    if command == "plan":
        config = sources.configure(
            args, protocol.load(args.config, args.smoke, args.small)
        )
        result = {
            "config": config,
            "stages": protocol.plan(config),
            "reproduction": protocol.recipe_status(config),
        }
        if args.output:
            write_json(args.output, result)
        print_recipe(config)
        print(f"training: {len(result['stages'])} stages")
        print(
            f"{config['steps']} optimizer steps; checkpoint at step {config['checkpoint_step']}"
        )
        for stage in result["stages"]:
            print(f"  {stage['name']}: DP {stage['dp']}")
    elif command == "run":
        print_recipe(
            protocol.load(args.config, args.smoke, args.small), file=sys.stderr
        )
        if not args.detach and not args.smoke:
            from ..report import plotting_python

            plotting_python(args.plots)
        with redirect_stdout(sys.stderr):
            run_campaign(args)
        if not args.detach:
            result = wait_and_collect(args.output, args.plots)
            return int(not result.get("expected_outcomes_met", result["exact"]))
    elif command == "continue":
        continue_campaign(args.run)
    elif command == "status":
        result = status(args.run)
        print(json.dumps(result, indent=2))
        return int(
            result["state"] in {"failed", "unknown"}
            or result.get("expected_outcomes_met", result.get("exact")) is False
        )
    elif command == "collect":
        from ..report import plotting_python
        from ..summary import finish

        plotting_python(args.plots)
        with redirect_stdout(sys.stderr):
            result = collect(args.run)
        finish(args.run, plots=args.plots)
        return int(not result.get("expected_outcomes_met", result["exact"]))
    return 0
