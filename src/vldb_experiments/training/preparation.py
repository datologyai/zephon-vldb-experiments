"""Prepare and check training inputs on the selected SSH host or Modal volume."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

from . import protocol, sources


def preview(args, config: dict) -> dict:
    prefix = protocol.validate_prefix(args.prefix)
    if getattr(args, "ssh", None):
        from . import ssh

        root = ssh.storage_root(args.remote_root)
        ssh.ssh_command(args.ssh, ["true"])
        destination = str(ssh.assets_directory(root, prefix))
    else:
        if getattr(args, "remote_root", None):
            raise ValueError("--remote-root requires --ssh")
        destination = f"{prefix}-training-assets (Modal volume)"
    return {
        "destination": destination,
        "host": getattr(args, "ssh", None) or "Modal",
        "dataset": config["dataset"]["repository"],
        "files": list(config["dataset"]["files"]),
        "tokenizer": config["tokenizer"]["repository"],
        "processing": "Download Parquet and tokenizer files, then build a Zephon index. Tokenization and packing run during training.",
        "gpus": 0,
    }


def ssh_inputs(args, config: dict, action: str) -> dict:
    from . import ssh

    assets = ssh.assets_directory(ssh.storage_root(args.remote_root), args.prefix)
    manifest = {"config": config, "code": protocol.code_identity()}
    image = ssh.image_name(manifest)
    if action == "prepare":
        # This context owns no campaign and can be removed after building.
        context = (
            ssh.remote(
                args.ssh,
                ["mktemp", "-d", "/tmp/zephon-training-data.XXXXXXXX"],
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )
        if not context.startswith(
            "/tmp/zephon-training-data."
        ) or "/" in context.removeprefix("/tmp/"):
            raise RuntimeError("Unexpected remote build directory")
        try:
            ssh.remote(
                args.ssh,
                ["tar", "-xz", "--no-same-owner", "-C", context],
                input=ssh.source_archive(manifest),
            )
            ssh.remote(args.ssh, ["docker", "build", "--tag", image, context])
        finally:
            ssh.remote(args.ssh, ["rm", "-rf", "--", context])
    return ssh.input_command(args.ssh, image, assets, config, action)


def modal_task(config: dict, prefix: str, action: str) -> dict:
    import modal

    from .assets import stage_assets, validate_assets

    receipt = (stage_assets if action == "prepare" else validate_assets)(
        config, Path("/assets")
    )
    if action == "prepare":
        modal.Volume.from_name(f"{prefix}-training-assets").commit()
    return receipt


def modal_inputs(args, config: dict, action: str) -> dict:
    import modal

    from .modal_app import training_image

    app = modal.App(f"{args.prefix}-training-data")
    volume = modal.Volume.from_name(
        f"{args.prefix}-training-assets",
        create_if_missing=action == "prepare",
        version=2,
    )
    if action == "check":
        volume = volume.with_mount_options(read_only=True)
    task = app.function(
        image=training_image(config),
        cpu=2,
        memory=8192,
        timeout=86400,
        volumes={"/assets": volume},
        max_containers=1,
    )(modal_task)
    try:
        with modal.enable_output(), app.run():
            return task.remote(config, args.prefix, action)
    except modal.exception.NotFoundError as error:
        raise FileNotFoundError(
            f"Training inputs volume {args.prefix}-training-assets was not found"
        ) from error


def require_inputs(args, config: dict) -> dict:
    """The same read-only input check for `check training` and training launches."""
    plan = preview(args, config)
    try:
        with redirect_stdout(sys.stderr):
            return (ssh_inputs if getattr(args, "ssh", None) else modal_inputs)(
                args, config, "check"
            )
    except (subprocess.CalledProcessError, OSError, ValueError) as error:
        detail = str(error)
        hint = (
            ""
            if "reproduce prepare training-data" in detail
            else (
                " Run reproduce prepare training-data with the same target and prefix."
            )
        )
        raise RuntimeError(
            f"Training inputs are not ready on {plan['host']}: {detail}{hint}"
        ) from error


def main(args) -> int:
    config = sources.configure(args, protocol.load(args.config, args.smoke, args.small))
    sources.use_modal(config)
    plan = preview(args, config)
    if getattr(args, "plan", False):
        print(json.dumps(plan, indent=2))
        return 0
    action = "prepare" if args.command == "prepare" else "check"
    if action == "check":
        receipt = require_inputs(args, config)
    else:
        try:
            with redirect_stdout(sys.stderr):
                receipt = (ssh_inputs if args.ssh else modal_inputs)(
                    args, config, action
                )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Training input preparation failed on {plan['host']}. Check the target's output above before retrying."
            ) from error
    if getattr(args, "output", None):
        from .execution import write_json

        write_json(args.output, {**plan, "files": receipt})
    print(
        "training: " + ("inputs prepared" if action == "prepare" else "checks passed")
    )
    print(f"  {plan['destination']}")
    print(
        f"  {len(config['dataset']['files'])} Parquet shards and tokenizer files; tokenization and packing run during training."
    )
    return 0
