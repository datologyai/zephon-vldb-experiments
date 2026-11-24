"""One machine profile for setup, preparation and all CPU experiments."""

from __future__ import annotations

import copy
import json
import os
import tomllib
from pathlib import Path

from . import ROOT


def read(path: Path) -> dict:
    path = Path(path).expanduser()
    result = (
        json.loads(path.read_text())
        if path.suffix == ".json"
        else tomllib.loads(path.read_text())
    )
    for name in ("arena", "plots", "zephon"):
        value = result.get("controller", {}).get(name)
        if value:
            directory = Path(value).expanduser()
            result["controller"][name] = str((path.parent / directory).resolve())
    return result


def load(path: Path, purpose="throughput") -> dict:
    path = Path(path).expanduser().absolute()
    raw = read(path)
    for group, names in (
        ("machine", ("root",)),
        ("paths", ("arena", "plots", "zephon", "environments", "text", "vlm")),
        ("throughput_data", ("text", "vlm")),
        ("ttfb_data", ("text", "vlm")),
        (
            "checkpoint",
            ("arena", "python", "cache", "measurement_lock", "assets_receipt"),
        ),
        ("checkpoint_data", ("scratch", "journal", "cache_dir", "shuffle_storage")),
        ("measurement", ("lock",)),
    ):
        for name in names:
            value = raw.get(group, {}).get(name)
            if value and "://" not in str(value):
                file = Path(value).expanduser()
                raw[group][name] = str(
                    (path.parent / file).resolve() if not file.is_absolute() else file
                )
    return resolve(raw, purpose)


def resolve(raw: dict, purpose="throughput") -> dict:
    result = copy.deepcopy(raw)
    machine = result.setdefault("machine", {})
    if machine.get("kind", "local") not in ("local", "ssh", "aws"):
        raise ValueError("machine.kind must be local, ssh or aws")
    if machine.get("kind", "local") != "local":
        raise ValueError("Remote profiles must be executed through the SSH target")
    root = Path(machine.get("root", "~/zephon-vldb")).expanduser().absolute()
    defaults = {
        "arena": root / "repositories/arena",
        "plots": root / "repositories/plots",
        "zephon": root / "repositories/zephon",
        "environments": root / "environments",
        "text": root / "data" / purpose / "text",
        "vlm": root / "data" / purpose / "vlm",
    }
    paths = result.setdefault("paths", {})
    result.setdefault(
        "_source_overrides",
        [name for name in ("arena", "plots", "zephon") if name in paths],
    )
    # Explicit per-experiment data paths share all other setup locations.
    data_paths = result.get(purpose + "_data", {})
    for name, default in defaults.items():
        value = (
            data_paths.get(name, paths.get(name, str(default)))
            if name in ("text", "vlm")
            else paths.get(name, str(default))
        )
        if "://" in str(value):
            raise ValueError(f"paths.{name} must be a local filesystem path")
        paths[name] = os.path.abspath(os.path.expanduser(value))
    root = Path(paths["environments"]).parent
    result.setdefault("measurement", {}).setdefault(
        "lock", str(root / ".measurement.lock")
    )
    result.setdefault("throughput_data", {}).setdefault("source", "public")
    result.setdefault("ttfb_data", {}).setdefault("source", "public")
    pins = json.loads((ROOT / "configs/checkpoint/assets.json").read_text())
    checkpoint = result.setdefault("checkpoint", {})
    result.setdefault("assets_receipt", str(root / "assets.json"))
    checkpoint.setdefault("assets_receipt", result["assets_receipt"])
    common = dict(
        arena=paths["arena"],
        zephon=paths["zephon"],
        python=str(Path(paths["environments"]) / "cpu-3.13/bin/python"),
    )
    for name, value in common.items():
        if (
            name in checkpoint
            and os.path.abspath(os.path.expanduser(checkpoint[name])) != value
        ):
            raise ValueError(
                f"checkpoint.{name} must use the shared CPU setup: {value}"
            )
        checkpoint[name] = value
    checkpoint.setdefault("cache", str(root / "cache/checkpoint"))
    checkpoint.setdefault("measurement_lock", result["measurement"]["lock"])
    checkpoint.setdefault("tokenizer_revision", pins["tokenizer"]["revision"])
    checkpoint.setdefault(
        "image_processor_revision", pins["image_processor"]["revision"]
    )
    shared_env = result.setdefault("env", {})
    shared_env.setdefault("HF_HUB_CACHE", str(root / "cache/huggingface/hub"))
    checkpoint["env"] = shared_env | checkpoint.get("env", {})
    prep = result.setdefault("checkpoint_data", {})
    prep.setdefault("source", "public")
    prep.setdefault("scratch", str(root / "cache/checkpoint-preparation"))
    # A caller must choose persistent journal storage for multi-TB preparation.
    return result


def local_target(raw: dict) -> dict:
    """Preserve target paths; never expand a remote ~ on the controller."""
    result = copy.deepcopy(raw)
    machine = result.setdefault("machine", {})
    if machine.get("kind") not in ("ssh", "aws"):
        raise ValueError("Remote machine.kind must be ssh or aws")
    root = machine.get("root")
    if not root or not str(root).startswith("/"):
        raise ValueError("Remote machine.root must be an absolute path on the target")
    for group, names in (
        ("paths", ("arena", "plots", "zephon", "environments", "text", "vlm")),
        ("checkpoint_data", ("scratch", "journal", "cache_dir")),
        (
            "checkpoint",
            ("arena", "python", "cache", "measurement_lock", "assets_receipt"),
        ),
        ("measurement", ("lock",)),
        ("throughput_data", ("text", "vlm")),
        ("ttfb_data", ("text", "vlm")),
    ):
        for name in names:
            value = result.get(group, {}).get(name)
            if value and not str(value).startswith("/"):
                raise ValueError(
                    f"Remote {group}.{name} must be absolute on the target"
                )
    machine["kind"] = "local"
    for key in ("host", "identity_file", "receipt", "user", "connect"):
        machine.pop(key, None)
    result.pop("aws", None)
    result.pop("controller", None)
    return result
