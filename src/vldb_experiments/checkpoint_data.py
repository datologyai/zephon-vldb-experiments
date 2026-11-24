"""Checkpoint data preparation and input validation, outside scientific timing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

from . import ROOT


def settings():
    spec = tomllib.loads((ROOT / "configs/checkpoint/data.toml").read_text())
    data = dict(spec["data"])
    for key, unit, target in (
        ("chunk_mib", 1024**2, "chunk_bytes"),
        ("spool_mib", 1024**2, "spool_bytes"),
        ("scratch_gib", 1024**3, "scratch_bytes"),
        ("max_download_tb", 10**12, "max_download_bytes"),
        ("max_spool_tb", 10**12, "max_spool_bytes"),
        ("max_output_tb", 10**12, "max_output_bytes"),
    ):
        data[target] = int(data.pop(key) * unit)
    return data | spec["validation"]


def load_profile(path):
    from .profiles import load

    raw = load(path, "checkpoint")
    return raw["checkpoint"] | {"data_preparation": raw["checkpoint_data"]}


def generator(profile):
    script = Path(profile["arena"]).expanduser().resolve() / "scripts/prepare_data.py"
    if not script.is_file():
        raise ValueError(f"Arena data preparation script is missing: {script}")
    return script


def execute(
    profile,
    *,
    plan=False,
    yes=False,
    check=False,
    all_objects=False,
    output=None,
    abandon=False,
):
    if sum((plan, check, abandon)) > 1:
        raise ValueError("Choose one of plan, check, or abandon")
    uri = profile["dataset"]
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("Checkpoint dataset must be an s3://bucket/prefix URI")
    script = generator(profile)
    preparation = profile.get("data_preparation", {})
    source = preparation.get("source", "public")
    if source not in ("public", "existing"):
        raise ValueError("checkpoint_data.source must be public or existing")
    if not plan and not check and source == "existing":
        raise ValueError(
            "Existing export selected; use reproduce check checkpoint to validate it"
        )
    with tempfile.TemporaryDirectory(prefix="checkpoint-data-") as directory:
        root = Path(directory)
        recipe = root / "recipe.json"
        recipe.write_text(json.dumps(settings()))
        receipt = root / "receipt.json"
        # Plan imports only the standard library; other actions use the same
        # prepared CPU environment as throughput, startup and checkpointing.
        command = (
            [sys.executable, str(script)] if plan else [profile["python"], str(script)]
        )
        command += [
            "vlm",
            "--recipe",
            str(recipe),
            "--output",
            uri,
            "--receipt",
            str(receipt),
        ]
        if plan:
            command += [
                "--plan",
                "--shuffle-storage",
                preparation.get("shuffle_storage", uri + "/_shuffle"),
            ]
            for key, option in (("scratch", "--scratch"), ("journal", "--journal")):
                if preparation.get(key):
                    command += [
                        option,
                        str(Path(preparation[key]).expanduser().resolve()),
                    ]
            if preparation.get("cache_dir") or preparation.get("scratch"):
                cache = preparation.get("cache_dir") or str(
                    Path(preparation["scratch"]) / "hub"
                )
                command += ["--cache-dir", str(Path(cache).expanduser().resolve())]
        elif check:
            command += ["--check-only"]
            if source == "existing":
                command += ["--existing"]
            if all_objects:
                command += ["--all-objects"]
        else:
            scratch = preparation.get("scratch")
            if not scratch:
                raise ValueError(
                    "Set checkpoint_data.scratch for bounded preparation/resume"
                )
            journal = preparation.get("journal")
            if not journal:
                raise ValueError(
                    "Set checkpoint_data.journal on persistent storage; losing it would prevent resume"
                )
            command += [
                "--scratch",
                str(Path(scratch).expanduser().resolve()),
                "--journal",
                str(Path(journal).expanduser().resolve()),
            ]
            shuffle_storage = preparation.get("shuffle_storage", uri + "/_shuffle")
            command += ["--shuffle-storage", shuffle_storage]
            command += [
                "--cache-dir",
                str(
                    Path(preparation.get("cache_dir", str(Path(scratch) / "hub")))
                    .expanduser()
                    .resolve()
                ),
            ]
            if yes:
                command += ["--yes"]
            if abandon:
                command += ["--abandon"]
        options = {"check": True}
        if profile.get("env"):
            options["env"] = os.environ | profile["env"]
        subprocess.run(command, **options)
        result = json.loads(receipt.read_text())
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(result, indent=2) + "\n")
        return result
