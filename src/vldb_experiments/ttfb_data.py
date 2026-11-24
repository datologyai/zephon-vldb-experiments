"""Paper-specific input checks, outside the timed arena benchmark."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from . import ROOT, provenance


def block_devices(device: Path) -> list[str]:
    """Follow RAID/device-mapper slaves to their physical devices in Linux sysfs."""
    if not device.exists():
        return []
    slaves = list((device / "slaves").glob("*"))
    if slaves:
        return sorted({name for slave in slaves for name in block_devices(slave)})
    return [device.resolve().name]


def local_storage(path: Path) -> dict:
    """Describe the actual mount and NVMe backing, including NVMe RAID arrays."""
    result = subprocess.run(
        [
            "findmnt",
            "--json",
            "--target",
            str(path),
            "--output",
            "SOURCE,FSTYPE,TARGET",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    mount = json.loads(result.stdout)["filesystems"][0]
    device = path.stat().st_dev
    devices = block_devices(
        Path(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}")
    )
    models = {}
    for name in devices:
        model = Path("/sys/class/block") / re.sub(r"p\d+$", "", name) / "device/model"
        if model.exists():
            models[name] = model.read_text().strip()
    return dict(
        **mount,
        devices=devices,
        models=models,
        ebs=any("Elastic Block Store" in model for model in models.values()),
        nvme=bool(devices)
        and all(re.fullmatch(r"nvme\d+n\d+(p\d+)?", d) for d in devices),
    )


def check(profile: dict, keys: list[str], *, hashes: bool = False) -> dict:
    """Reject throughput-sized inputs, remote mounts and changed serialization.

    The paper used fully staged local NVMe. Record/chunk counts and text sizes
    match exactly; new encoded VLM payloads have an explicit size tolerance.
    This checks the experiment's inputs, not the resulting timing values.
    """
    recipe = provenance.ttfb_data_settings()
    errors, warnings, datasets = [], [], {}
    for key in keys:
        raw = str(profile["paths"][key])
        if "://" in raw:
            errors.append(f"{key}: use a staged local NVMe directory, not {raw}")
            continue
        path = Path(raw).expanduser().resolve()
        try:
            if not path.is_dir():
                raise ValueError(f"Dataset directory does not exist: {path}")
            storage = local_storage(path)
            if storage.get("ebs"):
                raise ValueError(
                    "Dataset is on network-backed EBS; stage it on instance-store NVMe for this experiment"
                )
            # Allow a documented local SSD diagnostic, never cloud/FUSE/RAM data.
            if storage["fstype"] not in ("ext4", "xfs", "btrfs", "zfs"):
                raise ValueError(
                    f"Unexpected storage {storage['fstype']}: {storage['source']}; require a local disk filesystem"
                )
            if not storage["nvme"]:
                if profile.get("ttfb_data", {}).get("require_nvme", True):
                    raise ValueError(
                        f"Cannot verify NVMe backing for {path}: {storage}; the paper used local NVMe RAID"
                    )
                warnings.append(
                    f"{key}: NVMe backing is not verified; this is a different storage setup"
                )
            data = provenance.validate_datasets(
                profile, [key], hashes=hashes, purpose="ttfb"
            )[key]
            if any((path / f["name"]).resolve().parent != path for f in data["files"]):
                raise ValueError(
                    "Chunk symlinks leave the verified local dataset directory"
                )
            expected = recipe[key]
            counts = json.loads(
                (ROOT / "configs" / expected["shard_samples"]).read_text()
            )
            actual = [f["samples"] for f in data["files"]]
            if actual != counts:
                raise ValueError(
                    f"Wrong TTFB scale/layout: {data['samples']:,} records in {len(actual):,} shards; "
                    f"need {sum(counts):,} in {len(counts):,} with the reference per-shard counts"
                )
            difference = abs(data["bytes"] / expected["shard_file_bytes"] - 1)
            if key == "text" and data["bytes"] != expected["shard_file_bytes"]:
                raise ValueError(
                    "Text record/shard byte sizes differ from the 2,048-token reference"
                )
            if difference > recipe["validation"]["maximum_size_difference"]:
                raise ValueError(
                    f"Physical data size differs by {difference:.1%}; inspect VLM payloads before measuring"
                )
            if difference > recipe["validation"]["size_warning_difference"]:
                warnings.append(
                    f"{key}: physical data size differs by {difference:.1%} from the original export"
                )
            datasets[key] = dict(
                path=str(path),
                source=provenance.data_source(profile, "ttfb"),
                records=data["samples"],
                shards=len(actual),
                bytes=data["bytes"],
                reference_bytes=expected["shard_file_bytes"],
                size_difference=difference,
                index_sha256=data["index_sha256"],
                storage=storage,
                shard_samples_sha256=provenance.file_sha256(
                    ROOT / "configs" / expected["shard_samples"]
                ),
                content_hashed=hashes,
            )
        except (
            OSError,
            ValueError,
            KeyError,
            IndexError,
            subprocess.SubprocessError,
        ) as error:
            errors.append(
                f"{key}: {error}. Use prepare ttfb-data with dedicated output paths; keep throughput data intact."
            )
    return dict(valid=not errors, datasets=datasets, errors=errors, warnings=warnings)


def require_valid(report: dict) -> None:
    for warning in report["warnings"]:
        print("TTFB input warning:", warning, flush=True)
    if report["errors"]:
        raise ValueError("TTFB input preflight failed:\n" + "\n".join(report["errors"]))
