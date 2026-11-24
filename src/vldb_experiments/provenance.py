"""Identify the inputs of a measurement: sources, environments, datasets and host.

A campaign stores these states in its manifest. Resuming compares them again, so
measurements made with different code, packages or data are never pooled.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import tomllib
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from . import ROOT

# Runs inside a benchmark interpreter and prints what is installed there.
ENVIRONMENT_PROBE = """
import importlib.metadata
import json
import sys
import sysconfig

distributions = list(importlib.metadata.distributions())
direct_urls = {}
for distribution in distributions:
    # Present for packages installed from a URL or a local (editable) directory.
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        direct_urls[distribution.metadata["Name"]] = json.loads(direct_url)
print(
    json.dumps(
        {
            "python": sys.version,
            "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            "executable": sys.executable,
            "packages": dict(
                sorted((d.metadata["Name"], d.version) for d in distributions)
            ),
            "direct_urls": direct_urls,
        }
    )
)
"""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_state(path: Path) -> dict[str, Any]:
    """Identify a Git checkout or an exported snapshot, including local changes."""
    if (path / ".git").exists():
        return git_checkout_state(path)
    if (path / "SOURCE.json").exists():
        return snapshot_state(path)
    if not path.is_dir():
        raise ValueError(f"Source directory does not exist: {path}")
    return {"revision": None, "files": source_files(path)}


def source_files(path: Path) -> dict[str, str]:
    """Identify source files without requiring Git or including build products."""
    ignored = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "build",
        "dist",
        ".DS_Store",
    }
    generated = set()
    project = path / "pyproject.toml"
    if project.is_file():
        scm = (
            tomllib.loads(project.read_text()).get("tool", {}).get("setuptools_scm", {})
        )
        generated = {scm[key] for key in ("version_file", "write_to") if key in scm}
    git = (path / ".git").exists()
    metadata = path / "SOURCE.json"
    exported = (
        set(json.loads(metadata.read_text()).get("files", {}))
        if not git and metadata.is_file()
        else set()
    )
    launcher = (path / "src/vldb_experiments").is_dir()
    tracked = set()
    if git:
        entries = (
            subprocess.check_output(
                ["git", "-C", str(path), "ls-files", "--stage", "-z"]
            )
            .decode()
            .split("\0")
        )
        for entry in filter(None, entries):
            mode, name = entry.split("\t", 1)
            if mode.startswith("160000 "):
                raise ValueError(
                    f"Export submodule sources explicitly before use: {path / name}"
                )
            tracked.add(name)
        others = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(path),
                    "ls-files",
                    "-z",
                    "--others",
                    "--exclude-standard",
                ]
            )
            .decode()
            .split("\0")
        )
        files = [path / name for name in tracked | set(filter(None, others))]
    else:
        files = path.rglob("*")
    result = {}
    for file in sorted(set(files)):
        relative = file.relative_to(path)
        name = relative.as_posix()
        if name in generated or name == "SOURCE.json" or file.name == ".DS_Store":
            continue
        # Keep tracked/exported files, but leave generated products out of new files.
        if (
            name not in tracked
            and name not in exported
            and (
                file.suffix in (".pyc", ".pyo")
                or any(
                    part in ignored or part.endswith(".egg-info")
                    for part in relative.parts
                )
                or (
                    launcher
                    and relative.parts[0]
                    in ("runs", "plots", "data", ".cache", ".jobs")
                )
                or (
                    launcher
                    and relative.parts[0] == "profiles"
                    and ".example." not in file.name
                )
            )
        ):
            continue
        if file.is_symlink() and not file.exists():
            raise ValueError(f"Dangling source symlink: {file}")
        if file.is_symlink() and file.is_dir():
            raise ValueError(f"Source directory symlinks are unsupported: {file}")
        if file.is_file():
            result[name] = file_sha256(file)
    return result


def source_path(root: Path, name: str) -> Path:
    """Materialize internal file symlinks without copying files outside a repo."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(
            f"Source file must remain inside its repository: {root / name}"
        )
    return path


def git_checkout_state(path: Path) -> dict[str, Any]:
    """The checked-out commit, the uncommitted patch and any untracked files."""

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    patch = git("diff", "HEAD", "--binary")
    untracked = (
        subprocess.check_output(
            ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "-z"]
        )
        .decode()
        .split("\0")
    )
    inventory = source_files(path)
    return {
        "revision": git("rev-parse", "HEAD"),
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "patch": patch,
        "untracked": {name: inventory[name] for name in untracked if name in inventory},
    }


def snapshot_state(path: Path) -> dict[str, Any]:
    """Record an exported revision and its actual contents, including edits."""
    state = json.loads((path / "SOURCE.json").read_text())
    if not state.get("files"):
        raise ValueError(
            f"Source snapshot has no file inventory: {path / 'SOURCE.json'}"
        )
    actual = source_files(path)
    changes = sorted(
        name
        for name in state["files"].keys() | actual.keys()
        if state["files"].get(name) != actual.get(name)
    )
    return state | {
        "files": actual,
        "modified_files": sorted(set(state.get("modified_files", [])) | set(changes)),
    }


def environment_states(interpreters: Iterable[Path]) -> dict[str, Any]:
    """Installed packages and editable sources of each benchmark interpreter."""
    result = {}
    for python in sorted(set(interpreters)):
        # -I keeps the launcher's cwd and PYTHONPATH out of package discovery.
        # A source-tree .egg-info is controller metadata, not a benchmark dependency.
        state = json.loads(
            subprocess.check_output(
                [str(python), "-I", "-c", ENVIRONMENT_PROBE], text=True
            )
        )
        # Editable installs run code from a directory; record that directory's state.
        sources = {}
        for name, direct in state["direct_urls"].items():
            if direct.get("dir_info", {}).get("editable"):
                url = urlparse(direct["url"])
                if url.scheme == "file":
                    sources[name] = source_state(Path(unquote(url.path)))
        state["editable_sources"] = sources
        result[str(python)] = state
    return result


def dataset_manifest(
    path: Path,
    content_hashes: bool = False,
    *,
    sample_bytes: int | None = None,
) -> dict[str, Any]:
    """Check a LitData export's chunk files against its index and summarize them."""
    raw_index = (path / "index.json").read_bytes()
    index = json.loads(raw_index)
    if index.get("config", {}).get("compression"):
        raise ValueError(
            "This campaign requires the uncompressed paper LitData exports"
        )
    files = []
    for chunk in index["chunks"]:
        name = chunk["filename"]
        target = path / name
        if target.parent.resolve() != path.resolve():
            raise ValueError(f"Unexpected chunk path: {name}")
        size = target.stat().st_size
        samples = chunk["chunk_size"]
        # A chunk begins with uint32 values: its sample count, then the byte
        # offset where each sample starts and where the last one ends.
        header_bytes = 4 * (samples + 2)
        expected_bytes = chunk["chunk_bytes"]
        if size != expected_bytes:
            raise ValueError(
                f"Wrong byte count for {target}: {size} != {expected_bytes}"
            )
        with target.open("rb") as handle:
            header = struct.unpack(f"<{samples + 2}I", handle.read(header_bytes))
        count, *offsets = header
        if (
            count != samples
            or offsets[0] != header_bytes
            or offsets[-1] != size
            or any(a > b for a, b in pairwise(offsets))
        ):
            raise ValueError(f"Invalid LitData chunk offset table: {target}")
        if sample_bytes is not None and any(
            b - a != sample_bytes for a, b in pairwise(offsets)
        ):
            raise ValueError(f"Wrong serialized text sequence length: {target}")
        item = {"name": name, "bytes": size, "samples": samples}
        if content_hashes:
            with target.open("rb") as handle:
                item["sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        files.append(item)
    return {
        "root": str(path),
        "index_sha256": hashlib.sha256(raw_index).hexdigest(),
        "files": files,
        "samples": sum(f["samples"] for f in files),
        "bytes": sum(f["bytes"] for f in files),
        "content_hashed": content_hashes,
    }


def throughput_data_settings() -> dict[str, Any]:
    return tomllib.loads((ROOT / "configs/throughput/data.toml").read_text())


def ttfb_data_settings() -> dict[str, Any]:
    """Startup uses the public sources with the original export's shard shape."""
    base = throughput_data_settings()
    overrides = tomllib.loads((ROOT / "configs/ttfb/data.toml").read_text())
    return base | {key: base.get(key, {}) | value for key, value in overrides.items()}


def data_source(profile: dict[str, Any], purpose: str) -> str:
    # Throughput and TTFB use the checked-in public preparation recipes.
    source = profile.get(purpose + "_data", {}).get("source", "public")
    if source != "public":
        raise ValueError(f"{purpose}_data.source must be 'public'")
    return source


def validate_public_dataset(
    path: Path, key: str, hashes: bool = False, *, recipe: dict | None = None
) -> dict[str, Any]:
    """Verify the selected public recipe, serialization, record counts and hashes."""
    pin = (recipe or throughput_data_settings())[key]
    metadata_path = path / "manifest.json"
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "repo": pin["repository"],
        "revision": pin["revision"],
        "subset": pin["subset"],
    }
    if (
        metadata.get("format_version") != 1
        or metadata.get("mode") != key
        or metadata.get("source") != expected
    ):
        raise ValueError(f"Public {key} data has different source provenance")
    if (
        metadata.get("seed") != pin["seed"]
        or metadata.get("chunk_bytes") != pin["chunk_mib"] * 1024**2
    ):
        raise ValueError(f"Public {key} data uses different generation settings")
    if "shard_order" in pin and metadata.get("shard_order") != pin["shard_order"]:
        raise ValueError(
            f"Public {key} data uses a different sampling or record-order recipe; "
            "keep this export and prepare the configured recipe at a new path"
        )
    if key == "text":
        tokenizer = metadata.get("tokenizer") or {}
        if (
            metadata.get("schema") != "uint32[sequence_length]"
            or metadata.get("sequence_length") != pin["sequence_length"]
            or tokenizer.get("repo") != pin["tokenizer"]
            or tokenizer.get("revision") != pin["tokenizer_revision"]
            or tokenizer.get("add_special_tokens") is not False
        ):
            raise ValueError(
                "Public text data has different sequence or tokenizer settings"
            )
    elif (
        metadata.get("schema") != "encoded_images_conversation_metadata"
        or metadata.get("vlm_loader_settings", {}).get("decoded_image_input")
        is not False
    ):
        raise ValueError(
            "Public VLM data must contain encoded images and conversations"
        )
    index = json.loads((path / "index.json").read_text())
    expected_format = (
        ["no_header_numpy:6"]
        if key == "text"
        else ["bytes", "str", "str", "str", "int"]
    )
    if index.get("config", {}).get("data_format") != expected_format:
        raise ValueError(f"Public {key} input has a different serialization schema")
    result = dataset_manifest(
        path,
        hashes,
        sample_bytes=4 + 4 * pin["sequence_length"] if key == "text" else None,
    )
    if metadata.get("samples") != result["samples"] or metadata.get("chunks") != len(
        result["files"]
    ):
        raise ValueError(f"Public {key} manifest and chunk index have different counts")
    if (
        result["samples"] < pin["samples"]
        or len(result["files"]) < pin["minimum_chunks"]
    ):
        raise ValueError(
            f"Public {key} data is too small: {result['samples']:,} records in {len(result['files'])} shards; "
            f"need at least {pin['samples']:,} records and {pin['minimum_chunks']} shards"
        )
    if "shard_samples" in pin:
        counts = json.loads((ROOT / "configs" / pin["shard_samples"]).read_text())
        if [f["samples"] for f in result["files"]] != counts or metadata.get(
            "shard_samples"
        ) != counts:
            raise ValueError(
                f"Public {key} input has the wrong TTFB shard layout; "
                "use prepare ttfb-data with new output paths"
            )
    checksums = metadata.get("sha256", {})
    if checksums.get("index.json") != result["index_sha256"]:
        raise ValueError(f"Public {key} index does not match its generation manifest")
    for item in result["files"]:
        expected_hash = checksums.get(item["name"])
        if not expected_hash or (hashes and item["sha256"] != expected_hash):
            raise ValueError(
                f"Public {key} chunk differs from its generation manifest: {item['name']}"
            )
    if hashes:
        chunk_names = {item["name"] for item in result["files"]}
        for name, expected_hash in checksums.items():
            if Path(name).name != name:
                raise ValueError(f"Unexpected generated file path: {name}")
            if name not in chunk_names and file_sha256(path / name) != expected_hash:
                raise ValueError(f"Public {key} generated file changed: {name}")
    result.update(
        source=expected,
        generation_manifest=metadata,
        generation_manifest_sha256=file_sha256(metadata_path),
    )
    return result


def validate_datasets(
    profile: dict[str, Any],
    keys: list[str],
    hashes: bool = False,
    *,
    purpose: str = "throughput",
) -> dict[str, Any]:
    """Check prepared public inputs against their recipe and generation manifest."""
    data_source(profile, purpose)
    recipe = ttfb_data_settings() if purpose == "ttfb" else None
    return {
        key: validate_public_dataset(
            Path(profile["paths"][key]), key, hashes, recipe=recipe
        )
        for key in keys
    }


def host_snapshot() -> dict[str, Any]:
    """Describe the measurement host; recorded once when a campaign starts."""
    result = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    for name in (
        "/proc/cmdline",
        "/proc/meminfo",
        "/proc/sys/kernel/randomize_va_space",
        "/proc/sys/kernel/nmi_watchdog",
        "/sys/devices/system/cpu/smt/control",
    ):
        path = Path(name)
        if path.exists():
            result[name] = path.read_text().strip()
    for command in (
        ("lscpu",),
        ("lsblk", "-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS,MODEL"),
        ("df", "-h"),
    ):
        if shutil.which(command[0]):
            result[command[0]] = subprocess.check_output(command, text=True)
    return result
