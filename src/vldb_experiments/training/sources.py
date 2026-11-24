"""Select local training sources and preserve their identity during upload."""

from __future__ import annotations

import importlib
import io
import json
import shlex
import sys
import tarfile
from pathlib import Path

from .. import provenance
from . import protocol


def configure(args, config: dict) -> dict:
    selected = dict(config.get("local_sources", {}))
    for name in ("zephon", "torchtitan", "zephon_modal"):
        path = getattr(args, name, None) or selected.get(name)
        if not path:
            continue
        if isinstance(path, dict):
            path = path["path"]
        path = Path(path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Local {name} source does not exist: {path}")
        state = provenance.source_state(path)
        selected[name] = {
            "path": str(path),
            "state": {
                key: state[key]
                for key in (
                    "revision",
                    "repository",
                    "ref",
                    "version",
                    "patch_sha256",
                )
                if key in state
            }
            | {
                "modified": bool(
                    state.get("modified_files")
                    or state.get("patch")
                    or state.get("untracked")
                )
            },
            "files_sha256": protocol.digest(provenance.source_files(path)),
        }
    if selected:
        config = config | {"local_sources": selected}
    return config


def verify(entry: dict) -> Path:
    path = Path(entry["path"])
    if protocol.digest(provenance.source_files(path)) != entry["files_sha256"]:
        raise ValueError(f"Local source changed after selection: {path}")
    return path


def use_modal(config: dict) -> None:
    entry = config.get("local_sources", {}).get("zephon_modal")
    if not entry:
        return
    path = verify(entry)
    root = path / "src" if (path / "src/zephon_modal").is_dir() else path
    module = sys.modules.get("zephon_modal")
    if module and not Path(module.__file__).resolve().is_relative_to(root):
        raise ValueError(
            "A different zephon-modal is already imported; start a new launcher process"
        )
    sys.path.insert(0, str(root))
    importlib.invalidate_caches()


def add_to_archive(archive: tarfile.TarFile, name: str, entry: dict) -> None:
    path = verify(entry)
    files = provenance.source_files(path)
    for file in files:
        archive.add(
            provenance.source_path(path, file),
            arcname=f"local_sources/{name}/{file}",
            recursive=False,
        )
    metadata = json.dumps(entry["state"] | {"files": files}).encode()
    info = tarfile.TarInfo(f"local_sources/{name}/SOURCE.json")
    info.size = len(metadata)
    archive.addfile(info, io.BytesIO(metadata))
    verify(entry)


def image_sources(config: dict):
    """Sources that must be copied into the GPU/CPU preparation image."""
    return {
        name: entry
        for name, entry in config.get("local_sources", {}).items()
        if name in ("zephon", "torchtitan")
    }


def image_directory(entry: dict):
    """Select the same source inventory for one Modal directory layer."""
    root = verify(entry)
    allowed = set()
    for name in provenance.source_files(root):
        provenance.source_path(root, name)
        file = Path(name)
        allowed.add(file)
        allowed.update(file.parents)

    def ignore(path: Path) -> bool:
        relative = path.relative_to(root) if path.is_absolute() else path
        return relative not in allowed

    return root, ignore


def image_metadata(path: Path, entry: dict) -> str:
    """Check the copied inventory before installing, then stamp the image copy."""
    script = """import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}
identity=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
if identity!=sys.argv[3]: raise ValueError('Copied training source differs from the selected inventory')
(root/'SOURCE.json').write_text(json.dumps(json.loads(sys.argv[2]) | {'files':files}))
"""
    return shlex.join(
        [
            "python",
            "-c",
            script,
            str(path),
            json.dumps(entry["state"]),
            entry["files_sha256"],
        ]
    )


def verify_installed(config: dict, root: Path = Path("/opt/local_sources")) -> dict:
    """Validate uploaded sources independently of Git or package version labels."""
    result = {}
    for name, entry in image_sources(config).items():
        path = root / name
        if protocol.digest(provenance.source_files(path)) != entry["files_sha256"]:
            raise ValueError(
                f"Training image does not contain the selected {name} source"
            )
        result[name] = entry["files_sha256"]
    return result


def validate_observed(observed: dict, config: dict) -> None:
    expected = {
        name: entry["files_sha256"] for name, entry in image_sources(config).items()
    }
    if expected and observed.get("local_sources") != expected:
        raise ValueError(
            "Observed training sources differ from the selected local sources"
        )
    if "zephon" not in expected and observed["zephon"] != config["sources"]["zephon"]:
        raise ValueError("Observed Zephon version differs from the training recipe")
    if (
        "torchtitan" not in expected
        and observed["torchtitan_commit"] != config["sources"]["torchtitan"]
    ):
        raise ValueError(
            "Observed TorchTitan revision differs from the training recipe"
        )
