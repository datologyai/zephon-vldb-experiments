"""Download training inputs once; tokenization and packing stay online."""

from __future__ import annotations

import json
from pathlib import Path

from .protocol import file_digest


def expected_files(config: dict) -> dict:
    return {
        f"{folder}/{Path(name).name}": sha
        for kind, folder in (("dataset", "data"), ("tokenizer", "tokenizer"))
        for name, sha in config[kind]["files"].items()
    }


def validate_assets(config: dict, assets: Path) -> dict:
    """Check existing inputs without downloading files or rebuilding the index."""
    try:
        expected = expected_files(config)
        for name, sha in expected.items():
            if file_digest(assets / name) != sha:
                raise ValueError(f"Public asset checksum mismatch: {name}")
        if json.loads((assets / "assets.json").read_text()) != expected:
            raise ValueError("Training inputs differ from the selected recipe")
        index = json.loads((assets / "data/index.json").read_text())
        names = {Path(name).name for name in config["dataset"]["files"]}
        if {shard["basename"] for shard in index["shards"]} != names:
            raise ValueError("Training index differs from the selected Parquet shards")
        if {p.name for p in (assets / "data").glob("*.parquet")} != names:
            raise ValueError(
                "Unexpected Parquet shards in the training input directory"
            )
        return expected
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Training inputs are missing or invalid: {error}. "
            "Run reproduce prepare training-data with the same target and prefix."
        ) from error


def stage_assets(config: dict, assets: Path) -> dict:
    """Fetch the same public files and build their Zephon index, or reuse them."""
    import shutil

    from huggingface_hub import hf_hub_download
    from zephon.build_index import build_index

    if (assets / "assets.json").exists() and (assets / "data/index.json").exists():
        return validate_assets(config, assets)
    assets.mkdir(parents=True, exist_ok=True)
    for kind, folder in (("dataset", "data"), ("tokenizer", "tokenizer")):
        source = config[kind]
        for name, sha in source["files"].items():
            target = assets / folder / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                downloaded = hf_hub_download(
                    source["repository"],
                    name,
                    revision=source["revision"],
                    repo_type="dataset" if kind == "dataset" else "model",
                    token=False,
                )
                temporary = target.with_suffix(target.suffix + ".tmp")
                shutil.copyfile(downloaded, temporary)
                if file_digest(temporary) != sha:
                    raise ValueError(f"Public asset checksum mismatch: {name}")
                temporary.replace(target)
            elif file_digest(target) != sha:
                raise ValueError(f"Public asset checksum mismatch: {name}")
    build_index("parquet", str(assets / "data"))
    receipt = expected_files(config)
    temporary = assets / "assets.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(assets / "assets.json")
    return validate_assets(config, assets)


def main():
    import argparse
    import sys
    from contextlib import redirect_stdout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check"))
    args = parser.parse_args()
    config = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        receipt = (stage_assets if args.action == "prepare" else validate_assets)(
            config, Path("/assets")
        )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
