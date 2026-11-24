"""Prepare pinned dependencies, datasets and tokenizer assets for arena."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

from . import ROOT, core, matrix, provenance

CPU_PYTHONS = {"cpu-3.13": "3.13.15", "cpu-3.14t": "3.14.7t"}


def export_source(source: dict[str, str], destination: Path) -> None:
    """Use an existing source directory, or download the pinned revision."""
    revision = source["revision"]
    output = destination / source["directory"]
    if output.exists():
        provenance.source_state(output)
        print(f"Using existing source {output}", flush=True)
        return
    request = urllib.request.Request(
        f"https://api.github.com/repos/{source['repository']}/tarball/{revision}",
        headers={"User-Agent": "zephon-vldb-experiments"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            archive = response.read()
    except urllib.error.HTTPError as error:
        if error.code not in (403, 404):
            raise
        # Private source access uses the user's existing GitHub CLI login.
        # Credentials stay in gh; no token is copied into commands or logs.
        try:
            archive = subprocess.check_output(
                ["gh", "api", f"repos/{source['repository']}/tarball/{revision}"],
                stderr=subprocess.PIPE,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as auth_error:
            raise RuntimeError(
                f"Cannot fetch {source['repository']} at {revision}. "
                "For private repositories, install gh and run gh auth login, "
                "or supply an existing source directory."
            ) from auth_error
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{source['directory']}-", dir=destination)
    )
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            members = tar.getmembers()
            stripped = []
            for member in members:
                parts = PurePosixPath(member.name).parts[1:]
                if not parts:
                    continue
                member.name = str(PurePosixPath(*parts))
                stripped.append(member)
            members = stripped
            tar.extractall(temporary, members=members, filter="data")
        (temporary / "SOURCE.json").write_text(
            json.dumps(
                {
                    **source,
                    "ref": revision,
                    "archive_sha256": hashlib.sha256(archive).hexdigest(),
                    "files": {
                        str(p.relative_to(temporary)): hashlib.sha256(
                            p.read_bytes()
                        ).hexdigest()
                        for p in sorted(temporary.rglob("*"))
                        if p.is_file()
                    },
                },
                indent=2,
            )
            + "\n"
        )
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Exported {output} at {revision}", flush=True)


def sources(profile: dict, revision: str) -> None:
    """Use selected directories; download pinned sources only when absent."""
    pins = tomllib.loads((ROOT / "configs/shared/sources.toml").read_text())
    for name in profile.get("_source_overrides", []):
        if not Path(profile["paths"][name]).is_dir():
            raise ValueError(
                f"Supplied {name} source does not exist: {profile['paths'][name]}"
            )
    for name in ("arena", "plots"):
        output = Path(profile["paths"][name])
        export_source(pins[name] | {"directory": output.name}, output.parent)
    output = Path(profile["paths"]["zephon"])
    export_source(
        {
            "repository": "datologyai/zephon",
            "revision": matrix.revision_pin(revision)["commit"],
            "directory": output.name,
            "version": revision.removeprefix("v"),
        },
        output.parent,
    )


def controller_plots(profile: dict) -> Path:
    """Local plotting location; independent of paths on the benchmark target."""
    return (
        Path(
            profile.get("controller", {}).get(
                "plots", ROOT.parent / "zephon-vldb-plots"
            )
        )
        .expanduser()
        .absolute()
    )


def plotting(path: Path, *, python="3.13", existing=False) -> Path:
    """Fetch the pinned renderer if absent and prepare its locked environment."""
    path = path.expanduser().absolute()
    if existing and not path.is_dir():
        raise ValueError(f"Supplied plotting source does not exist: {path}")
    pin = tomllib.loads((ROOT / "configs/shared/sources.toml").read_text())["plots"]
    export_source(pin | {"directory": path.name}, path.parent)
    call(
        [
            "uv",
            "sync",
            "--project",
            ROOT / "environments/plotting",
            "--python",
            python,
            "--locked",
        ],
        env=os.environ | {"UV_PROJECT_ENVIRONMENT": str(path / ".venv")},
    )
    from .report import plotting_python

    plotting_python(path)
    return path


def call(command, **kwargs) -> None:
    print(" ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def pip_install(python: Path, *arguments, scm_version: dict | None = None) -> None:
    """Install into one benchmark environment.

    Exported snapshots carry no Git tags, so ``scm_version`` sets the variable
    that tells setuptools-scm which version to build.
    """
    env = (os.environ | scm_version) if scm_version else None
    call(["uv", "pip", "install", "--python", python, *arguments], env=env)


def environments(profile: dict, revision: str, only: list[str]) -> None:
    """One locked CPU dependency set, with the paper's two interpreter builds."""
    environments_root = Path(profile["paths"]["environments"])
    arena = Path(profile["paths"]["arena"])
    provenance.source_state(arena)
    specifications = CPU_PYTHONS
    if set(only) - specifications.keys():
        raise ValueError(f"Unknown environment; choose from {list(specifications)}")
    source = Path(profile["paths"]["zephon"])
    provenance.source_state(source)
    for name, version in specifications.items():
        if only and name not in only:
            continue
        environment = environments_root / name
        python = environment / "bin/python"
        command = [
            "uv",
            "sync",
            "--project",
            str(arena),
            "--locked",
            "--python",
            version,
            "--no-install-package",
            "zephon",
        ]
        for extra in (
            "zephon",
            "litdata",
            "grain",
            "vlm-pipeline",
            "sweep",
            "checkpoint",
        ) + (("data-preparation",) if name == "cpu-3.13" else ()):
            command += ["--extra", extra]
        call(command, env=os.environ | {"UV_PROJECT_ENVIRONMENT": str(environment)})
        pip_install(
            python,
            "--no-deps",
            "-e",
            source,
            scm_version=source_build_environment(source, "zephon"),
        )
        frozen = subprocess.check_output(
            ["uv", "pip", "freeze", "--python", str(python)], text=True
        )
        (environment / "requirements-resolved.txt").write_text(frozen)


def throughput_data(profile: dict) -> None:
    """Generate the paper's public throughput inputs."""
    provenance.data_source(profile, "throughput")
    prepare_public_throughput_data(profile)


def prepare_public_throughput_data(profile: dict) -> None:
    """Use arena's generator in the shared CPU environment with the paper settings."""
    prepare_public_data(profile, provenance.throughput_data_settings(), "throughput")


def ttfb_data(profile: dict) -> None:
    """Generate public startup inputs with the reference record/shard counts."""
    provenance.data_source(profile, "ttfb")
    prepare_public_data(profile, provenance.ttfb_data_settings(), "ttfb")


def prepare_public_data(profile: dict, settings: dict, purpose: str) -> None:
    missing = []
    for key in ("text", "vlm"):
        output = Path(profile["paths"][key])
        if output.exists():
            if not (output / "manifest.json").is_file():
                raise ValueError(
                    f"Existing directory has no public generation manifest: {output}; choose a new output path"
                )
            print(
                f"Verifying existing {key} {purpose} inputs in {output} (reads every chunk)",
                flush=True,
            )
            provenance.validate_public_dataset(
                output, key, hashes=True, recipe=settings
            )
            print(f"Already prepared {output}", flush=True)
        else:
            missing.append(key)
    if not missing:
        return
    generator = Path(profile["paths"]["arena"]) / "scripts/prepare_data.py"
    if not generator.is_file():
        raise ValueError(f"Arena data preparation script is missing: {generator}")
    cache = Path(profile["paths"]["environments"]).parent / "throughput-data-cache"
    for key in missing:
        output = Path(profile["paths"][key])
        pin = settings[key]
        command = [
            core.environment_path(profile, {"python": "3.13"}, "v0.1.0"),
            generator,
            key,
            "--output",
            output,
            "--cache-dir",
            cache,
        ]
        for option in ("samples", "chunk_mib", "max_download_mib", "seed", "revision"):
            command += ["--" + option.replace("_", "-"), str(pin[option])]
        if "shard_samples" in pin:
            command += ["--shard-samples", ROOT / "configs" / pin["shard_samples"]]
        if key == "text":
            for option in ("sequence_length", "tokenizer", "tokenizer_revision"):
                command += ["--" + option.replace("_", "-"), str(pin[option])]
        call(command)
        provenance.validate_public_dataset(output, key, hashes=True, recipe=settings)


def assets(profile: dict, revision: str) -> None:
    """Prepare throughput and checkpoint tokenizer/processor assets; no weights."""
    assets = [core.read_json(ROOT / "configs/shared/vlm-tokenizer.json")]
    assets += list(core.read_json(ROOT / "configs/checkpoint/assets.json").values())
    python = core.environment_path(
        profile, {"system": "zephon", "python": "3.13"}, revision
    )
    code = """import hashlib, json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
assets = json.loads(sys.argv[1])
receipt = []
for asset in assets:
    patterns = list(asset.get('files', asset.get('patterns', [])))
    path = Path(snapshot_download(asset['model'], revision=asset['revision'], allow_patterns=patterns))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file() and p.name in patterns}
    for name, expected in asset.get('files', {}).items():
        if files.get(name) != expected:
            raise ValueError('Asset mismatch: ' + name)
    if not files:
        raise ValueError('No preprocessing assets downloaded')
    receipt.append(dict(model=asset['model'], revision=asset['revision'], files=files))
Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2) + '\\n')
print('Prepared tokenizer and image-processor assets')
"""
    receipt = Path(profile["paths"]["environments"]).parent / "assets.json"
    call(
        [python, "-c", code, json.dumps(assets), str(receipt)],
        env=os.environ | profile.get("env", {}),
    )


def plan(profile: dict, component: str) -> dict:
    """Resource preview; no installation, source lookup or cloud requests."""
    if component == "checkpoint-data":
        from . import checkpoint_data

        return dict(
            component=component,
            dataset=profile["checkpoint"].get("dataset"),
            records=checkpoint_data.settings()["samples"],
            final_storage_estimate="7–10 TB",
            temporary_storage_estimate="4.8 TB",
            peak_s3_estimate="12–15 TB",
            local_scratch_cap_gib=128,
            estimates="Full-scale storage/traffic remain estimates.",
        )
    if component in ("throughput-data", "ttfb-data"):
        spec = (
            provenance.ttfb_data_settings()
            if component == "ttfb-data"
            else provenance.throughput_data_settings()
        )
        return dict(
            component=component,
            paths={k: profile["paths"][k] for k in ("text", "vlm")},
            records={k: spec[k]["samples"] for k in ("text", "vlm")},
            free_disk_estimate="300 GiB plus headroom"
            if component == "ttfb-data"
            else "275 GiB plus headroom",
        )
    return dict(
        component=component,
        paths=profile["paths"],
        environments=["cpu-3.13", "cpu-3.14t"],
        default_zephon_revision=matrix.revision_pin("v0.1.0")["commit"],
        litdata="0.2.61",
    )


def verify_assets(profile: dict) -> dict:
    """Check cached files against the setup receipt without network access."""
    path = Path(profile["assets_receipt"])
    assets = core.read_json(path)
    tokenizer = core.read_json(ROOT / "configs/shared/vlm-tokenizer.json")
    expected = [
        tokenizer,
        *core.read_json(ROOT / "configs/checkpoint/assets.json").values(),
    ]
    if {(a["model"], a["revision"]) for a in assets} != {
        (a["model"], a["revision"]) for a in expected
    }:
        raise ValueError(
            "Preprocessing asset receipt does not cover the shared CPU setup"
        )
    for spec in expected:
        files = next(a["files"] for a in assets if a["model"] == spec["model"])
        if not files or any(
            files.get(name) != digest for name, digest in spec.get("files", {}).items()
        ):
            raise ValueError(
                "Preprocessing asset receipt differs from the pinned configuration"
            )
    cache = Path(profile["env"]["HF_HUB_CACHE"])
    for asset in assets:
        snapshot = (
            cache
            / ("models--" + asset["model"].replace("/", "--"))
            / "snapshots"
            / asset["revision"]
        )
        for name, expected in asset["files"].items():
            file = snapshot / name
            if not file.is_file() or provenance.file_sha256(file) != expected:
                raise ValueError(
                    f"Prepared preprocessing asset changed or missing: {file}"
                )
    return assets


def check_environments(profile: dict, interpreters=None) -> dict:
    """Validate the pinned CPU setup without downloading or changing packages."""
    root = Path(profile["paths"]["environments"])
    states = provenance.environment_states(
        interpreters or [root / name / "bin/python" for name in CPU_PYTHONS]
    )
    validate_environments(states, profile)
    for name in ("arena", "plots"):
        provenance.source_state(Path(profile["paths"][name]))
    return states


def validate_environments(states: dict, profile: dict | None = None) -> None:
    for executable, state in states.items():
        name = Path(executable).parent.parent.name
        expected = CPU_PYTHONS[name].removesuffix("t")
        if state.get("free_threaded") != name.endswith("t"):
            raise ValueError(
                f"Wrong GIL build in {executable}; run machine setup in a new root"
            )
        if state["python"].split()[0] != expected:
            raise ValueError(
                f"Expected Python {expected} in {executable}; run machine setup in a new root"
            )
        packages = {
            k.lower().replace("_", "-"): v for k, v in state["packages"].items()
        }
        for package, version in (("litdata", "0.2.61"),):
            if packages.get(package) != version:
                raise ValueError(f"Expected {package}=={version} in {executable}")
        for package in (
            "torch",
            "transformers",
            "pillow",
            "obstore",
            "psutil",
            "pandas",
            "zephon",
        ) + (
            (
                "grain",
                "pyarrow",
                "ijson",
                "huggingface-hub",
                "tokenizers",
                "boto3",
                "zstd",
            )
            if name == "cpu-3.13"
            else ()
        ):
            if package not in packages:
                raise ValueError(
                    f"Missing shared CPU package {package} in {executable}"
                )
        if profile:
            from urllib.parse import unquote, urlparse

            direct = {
                k.lower().replace("_", "-"): v
                for k, v in state.get("direct_urls", {}).items()
            }
            for package in ("zephon", "dataloader-arena"):
                source_name = "arena" if package == "dataloader-arena" else package
                selected = Path(profile["paths"][source_name]).resolve()
                installed = direct.get(package, {}).get("url", "")
                if (
                    not installed.startswith("file:")
                    or Path(unquote(urlparse(installed).path)).resolve() != selected
                ):
                    raise ValueError(
                        f"Installed {package} does not use {selected}; run machine setup"
                    )


def source_build_environment(path: Path, package: str) -> dict:
    """SCM builds use Git when present, or archive metadata when exported."""
    if (path / ".git").exists():
        return {}
    metadata = path / "SOURCE.json"
    state = provenance.source_state(path) if metadata.exists() else {}
    version = (
        state.get("version")
        if not (state.get("modified_files") or state.get("modified"))
        else None
    )
    return {
        f"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_{package.upper().replace('-', '_')}": version
        or "0+unknown"
    }
