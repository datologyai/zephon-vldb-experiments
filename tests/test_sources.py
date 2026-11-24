"""Local checkouts and extracted software archives are authoritative inputs."""

import io
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vldb_experiments import cli, core, prepare, profiles, provenance, remote
from vldb_experiments.training import protocol, sources, ssh


def source(path, name="module.py", content="value = 1\n"):
    path.mkdir(parents=True)
    (path / name).write_text(content)
    return path


def test_plain_archive_tracks_edits_and_ignores_runtime_and_scm_products(tmp_path):
    path = source(tmp_path / "export")
    (path / "src/vldb_experiments").mkdir(parents=True)
    (path / "pyproject.toml").write_text(
        '[tool.setuptools_scm]\nversion_file="_version.py"\n'
    )
    before = provenance.source_state(path)
    assert before["revision"] is None
    for name in (".venv", "runs", "__pycache__", "package.egg-info"):
        source(path / name)
    (path / ".DS_Store").write_bytes(b"desktop metadata")
    (path / "_version.py").write_text("generated = True")
    assert provenance.source_state(path) == before
    (path / "module.py").write_text("value = 2\n")
    assert provenance.source_state(path) != before


def test_exported_revision_preserves_local_edits_without_rewriting_metadata(tmp_path):
    path = source(tmp_path / "export")
    stamp = {"revision": "custom", "files": provenance.source_files(path)}
    (path / "SOURCE.json").write_text(json.dumps(stamp))
    (path / "module.py").write_text("value = 2\n")
    state = provenance.source_state(path)
    assert state["revision"] == "custom"
    assert state["modified_files"] == ["module.py"]
    assert json.loads((path / "SOURCE.json").read_text()) == stamp


def test_custom_git_checkout_keeps_dirty_and_untracked_sources(tmp_path):
    path = source(tmp_path / "git repo")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True)

    git("init", "-q")
    source(path / "data", "schema.py")
    source(path / "build", "builder.py")
    git("add", ".")
    git(
        "-c",
        "commit.gpgsign=false",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-qm",
        "fixture",
    )
    (path / "module.py").write_text("changed = True\n")
    (path / "extra.py").write_text("extra = True\n")
    (path / "naïve helper.py").write_text("named = True\n")
    (path / " leading.py").write_text("spaced = True\n")
    before = provenance.source_state(path)
    assert {"naïve helper.py", " leading.py"} <= before["untracked"].keys()
    source(path / "__pycache__", "module.cpython-313.pyc", "generated")
    source(path / "package.egg-info", "PKG-INFO", "generated")
    assert provenance.source_state(path) == before
    assert {"data/schema.py", "build/builder.py"} <= provenance.source_files(
        path
    ).keys()
    with patch.object(
        prepare.urllib.request, "urlopen", side_effect=AssertionError("download")
    ):
        prepare.export_source(
            {"revision": "different", "repository": "unused", "directory": path.name},
            path.parent,
        )
    assert provenance.source_state(path) == before
    assert before["patch"] and "extra.py" in before["untracked"]
    _, data = remote.repository_archive(path)
    target = tmp_path / "uploaded"
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(target, filter="data")
    assert provenance.source_files(target) == provenance.source_files(path)
    # Modal copies the selected directory before generating its source stamp.
    selected = sources.configure(SimpleNamespace(zephon=path), protocol.load())
    entry = selected["local_sources"]["zephon"]
    (target / "SOURCE.json").unlink()
    command = shlex.split(sources.image_metadata(target, entry))
    command[0] = sys.executable
    subprocess.run(command, check=True)
    assert protocol.digest(provenance.source_files(target)) == entry["files_sha256"]
    git(
        "update-index",
        "--add",
        "--cacheinfo",
        "160000," + git("rev-parse", "HEAD").strip() + ",submodule",
    )
    with pytest.raises(ValueError, match="submodule"):
        provenance.source_files(path)


def test_source_upload_materializes_internal_links_and_rejects_external_files(tmp_path):
    path = source(tmp_path / "repo")
    (path / "alias.py").symlink_to("module.py")
    _, data = remote.repository_archive(path)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        assert archive.getmember("alias.py").isfile()
        assert (
            archive.extractfile("alias.py").read() == (path / "module.py").read_bytes()
        )
    outside = tmp_path / "outside.py"
    outside.write_text("external = True")
    (path / "external.py").symlink_to(outside)
    with pytest.raises(ValueError, match="inside its repository"):
        remote.repository_archive(path)


def test_uploaded_snapshot_keeps_modified_revision_label(tmp_path):
    path = source(tmp_path / "snapshot")
    (path / "SOURCE.json").write_text(
        json.dumps(
            {
                "revision": "original",
                "version": "0.1.0",
                "files": provenance.source_files(path),
            }
        )
    )
    (path / "module.py").write_text("changed = True\n")
    _, data = remote.repository_archive(path)
    target = tmp_path / "uploaded"
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(target, filter="data")
    state = provenance.source_state(target)
    assert state["revision"] == "original"
    assert state["modified_files"] == ["module.py"]
    assert (
        prepare.source_build_environment(target, "zephon")[
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ZEPHON"
        ]
        == "0+unknown"
    )


def test_dangling_source_link_is_rejected(tmp_path):
    path = source(tmp_path / "repo")
    (path / "missing.py").symlink_to("absent.py")
    with pytest.raises(ValueError, match="Dangling source"):
        provenance.source_files(path)


def test_relative_archive_profile_uses_all_three_directories_without_network(tmp_path):
    for name in ("arena", "plots", "zephon"):
        source(tmp_path / name)
    directory = tmp_path / "launcher/profiles"
    directory.mkdir(parents=True)
    profile = directory / "archive.toml"
    profile.write_text(
        '[paths]\narena="../../arena"\nplots="../../plots"\nzephon="../../zephon"\n'
    )
    values = profiles.load(profile)
    before = {
        name: provenance.source_state(tmp_path / name)
        for name in ("arena", "plots", "zephon")
    }
    with patch.object(
        prepare.urllib.request, "urlopen", side_effect=AssertionError("download")
    ):
        prepare.sources(values, "v0.1.0")
    assert {name: provenance.source_state(tmp_path / name) for name in before} == before
    shutil.rmtree(tmp_path / "zephon")
    with (
        patch.object(
            prepare.urllib.request, "urlopen", side_effect=AssertionError("download")
        ),
        pytest.raises(ValueError, match="Supplied zephon"),
    ):
        prepare.sources(values, "v0.1.0")


def test_installed_sources_must_match_selection_not_the_default_revision(tmp_path):
    packages = {
        name: "present"
        for name in (
            "torch",
            "transformers",
            "pillow",
            "obstore",
            "psutil",
            "pandas",
            "grain",
            "pyarrow",
            "ijson",
            "huggingface-hub",
            "tokenizers",
            "boto3",
            "zstd",
        )
    }
    packages.update(zephon="9.9.9", litdata="0.2.61")
    profile = profiles.resolve({"machine": {"root": str(tmp_path)}})
    state = {
        "python": "3.13.15",
        "free_threaded": False,
        "packages": packages,
        "direct_urls": {
            "zephon": {"url": Path(profile["paths"]["zephon"]).as_uri()},
            "dataloader-arena": {"url": Path(profile["paths"]["arena"]).as_uri()},
        },
    }
    values = {str(tmp_path / "environments/cpu-3.13/bin/python"): state}
    prepare.validate_environments(values, profile)
    state["direct_urls"]["zephon"]["url"] = (tmp_path / "other").as_uri()
    with pytest.raises(ValueError, match="Installed zephon"):
        prepare.validate_environments(values, profile)


@pytest.mark.parametrize("default_present", [False, True])
def test_ssh_deploy_copies_plain_controller_sources_without_github(
    tmp_path, monkeypatch, default_present
):
    from test_remote import fake_target

    profile = fake_target(tmp_path, monkeypatch)
    raw = profiles.read(profile)
    raw["controller"] = {}
    for name in ("arena", "plots", "zephon"):
        raw["controller"][name] = str(source(tmp_path / f"local {name}"))
        if default_present:
            source(
                Path(raw["machine"]["root"]) / "repositories" / name,
                content="default = True\n",
            )
    with patch.object(
        prepare.urllib.request, "urlopen", side_effect=AssertionError("download")
    ):
        _, _, target_profile = remote.deploy(profile, raw)
    target = core.read_json(Path(target_profile))
    for name, local in raw["controller"].items():
        deployed = Path(target["paths"][name])
        default = Path(raw["machine"]["root"]) / "repositories" / name
        assert not deployed.is_relative_to(default)
        assert default.exists() == default_present
        if default_present:
            assert (default / "module.py").read_text() == "default = True\n"
        assert (deployed / "module.py").read_text() == (
            Path(local) / "module.py"
        ).read_text()
        assert not (deployed / ".git").exists()
        assert provenance.source_state(deployed)["files"] == provenance.source_files(
            Path(local)
        )


def test_training_upload_uses_local_sources_and_changes_image_identity(tmp_path):
    args = SimpleNamespace(
        zephon=source(tmp_path / "my zephon"),
        torchtitan=source(tmp_path / "my trainer"),
    )
    config = sources.configure(args, protocol.load(small=True))
    manifest = {"config": config, "code": protocol.code_identity()}
    with tarfile.open(fileobj=io.BytesIO(ssh.source_archive(manifest))) as archive:
        dockerfile = archive.extractfile("Dockerfile").read().decode()
        assert "git clone" not in dockerfile
        assert "--no-emit-package zephon" in dockerfile
        assert (
            archive.extractfile("local_sources/zephon/module.py").read()
            == b"value = 1\n"
        )
        assert (
            archive.extractfile("local_sources/torchtitan/module.py").read()
            == b"value = 1\n"
        )
        extracted = tmp_path / "image"
        archive.extractall(extracted, filter="data")
    identity = ssh.image_name(manifest)
    observed = sources.verify_installed(config, extracted / "local_sources")
    sources.validate_observed(
        {"local_sources": observed, "zephon": "custom", "torchtitan_commit": None},
        config,
    )
    (args.zephon / "module.py").write_text("new_source = True\n")
    with pytest.raises(ValueError, match="changed after selection"):
        ssh.source_archive(manifest)
    updated = sources.configure(args, protocol.load(small=True))
    assert ssh.image_name(manifest | {"config": updated}) != identity
    (extracted / "local_sources/zephon/module.py").write_text("wrong = True\n")
    with pytest.raises(ValueError, match="selected zephon"):
        sources.verify_installed(config, extracted / "local_sources")


def test_large_training_source_inventories_do_not_travel_in_environment(tmp_path):
    roots = {}
    for name in ("zephon", "torchtitan"):
        root = source(tmp_path / name)
        for index in range(800):
            (root / f"module_{index:04d}_with_a_descriptive_name.py").write_text(
                "value = 1\n"
            )
        roots[name] = root
    assert (
        len(
            json.dumps(
                {name: provenance.source_files(root) for name, root in roots.items()}
            )
        )
        > 131072
    )
    config = sources.configure(SimpleNamespace(**roots), protocol.load(small=True))
    assert len(json.dumps(config).encode()) < 16384
    for entry in config["local_sources"].values():
        assert "files" not in entry and "files" not in entry["state"]
        sources.verify(entry)


def test_training_source_options_apply_to_prepare_check_plan_and_run(tmp_path):
    directory = source(tmp_path / "zephon")
    commands = [
        ["prepare", "training-data"],
        ["check", "training"],
        ["plan", "training"],
        ["run", "training"],
    ]
    for command in commands:
        extra = [] if command[0] == "plan" else ["--prefix", "source-test"]
        if command[0] == "run":
            extra += ["--output", str(tmp_path / "run")]
        args = cli.parser().parse_args([*command, *extra, "--zephon", str(directory)])
        config = sources.configure(args, protocol.load())
        assert config["local_sources"]["zephon"]["path"] == str(directory)


def test_local_modal_source_needs_no_installed_distribution(tmp_path, monkeypatch):
    package = source(
        tmp_path / "helper/zephon_modal", "__init__.py", "SELECTED = True\n"
    )
    args = SimpleNamespace(zephon_modal=package.parent)
    config = sources.configure(args, protocol.load())
    monkeypatch.delitem(sys.modules, "zephon_modal", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    sources.use_modal(config)
    import zephon_modal

    assert zephon_modal.SELECTED
    monkeypatch.delitem(sys.modules, "zephon_modal")


def test_modal_dependency_pin_applies_only_without_an_explicit_source(
    tmp_path, monkeypatch
):
    import importlib.metadata
    from vldb_experiments.training.cli import dependency_identity

    package = source(tmp_path / "helper/zephon_modal", "__init__.py")
    config = protocol.load()
    monkeypatch.setitem(
        sys.modules,
        "zephon_modal",
        SimpleNamespace(__file__=str(package / "__init__.py")),
    )
    monkeypatch.setitem(
        sys.modules, "modal", SimpleNamespace(__version__=config["sources"]["modal"])
    )
    installed = {"vcs_info": {"commit_id": "different"}}
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda file: json.dumps(installed)),
    )
    with pytest.raises(ValueError, match="pinned zephon-modal"):
        dependency_identity(config)
    installed["vcs_info"]["commit_id"] = config["sources"]["zephon_modal"]
    assert dependency_identity(config)["commit"] == config["sources"]["zephon_modal"]
    selected = sources.configure(SimpleNamespace(zephon_modal=package.parent), config)
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert dependency_identity(selected)["commit"] is None


def test_modal_image_copies_selected_sources_without_git_or_pypi_zephon(
    tmp_path, monkeypatch
):
    from vldb_experiments.training.modal_app import training_image

    class Image:
        def __init__(self):
            self.commands = []
            self.files = []

        def apt_install(self, *args):
            return self

        def pip_install(self, *args):
            return self

        def add_local_python_source(self, *args):
            return self

        def env(self, *args):
            return self

        def add_local_file(self, local, remote, **kwargs):
            self.files.append((Path(local), remote))
            return self

        def add_local_dir(self, local, remote, *, ignore, copy):
            assert copy
            for file in Path(local).rglob("*"):
                relative = file.relative_to(local)
                if file.is_file() and not ignore(relative):
                    self.files.append((file, f"{remote}/{relative.as_posix()}"))
            return self

        def run_commands(self, *commands):
            self.commands.extend(commands)
            return self

    image = Image()
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(Image=SimpleNamespace(from_registry=lambda *a, **k: image)),
    )
    args = SimpleNamespace(
        zephon=source(tmp_path / "zephon"), torchtitan=source(tmp_path / "trainer")
    )
    config = sources.configure(args, protocol.load(small=True))
    assert training_image(config) is image
    assert not any("git clone" in command for command in image.commands)
    assert any("--no-emit-package zephon" in command for command in image.commands)
    assert any(
        "--no-deps -e /opt/local_sources/zephon" in command
        for command in image.commands
    )
    for name in ("zephon", "torchtitan"):
        assert any(
            remote == f"/opt/local_sources/{name}/module.py"
            for _, remote in image.files
        )
