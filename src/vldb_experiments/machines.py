"""Prepare CPU hosts and provision EC2 through arena's instance launcher."""

from __future__ import annotations

import contextlib
import fcntl
import importlib.util
import json
import platform
import re
import shutil
import sys
import tomllib
import uuid
from pathlib import Path

from . import ROOT, core, prepare, profiles, provenance


@contextlib.contextmanager
def setup_lock(profile):
    path = Path(profile["measurement"]["lock"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "The target is measuring or preparing another experiment"
            ) from None
        yield


def check() -> dict:
    result = dict(
        system=platform.system(),
        architecture=platform.machine(),
        python=platform.python_version(),
        uv=shutil.which("uv"),
        host=provenance.host_snapshot(),
    )
    result["ready"] = (
        result["system"] == "Linux"
        and result["architecture"] in ("x86_64", "AMD64")
        and bool(result["uv"])
    )
    return result


def setup(profile):
    host = check()
    if not host["ready"]:
        raise ValueError(
            "CPU setup requires Linux x86-64 and uv on the target; AWS bootstrap installs these"
        )
    with setup_lock(profile):
        prepare.sources(profile, "v0.1.0")
        prepare.environments(profile, "v0.1.0", [])
        prepare.assets(profile, "v0.1.0")
        prepare.plotting(Path(profile["paths"]["plots"]), python="3.13.15")
        normal = core.environment_path(profile, {"python": "3.13"}, "v0.1.0")
        prepare.call(
            [
                normal,
                "-c",
                "import dataloader_arena.checkpoint, grain, litdata, zephon; print('CPU environment ready')",
            ]
        )
        root = Path(profile["paths"]["environments"]).parent
        core.write_json(
            root / "setup.json",
            dict(
                host=host,
                profile=profile,
                sources={
                    k: provenance.source_state(Path(profile["paths"][k]))
                    for k in ("arena", "plots")
                },
                environments=provenance.environment_states(
                    [
                        Path(profile["paths"]["environments"]) / n / "bin/python"
                        for n in ("cpu-3.13", "cpu-3.14t")
                    ]
                ),
                assets=core.read_json(root / "assets.json"),
            ),
        )


def receipt_path(path: Path, raw: dict) -> Path:
    value = raw.get("machine", {}).get("receipt")
    return (
        Path(value).expanduser().absolute()
        if value
        else Path(path).expanduser().absolute().with_suffix(".instance.json")
    )


def aws_plan(raw: dict) -> dict:
    profiles.local_target(raw)
    aws = raw.get("aws", {})
    required = ("name", "security_group", "key_pair_name")
    if any(not aws.get(name) for name in required):
        raise ValueError(
            "Set aws.name, aws.security_group and aws.key_pair_name; SSH authentication must be configured"
        )
    connect = raw.get("machine", {}).get("connect", "ssm")
    if connect not in ("ssm", "private-ip"):
        raise ValueError("machine.connect must be ssm or private-ip")
    if connect == "ssm" and not aws.get("iam_instance_profile"):
        raise ValueError(
            "SSM SSH requires aws.iam_instance_profile with Session Manager permissions"
        )
    return dict(
        name=aws["name"],
        instance_type=aws.get("instance_type", "r8id.48xlarge"),
        region=aws.get("region", "us-east-1"),
        security_group=aws["security_group"],
        subnet_tag=aws.get("subnet_tag"),
        key_pair_name=aws["key_pair_name"],
        iam_instance_profile=aws.get("iam_instance_profile"),
        connect=connect,
        root=raw["machine"]["root"],
        ami_id=aws.get("ami_id"),
        root_volume_size=aws.get("root_volume_size", 200),
        default_arena_revision=tomllib.loads(
            (ROOT / "configs/shared/sources.toml").read_text()
        )["arena"]["revision"],
    )


def create(path: Path, *, plan=False) -> dict:
    raw = profiles.read(path)
    if raw.get("machine", {}).get("kind") != "aws":
        raise ValueError("machine create requires machine.kind=aws")
    request = aws_plan(raw)
    if plan:
        return request
    receipt = receipt_path(path, raw)
    if receipt.exists():
        previous = core.read_json(receipt)
        previous_request = dict(previous["request"])
        if "arena_revision" in previous_request:
            previous_request["default_arena_revision"] = previous_request.pop(
                "arena_revision"
            )
        if previous_request != request:
            raise ValueError(
                "Instance request changed; use a new profile/receipt for a new machine"
            )
        if previous["state"] != "ready":
            raise RuntimeError(
                "A previous launch is unresolved; inspect its ownership tag in EC2 before creating another instance"
            )
        return previous
    # Use a public arena checkout only on the controller for its provisioning code.
    supplied = raw.get("controller", {}).get("arena")
    arena = (
        Path(supplied).expanduser()
        if supplied
        else Path.home()
        / ".cache/zephon-vldb/provisioning"
        / request["default_arena_revision"]
        / "arena"
    )
    if not supplied:
        prepare.export_source(
            dict(
                repository="datologyai/dataloader-arena-public",
                revision=request["default_arena_revision"],
                directory="arena",
            ),
            arena.parent,
        )
    arena_source = provenance.source_state(arena)
    module = provisioner(arena)
    for handler in module.LOGGER.handlers:
        handler.setStream(sys.stderr)
    module.REGION_NAME = request["region"]
    user_data = module.load_user_data(arena / "scripts/bootstrap_benchmark.sh")
    user_data, skipped_clones = re.subn(
        r"(?m)^[ \t]*clone_repo_to_nvme[ \t]*(?:#.*)?$",
        "    : # The reproducer prepares the selected repository sources after bootstrap.",
        user_data,
    )
    if not skipped_clones and "clone_repo_to_nvme" in user_data:
        raise ValueError(
            "Arena bootstrap has an unrecognized repository clone call; use a compatible bootstrap script"
        )
    user_data = user_data.replace(
        "\n",
        "\nexport REPO_URL=https://github.com/datologyai/dataloader-arena-public.git\n",
        1,
    )
    user_data = module.inject_cstate_setup(user_data)
    user_data = module.inject_open_file_limit(user_data, 65536)
    user_data = module.inject_perf_stabilize(user_data)
    owner = uuid.uuid4().hex
    config = module.LaunchConfig(
        instance_name=request["name"],
        instance_type=request["instance_type"],
        key_pair_name=request["key_pair_name"],
        security_group_name=request["security_group"],
        subnet_tag=request["subnet_tag"],
        ami_pattern=None,
        ami_owner=module.UBUNTU_OWNER_ID,
        ami_id=request["ami_id"],
        iam_instance_profile=request["iam_instance_profile"],
        root_volume_size=request["root_volume_size"],
        shutdown_behavior="stop",
        user_data=user_data,
        tags={
            "Name": request["name"],
            "Role": "vldb-reproduction",
            "vldb-owner": owner,
        },
        restrict_cstates=True,
        open_file_limit=65536,
        disable_smt=False,
        core_count=None,
        stabilize_performance=True,
        ubuntu_version="24.04",
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    # The exclusive receipt is written before requesting an instance. An uncertain
    # API outcome cannot cause a second instance on a repeated command.
    with receipt.open("x") as file:
        json.dump(
            dict(
                state="launching",
                request=request,
                owner=owner,
                arena_source=arena_source,
                bootstrap_repo_clone_skipped=bool(skipped_clones),
            ),
            file,
            indent=2,
        )
    instance_id, private_ip = module.create_instance(config)
    result = dict(
        state="ready",
        request=request,
        arena_source=arena_source,
        bootstrap_repo_clone_skipped=bool(skipped_clones),
        owner=owner,
        instance_id=instance_id,
        private_ip=private_ip,
        region=request["region"],
        bootstrap="Wait for cloud-init; reboot once to apply the CPU boot settings, then run machine setup.",
    )
    core.write_json(receipt, result)
    return result


def provisioner(arena: Path):
    """Load the pinned arena implementation without constructing a cloud client."""
    script = arena / "scripts/launch_benchmark_instance.py"
    spec = importlib.util.spec_from_file_location("vldb_arena_instance", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Install provisioning dependencies with uv sync --extra aws"
        ) from error
    return module
