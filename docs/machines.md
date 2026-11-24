# Machine setup

Throughput, TTFB and checkpointing can run on a local
machine, connect to an existing one over SSH, or let the launcher create an AWS
instance that matches our paper setup. [Training](training.md) uses a separate GPU setup.

The paper used an AWS **r8id.48xlarge** running Ubuntu 24.04: 96 physical Intel
Xeon cores, 1.5 TiB RAM and local NVMe in RAID-0. The AWS profile
selects this instance type by default. Other machines can run the experiments,
but their absolute timings will differ.

## Choose a profile

From the launcher directory, run `uv sync`, then copy and edit one of these
example profiles. The commands below use `profiles/bench.toml` as an example:

| Profile | Where benchmark runs |
| --- | --- |
| [local.example.toml](../profiles/local.example.toml) | On the machine running the CLI. |
| [ssh.example.toml](../profiles/ssh.example.toml) | On an existing machine reached through SSH. |
| [aws.example.toml](../profiles/aws.example.toml) | On an AWS instance created by the launcher. |

`machine.root` is the working directory on the benchmark machine. It holds the
repositories, environments, local datasets, and caches. For the cache and data you should use a fast local NVMe. Checkpointing also needs S3 access because it needs a realistic (i.e., terabyte-scale) dataset to give useful measurements.

### Local machine or existing SSH machine

Keep your benchmarking machine free of other work while measuring. The benchmark
user needs sudo for operations such as dropping the OS page cache between
measurements. The launcher sets up the experiment dependencies.

For SSH, set `machine.host` to an SSH alias or `user@host`. Normal SSH login and
host-key verification must already work. Use absolute paths for locations on
the remote machine. It also needs network access to the data sources and packages.

```sh
uv run reproduce machine setup --profile profiles/bench.toml --plan
uv run reproduce machine setup --profile profiles/bench.toml
uv run reproduce check machine --profile profiles/bench.toml
```

`machine setup` prepares the repositories, shared CPU environments and
preprocessing assets. It does not generate benchmark data; continue with the
[throughput](throughput.md), [TTFB](ttfb.md) or [checkpointing](checkpoint.md) guide.

For SSH/AWS runs, `machine setup` also prepares plotting on the local machine,
where results are downloaded and plotted after measurement. Training does not
use `machine setup`, so use `prepare plotting` separately.

### Create an AWS machine

Start from `profiles/aws.example.toml`. Set the region, subnet, security group,
EC2 key pair, and IAM instance profile for your AWS account. Configure AWS
credentials on the local machine. The default connection uses SSH through SSM,
which needs the AWS CLI, Session Manager plugin, and an appropriate instance role.
You can instead set `machine.connect = "private-ip"` if you have network access
to the instance's private IP.

```sh
uv sync --extra aws
uv run reproduce machine create --profile profiles/bench.toml --plan
uv run reproduce machine create --profile profiles/bench.toml
uv run reproduce machine setup --profile profiles/bench.toml
uv run reproduce check machine --profile profiles/bench.toml
```

Creation installs the base tools, prepares local NVMe and applies benchmark host
settings, such as restricting deep C-states to reduce timing variance. Setup
waits for this to finish and reboots if needed. On an existing machine, similar to any other SSH host, arrange
storage and host tuning before running the benchmarks.

The launcher does not stop AWS instances automatically. Stop or terminate yours
through AWS when finished. Stopping the instance erases its NVMe data, including
prepared throughput and TTFB inputs.

## Repositories and credentials

Only this repository needs to be present on the local machine initially. For SSH
or AWS execution, the launcher uploads its code and recipes to the benchmark machine
and prepares the other repositories there.

Credentials are not copied. You need to give the benchmark machine its own data access,
usually through an IAM role on AWS.

### Using existing source directories

By default, setup downloads missing repositories at the revisions in
[`configs/shared/sources.toml`](../configs/shared/sources.toml) and
[`configs/shared/revisions.toml`](../configs/shared/revisions.toml). If you already
have the source code, point the launcher at its directory instead. The same
settings accept a Git checkout or a standalone copy, such as an unpacked GitHub
source ZIP or code from a Zenodo software archive.

In the CPU machine profile, `[paths]` selects directories on the benchmark machine:

```toml
[paths]
arena = "/mnt/local/dataloader-arena-public"
plots = "/mnt/local/zephon-vldb-plots"
zephon = "/mnt/local/zephon"
```

For SSH/AWS execution, `[controller]` selects directories on the local machine
running the launcher. Setup uploads their source files and installs their
dependencies on the target:

```toml
[controller]
plots = "/home/USER/dev/zephon-vldb-plots"
arena = "/home/USER/dev/dataloader-arena-public"
zephon = "/home/USER/dev/zephon"
```

An explicit `[paths]` entry selects a directory already on the target and takes
precedence over a `[controller]` entry for the same repository. Remote `[paths]`
must be absolute; local paths can be relative to the profile file.

Supplied directories take precedence over the default download revisions.
Their names and revisions do not have to match the defaults, but their code must
support the experiment being run. A missing supplied directory causes an error.
Setup leaves their source files unchanged and records the actual contents used.
Changing sources requires a new campaign directory; completed observations retain
their original provenance.

For plotting commands, `--plots PATH` selects an existing plotting source
directory on the machine running the launcher. Without a controller plotting
override, remote setup prepares the renderer beside this launcher as
`zephon-vldb-plots`.

#### Example: standalone copies in sibling directories

If you obtained standalone source copies, for example from a Zenodo software
archive provided for reproducibility, you can arrange them as siblings:

```text
zephon-vldb-experiments/
dataloader-arena-public/
zephon-vldb-plots/
zephon/
```

[`profiles/artifact.example.toml`](../profiles/artifact.example.toml) shows the
same `[paths]` settings above, with relative paths to the siblings. Adjust the
paths for the locations of your profile and source directories. Set the machine
storage root and, for checkpointing, your S3 prefixes, then follow the normal
setup, preparation, check and run commands.

For an SSH/AWS target, use the same sibling paths in `[controller]` instead of
`[paths]`, along with the machine settings from the SSH/AWS example profile.
Setup uploads those sources and uses them on the target. Selecting local source
code does not provide Python interpreters, third-party packages, model/tokenizer
assets or benchmark datasets; setup and data preparation obtain those separately.

[Training](training.md#using-existing-source-directories) selects existing sources
through command-line options instead of CPU machine profiles.

For remote execution and collecting results, see the [overview](README.md#remote-runs-and-recovery).
