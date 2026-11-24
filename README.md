# Zephon VLDB experiments

In this repository, you'll find a one-stop shop CLI (`reproduce`) and recipes to reproduce the experiments from
our Zephon VLDB paper.

| Experiment | Command name | Paper reference |
| --- | --- | --- |
| Elastic training determinism | `training` | Figure 1 |
| Data loading throughput | `throughput` | Figures 6 and 7, §8.1.1–8.1.2 |
| Time to first batch | `ttfb` | §8.1.3 |
| Checkpointing | `checkpoint` | §8.2 |

## How to run the experiments

For all experiments that don't require a GPU (i.e., everything besides `training`), you have the choice between running this launcher directly on the machine being benchmarked, connecting
to an existing machine over SSH, or creating an AWS instance through the launcher.
The AWS option uses the paper's **r8id.48xlarge** instance type: 96 physical cores,
1.5 TiB RAM and local NVMe storage. It applies benchmark host settings such as
restricting deep C-states. See [machine setup](docs/machines.md) to get started.

The CPU benchmark machine needs Linux on x86-64, uv, local NVMe and sudo access.
Keep it free of other work while measuring.

Training requires GPUs. You can run a smaller version of the experiment on
any eight-GPU machine over SSH, or run the paper's 8–64 GPU experiment
through [Modal](https://modal.com/). Both compare uninterrupted training with
training resumed on a different number of GPUs.

You need [uv](https://docs.astral.sh/uv/) and a local copy of this repository.
The commands below clone it from GitHub. If you already have a checkout or an
extracted standalone copy, start at `uv sync` inside that directory.
Then prepare a profile, which describes on which machine to run the benchmark:

```sh
git clone https://github.com/datologyai/zephon-vldb-experiments.git
cd zephon-vldb-experiments
uv sync

# Choose local.example.toml, ssh.example.toml or aws.example.toml.
cp profiles/local.example.toml profiles/bench.toml
# Edit profiles/bench.toml for your machine and storage paths.
```

| Profile | Fields to edit |
| --- | --- |
| `local.example.toml` | `machine.root` |
| `ssh.example.toml` | `machine.host`, `machine.root` |
| `aws.example.toml` | The `[aws]` section and `machine.identity_file` |

`machine.root` is the working directory on the CPU machine; put it on local
NVMe. For checkpointing, also set your S3 prefixes under `[checkpoint]`.

Then, to prepare the data and run throughput on the machine described in `profiles/bench.toml` (allow for
sufficient free disk space for preparation):

```sh
uv run reproduce machine setup --profile profiles/bench.toml
uv run reproduce prepare throughput-data --profile profiles/bench.toml
uv run reproduce check throughput --profile profiles/bench.toml
uv run reproduce run throughput --profile profiles/bench.toml --output runs/throughput
```

`machine setup` uses supplied sources or downloads pinned repositories and installs their
dependencies. `prepare` writes the input data, `check` validates it, and `run`
measures, prints the results and saves them in the output directory.

For an AWS profile, install the AWS dependencies with `uv sync --extra aws`, then
run `machine create` before `machine setup`. The
[machine guide](docs/machines.md) covers credentials, network access and host
configuration. To use source directories you already have, see
[using existing source directories](docs/machines.md#using-existing-source-directories).
Git checkouts and standalone copies, such as code extracted from a Zenodo software
archive for reproducibility, use the same path settings.


For each experiment, we provide a guide that walks through its data preparation, run
command, and results:

- [Elastic training determinism](docs/training.md)
- [Throughput](docs/throughput.md)
- [Time to first batch](docs/ttfb.md)
- [Checkpointing](docs/checkpoint.md) — preparation creates an estimated 7–10 TB on S3.

For more information, we refer to the [overview](docs/README.md) which explains the workflow and result layout.

## Further repositories

This repository here only contains a convenient CLI interface to run all experiments to make reproduction as simple as possible.
We leverage further repositories for running experiments and creating plots:

- Our [dataloader arena](https://github.com/datologyai/dataloader-arena-public)
implements the CPU benchmarks and data preparation
- Our [plotting repo](https://github.com/datologyai/zephon-vldb-plots) provides the
paper's plotting and aggregation code
- Training uses our [torchtitan-zephon](https://github.com/datologyai/torchtitan-zephon)
integration
- The Modal backend also uses
[zephon-modal](https://github.com/datologyai/zephon-modal)

By default, setup downloads missing repositories at pinned revisions and prepares
their dependencies. You can instead [select existing source directories](docs/machines.md#using-existing-source-directories).

## CLI overview

```text
reproduce
  machine create       Create an AWS CPU benchmark machine
  machine setup        Set up a local, SSH or newly created AWS CPU machine
  prepare              Prepare experiment data or set up plotting
  plan                 Show the experiment without setting up or running it
  check                Validate a CPU machine or an experiment's inputs
  run                  Run an experiment, print its summary and make its figures
  status               Show progress for a saved run
  collect              Fetch results from an SSH or Modal run, summarize and plot
  resume               Resume a failed remote run after inspecting its logs
  summarize            Reprint a run's results from its saved measurements
  plot                 Regenerate the paper figures from saved measurements
```

`plan` and `run` take an experiment name from the table above. `prepare` takes
`throughput-data`, `ttfb-data` or `checkpoint-data` for the performance experiments, and
`training-data` for training. `check` takes `machine` or any experiment name.
You don't need a machine profile for training. For every experiment, prepare the data first,
then use `check` and `run`. A run fails if its inputs are missing or invalid.

`plan` and `--plan` do different things:

- `reproduce plan EXPERIMENT` lists what an experiment will run. It needs no
  machine or data.
- `--plan` on `machine create`, `machine setup` or `prepare` shows what would
  change, without changing it.

Some useful commands:

```sh
uv run reproduce plan throughput             # What throughput will run
uv run reproduce prepare throughput-data --profile profiles/bench.toml --plan # Preview data preparation
uv run reproduce status runs/throughput      # Progress of a saved run
uv run reproduce summarize runs/throughput   # Print the results again
uv run reproduce plot runs/throughput        # Make the figures again
uv run reproduce run throughput --help       # Options for one experiment
```

Common options:

| Option | Use |
| --- | --- |
| `--profile PATH` | Select the CPU machine and data locations. Not used by training. |
| `--output PATH` | Choose where a run's results are saved. |
| `--plan` | Preview machine creation, setup or preparation without making changes. |
| `--detach` | Submit an SSH, AWS or Modal run and collect its results later. |
| `--plots PATH` | Use an existing plotting source directory for collection, summaries or figures. |
| `--format json` | Print a machine-readable summary with `summarize`. |

Each experiment might have additional options, described in its guide and
`--help`.  Results print to the terminal; raw measurements and logs are kept in the output directory.
Throughput and training also generate the paper figures.
