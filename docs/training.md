# Elastic training determinism

This experiment reproduces the loss curve comparison in Figure 1.
It trains a model, saves a checkpoint, and resumes on different GPU counts, in order to simulate a fault and continuing on a different GPU count.
Each resumed loss curve and sequence of training batches is compared with its
own uninterrupted reference.

The experiment has two loader configurations: a standard control whose data
order depends on GPU count, and Zephon with a fixed number of canonical lanes.
Both restore the model and optimizer; Zephon also restores the loader's position
without changing the global batches when the GPU count changes.

The experiment trains a randomly initialized Llama3 1B model on
FineWeb-Edu text. Preparation downloads two Parquet shards and the tokenizer,
then builds an index. The files stay on the SSH host or in a shared Modal volume;
tokenization and packing happen during training.

You can either use an existing eight-GPU machine reachable over SSH for a smaller scale variant of the experiment, or use
Modal for the paper's 8–64 GPU experiment. Neither uses the CPU machine profiles
or `machine create`.

Set up the plotting code on the local machine once, unless you already ran
`machine setup` for another experiment:

```sh
uv run reproduce prepare plotting
```

To use an existing plotting source directory, add `--plots PATH` here and to `run`,
`collect` or `plot`.

## Single-node experiment over SSH

The host needs Linux, Docker with NVIDIA Container Toolkit, and eight NVIDIA
GPUs with BF16 support and enough memory for the 1B model. Verify SSH login and
GPU access through Docker before launching. `prepare training-data` sets up the
training environment and input data on that host.

```sh
uv sync --extra training
uv run reproduce plan training --small
uv run reproduce prepare training-data --small --ssh USER@GPU_HOST \
  --remote-root /scratch/USER/vldb --prefix zephon-small-01
uv run reproduce check training --small --ssh USER@GPU_HOST \
  --remote-root /scratch/USER/vldb --prefix zephon-small-01
uv run reproduce run training --small --ssh USER@GPU_HOST \
  --remote-root /scratch/USER/vldb --prefix zephon-small-01 \
  --output runs/training-small
```

The smaller experiment uses a two-GPU baseline and resumes on one, four and
eight GPUs. Choose a unique prefix using lowercase letters, digits and
hyphens, and a new output directory for each run.

## Paper-scale experiment on Modal

You first need to set up Modal access. The full experiment needs
multi-node Clusters/RDMA access and quota for eight nodes with eight H100s each.
These runs will incur GPU charges.

```sh
uv sync --extra training
uv run modal setup
uv run reproduce plan training
uv run reproduce prepare training-data --prefix zephon-training-01
uv run reproduce check training --prefix zephon-training-01
uv run reproduce run training --prefix zephon-training-01 --output runs/training
```

Preparation runs without GPUs. Add `--plan` to `prepare training-data` to see
which files it will download and where they will be stored. Use the same prefix
and target when you run the experiment. `check training` checks the files and
index; the GPU setup is checked when you launch training. As with the other
experiments, you need to prepare the inputs before running. If they are missing
or invalid, the run stops before starting GPU work.

For each loader configuration, the baseline trains on 16 GPUs through step 300.
It saves a checkpoint at step 150, then resumes from that checkpoint on 8, 32 and
64 GPUs through step 300. Each continuation uses its own loader configuration's
checkpoint and uninterrupted reference.

The complete settings are in
[`configs/training/experiment.json`](../configs/training/experiment.json);
[`small.json`](../configs/training/small.json) changes the GPU counts for the
single-node experiment.

### Using existing source directories

`prepare training-data`, `check training`, `plan training` and `run training`
accept `--zephon PATH`, `--torchtitan PATH` and `--zephon-modal PATH`. These are the
training equivalents of the CPU profile's [source directory settings](machines.md#using-existing-source-directories).
They select directories on the machine running the launcher. A Git checkout and a
standalone copy, such as code extracted from a Zenodo software archive, use the
same options.
Selected Zephon and TorchTitan sources are copied into the SSH/Modal
training image; the Modal integration is imported from its selected directory.
Their actual contents are recorded instead of enforcing the default source pins.
Other packages still use the locked training environment.

The CPU baseline uses Zephon v0.1.0; training defaults to v0.2.1 and requires its
training API. Supply a compatible checkout when overriding either baseline.
`--zephon-modal` selects the Modal integration and is unnecessary for SSH runs.

When providing a local `zephon-modal`, install the client dependencies without
fetching the default integration, then run with syncing disabled:

```sh
uv sync --extra training --no-install-package zephon-modal
uv run --no-sync reproduce prepare training-data --small --prefix zephon-local-01 \
  --torchtitan ../torchtitan-zephon --zephon-modal ../zephon-modal
```

Add `--ssh` and `--remote-root` for an SSH target. Without a source option, the
normal pinned package/repository is used. Supply paths for the sources you have;
setup obtains the remaining dependencies normally.

You can also use `--small` on Modal when multi-node access is not available.

## Results

`run` waits for completion, collects the measurements, prints the summary and
renders the control and Zephon panels of Figure 1. The summary reports whether
training batches and losses match the uninterrupted reference. Zephon continuations
are expected to match exactly; the control is expected to diverge after resizing.

The output directory keeps the measured losses, batch traces, training settings
and checkpoint records. Figures appear as `figure-1-control.*` and
`figure-1-zephon.*` under `report/figures/`.

```sh
uv run reproduce summarize runs/training
uv run reproduce plot runs/training
```

## Collect later or recover a run

Use `--detach` to return after submission, then check progress and collect:

```sh
uv run reproduce run training --prefix zephon-training-01 \
  --output runs/training --detach
uv run reproduce status runs/training
uv run reproduce collect runs/training
```

The remote run continues if the local client disconnects. After inspecting a
failed run's logs, `uv run reproduce resume runs/training` resumes the same
run. Keep the same prefix, code, data and settings.

After collecting a Modal run, find its app ID with `uv run modal app list` and
stop it with `uv run modal app stop APP_ID` (see the
[Modal CLI reference](https://modal.com/docs/cli/latest/app)). Keep the volumes or
remote files containing checkpoints and measurements if you need to resume or
inspect the run later.
