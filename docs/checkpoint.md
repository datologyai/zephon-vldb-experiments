# Checkpointing

The checkpointing experiment reproduces §8.2: how much loader state is retained,
how long saving and recovery take, and how those costs change with training
progress and lane count. It runs eight CPU processes through the VLM pipeline,
without a model or optimizer. Each canonical lane shuffles and packs its own
64-sample windows, preserving all contributors when constructing batches.

Follow [machine setup](machines.md) before preparing data or running measurements.

## Prepare the data

**This is a multi-terabyte preparation.** The recipe needs about
10 million MAmmoTH-VL conversations, with **7–10 TB of final S3 data**
and **12–15 TB of peak S3 storage** while preparation is in progress. Preparation also makes
substantial network transfers. Review the plan before starting. If you already
have a suitable S3 dataset, use the [existing-data option](#reuse-an-existing-dataset).

Set your own S3 prefixes in `profiles/bench.toml`:

```toml
[checkpoint]
dataset = "s3://YOUR_BUCKET/vldb/vlm"
artifacts = "s3://YOUR_BUCKET/vldb/checkpoints"
aggregation = "s3://YOUR_BUCKET/vldb/aggregation"

[checkpoint_data]
source = "public"
journal = "/home/USER/.local/state/zephon-vldb/checkpoint-preparation"
```

The dataset prefix holds the prepared input; the other two are used for saving
checkpoints. All three must be S3 locations. Configure S3 access on the benchmark
machine before continuing.

```sh
uv run reproduce prepare checkpoint-data --profile profiles/bench.toml --plan
uv run reproduce prepare checkpoint-data --profile profiles/bench.toml
```

Preparation prints its storage and transfer estimates and asks you to type
`yes`. For unattended preparation, pass `--yes` to accept those estimates.

You do not need space for the entire dataset locally. Preparation streams through
bounded local scratch, using S3 for the large intermediate output and final
shards. The default scratch budget is 128 GiB. If preparation is interrupted, repeat the same command with the same
recipe and locations to resume it.

### Reuse an existing dataset

If you already have a compatible full-scale S3 export, set
`checkpoint_data.source = "existing"`. Skip `prepare checkpoint-data` and
validate it with the check below.

## Run the experiment

Validate either the new or existing dataset before measuring. The check verifies
that the input is on S3 and has the expected scale, shard layout and format:

```sh
uv run reproduce check checkpoint --profile profiles/bench.toml
uv run reproduce plan checkpoint
uv run reproduce run checkpoint --profile profiles/bench.toml --output runs/checkpoint
```

The default runs all three parts: checkpoint costs, training progress and lane
scaling. It measures fresh startup, saving and restoring with a cold download
cache. Restored output is checked against the uninterrupted continuation.

Use `--suite main`, `--suite progress` or `--suite lanes` to run one part.
`--seed` and `--repetition` select individual measurements and can be repeated.
All of these options also work with `plan`. The complete recipe is in
[`configs/checkpoint/experiment.json`](../configs/checkpoint/experiment.json).

## Results

The summary follows the paper's order: retained state and its scaling with
training progress, save cost, restore cost, and lane scaling. It reports checkpoint bytes, save time and
fresh/restore time to first batch, with medians and interquartile ranges.

Raw measurements, logs and setup details are saved under `runs/checkpoint/`.
Temporary preparation data is removed after successful preparation; the final
input and saved checkpoints remain on S3 until you delete them.

```sh
uv run reproduce summarize runs/checkpoint
```
