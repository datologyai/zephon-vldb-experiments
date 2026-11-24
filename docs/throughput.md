# Throughput

The throughput experiment reproduces Figures 6 and 7. Figure 6 measures fetching
and batching pretokenized text and raw VLM records. Figure 7 runs the complete VLM
pipeline, including image processing, tokenization, shuffling, packing, and batching.
Both run on the CPU benchmark machine without a training model.

Follow [machine setup](machines.md) first. The commands below use the profile
saved as `profiles/bench.toml` and work for local, SSH and AWS targets.

## Prepare the data

```sh
uv run reproduce prepare throughput-data --profile profiles/bench.toml --plan
uv run reproduce prepare throughput-data --profile profiles/bench.toml
uv run reproduce check throughput --profile profiles/bench.toml --hash-data
```

Preparation downloads public FineWeb text and MAmmoTH-VL conversations, then
writes the benchmark inputs to local NVMe on the selected machine. Allow roughly
275 GiB of free space for preparation, plus headroom. No S3 upload is needed.
`--hash-data` verifies the checksum of every prepared shard. The input recipes are in
[`configs/throughput/`](../configs/throughput/).

## Run the experiment

```sh
uv run reproduce plan throughput
uv run reproduce run throughput --profile profiles/bench.toml --output runs/throughput
```

The default runs both figures, including the CPU-parallelism sweeps and the GIL
and shared-memory (`shm`) panels. Each configuration is measured with three seeds.

To run a smaller selection, use these options with `plan` or `run`:

| Option | Selection |
| --- | --- |
| `--figure 6` or `--figure 7` | One throughput figure. |
| `--panel overview`, `gil` or `shm` | One Figure 6 panel. |
| `--system zephon` | One loader; repeat the flag for several. |
| `--dop 64` | One degree of parallelism; repeat for several. |
| `--seed 42` | One workload seed; repeat for several. |
| `--smoke` | A short setup check. |

Of course, the short setup checks do not produce paper figures.

## Results

The run prints the paper throughput measurements and saves Figures 6 and 7.
Figure 6 reports samples/s. Figure 7 reports non-padding tokens/s divided by the
degree of parallelism (DOP).

A full run keeps its two parts under `runs/throughput/fetch-and-batch/` and
`runs/throughput/vlm-pipeline/`. Each contains raw measurements and logs, with
figures under `report/figures/`. A run of just one figure writes that part directly
to the chosen output directory.

```sh
uv run reproduce summarize runs/throughput
uv run reproduce plot runs/throughput
```

These commands use the saved measurements. See the [overview](README.md) for
remote collection and recovery.
