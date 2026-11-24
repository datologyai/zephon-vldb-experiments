# Time to first batch

The TTFB experiment reproduces §8.1.3: how long it takes to construct a loader
and receive its first batch. It covers text fetch-and-batch, raw VLM
fetch-and-batch, and the complete VLM pipeline.

Follow [machine setup](machines.md) first. TTFB uses the same CPU machine as
throughput, with a different input dataset.

## Prepare the data

```sh
uv run reproduce prepare ttfb-data --profile profiles/bench.toml --plan
uv run reproduce prepare ttfb-data --profile profiles/bench.toml
uv run reproduce check ttfb --profile profiles/bench.toml --hash-data
```

Preparation writes public text and VLM records to local NVMe on the benchmark
machine. Allow roughly 300 GiB of free space, plus headroom. No S3 upload is needed.

Startup cost depends on dataset metadata as well as record size. This preset
therefore preserves the paper input's shard-count layout. Do not substitute the
smaller throughput text input. The recipes and layout templates are in
[`configs/ttfb/`](../configs/ttfb/).

`check ttfb` validates the input layout, size and local NVMe location.
`--hash-data` also reads every input shard to verify its checksum.

## Run the experiment

```sh
uv run reproduce plan ttfb
uv run reproduce run ttfb --profile profiles/bench.toml --output runs/ttfb
```

The default measures all three workloads and their CPU-parallelism settings.
The measured window starts at loader construction and ends at the first batch.

Every trial requires a confirmed page-cache drop. If a drop fails or its outcome
is missing, the launcher rejects the sweep and keeps its raw measurements.
Check that the benchmark user has passwordless sudo before rerunning it.

Use `--scenario fetch-text`, `--scenario fetch-vlm` or `--scenario vlm-pipeline`
to select a workload. `--system`, `--dop` and `--seed` select loaders,
parallelism and seeds; repeat a flag to select several values. The same options
work with `plan` and `run`.

## Results

The run prints the fetch-and-batch startup times, followed by the full VLM
pipeline's startup times and comparisons, as discussed in §8.1.3.
Raw measurements, logs and input details are saved under `runs/ttfb/`.

```sh
uv run reproduce summarize runs/ttfb
```

For remote collection and recovery, see the [overview](README.md).
