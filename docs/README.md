# Reproducing the experiments

The experiments share the same workflow: choose a machine, prepare the data,
and run the recipe.

| Guide | What you will run |
| --- | --- |
| [Machine setup](machines.md) | A local CPU machine, an existing SSH machine or a new AWS instance. |
| [Elastic training determinism](training.md) | Training across a checkpoint and a change in GPU count, Figure 1. |
| [Throughput](throughput.md) | Fetch-and-batch and the full VLM pipeline, Figures 6 and 7. |
| [Time to first batch](ttfb.md) | Startup measurements for text, raw VLM data and the full VLM pipeline. |
| [Checkpointing](checkpoint.md) | Checkpoint size, save cost, recovery cost, and scaling. |

## Machines and data

By choosing a profile you define the machine to run the non-training benchmarks on. The same profile works for
throughput, TTFB and checkpointing. With a local profile, preparation and measurement
run on the local machine. With an SSH or AWS profile, they run on the remote
machine. The launcher transfers the code it needs and collects the results.

Data preparation belongs to each experiment. Throughput and TTFB read local
NVMe and checkpointing reads from S3 and needs a much larger dataset. Follow the guide for the experiment you
are running.

Training has its own setup. `prepare training-data` downloads its input data
and tokenizer on Modal or the supplied SSH host; `check training` validates
them. Training does not need a machine profile.

`plan` previews an experiment without needing a machine or data. After setup and
preparation, you can use `check` to validate the experiment's inputs and `run` to measure
it. The experiment guides give the commands for each step.
For every experiment, `run` uses prepared data. If the inputs are missing or
invalid, it stops and tells you what needs to be prepared.

## Results

The summary prints **Paper metrics** first, in paper order, followed by
**Additional measurements** that are not presented in the paper due to space constraints.

The output directory keeps raw measurements, logs, the recipe, and information
about the machine, software, and inputs used.
Throughput and training figures are saved under `report/figures/` within their
result directories. For TTFB and checkpointing, the summary shows the numerical
results discussed in the paper's text, in the same order.

Note that these recipes rely on fully public and open data, whereas in some experiments for the paper we have relied on Datology-internal assets that we cannot release for IP reasons. However, we have validated that our conclusions still hold with the publicly available datasets. For example, for the elastic determinism training experiment, we used an internal training dataset; for the public recipe, we rely on FineWeb, which means the loss curves look different, but importantly, still overlap using Zephon.

For the throughput experiments, note that timings can vary depending on the machine. Recipes are listed in
[the configuration directory](../configs/README.md).

## Remote runs and recovery

By default, a remote `run` waits for completion, collects the results, prints a
summary and generates the applicable figures. To return after submission, use
`--detach` and collect later:

```sh
uv run reproduce run throughput --profile profiles/bench.toml \
  --output runs/throughput --detach
uv run reproduce status runs/throughput
uv run reproduce collect runs/throughput
```

Disconnecting the local client does not cancel the remote run. If a remote run
fails, inspect its logs, then use `uv run reproduce resume runs/throughput` with
the same recipe and setup. For local CPU runs, repeat the original `run` command;
checkpointing additionally requires `--retry-failed` to retry a failed case.
Use a new output directory if you change the code, configuration or inputs.
