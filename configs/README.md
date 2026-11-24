# Experiment configuration

The four experiment directories contain the checked-in reproduction recipes.
`shared/` contains common setup and provenance pins. The launcher loads these
files automatically; you select an experiment and target profile in the CLI.
Machine addresses, storage paths and credentials do not belong in these recipes.

Every file here supports reproduction or shared setup. The one optional preset,
`training/small.json`, runs a cheaper training validation.

## Throughput

| File | Status | Purpose |
| --- | --- | --- |
| [fetch-and-batch.json](throughput/fetch-and-batch.json) | Reproduction | Figure 6 loaders, DOPs, batch windows, variants and seeds. |
| [vlm-pipeline.json](throughput/vlm-pipeline.json) | Reproduction | Figure 7 pipeline configurations, allocations and seeds. |
| [data.toml](throughput/data.toml) | Reproduction | Public text/VLM data preparation, source pins, sampling and mixing. TTFB also inherits these source settings. |

## TTFB

| File | Status | Purpose |
| --- | --- | --- |
| [experiment.json](ttfb/experiment.json) | Reproduction | Startup measurements and the fixed VLM allocation used for the results in §8.1.3. |
| [data.toml](ttfb/data.toml) | Reproduction | TTFB-specific sample counts, shard layout and size checks; inherits source/tokenizer/mixing settings from throughput. |
| [text-shards.json](ttfb/text-shards.json) | Reproduction | Per-shard record-count template used to generate and validate text input. |
| [vlm-shards.json](ttfb/vlm-shards.json) | Reproduction | Per-shard record-count template used to generate and validate VLM input. |

The shard templates specify the ordered per-shard record counts used to prepare
and validate the TTFB inputs.

## Checkpointing

| File | Status | Purpose |
| --- | --- | --- |
| [experiment.json](checkpoint/experiment.json) | Reproduction | Save/restore, progress and lane-scaling matrix, loader settings and protocol. |
| [data.toml](checkpoint/data.toml) | Reproduction | Large S3 data preparation, decoded-image format, resource limits and scale checks. |
| [assets.json](checkpoint/assets.json) | Reproduction | Pinned checkpoint tokenizer and image processor files. |

## Training

| File | Status | Purpose |
| --- | --- | --- |
| [experiment.json](training/experiment.json) | Reproduction | Figure 1 Llama3 1B recipe, both loader arms, resize stages, public assets and training dependency pins. |
| [small.json](training/small.json) | Optional validation | Cheaper single-node run selected by `--small`. Keeps the training recipe while reducing GPU counts. |

## Shared setup

| File | Status | Purpose |
| --- | --- | --- |
| [sources.toml](shared/sources.toml) | Reproduction | Arena and plotting repository revisions. |
| [revisions.toml](shared/revisions.toml) | Reproduction | Zephon v0.1.0 source pin used by the CPU experiments. Training pins its own version in its recipe. |
| [vlm-tokenizer.json](shared/vlm-tokenizer.json) | Reproduction | Common VLM tokenizer assets for throughput and TTFB. Checkpointing and training declare their own assets. |

Source revisions are pinned once in the shared files. Generated dataset manifests
record provenance and supply the checksums used by optional `--hash-data` checks.
