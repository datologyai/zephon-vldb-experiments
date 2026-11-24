# Environments

Each environment maintained here has the same two files:

- `pyproject.toml`: direct dependencies, supported Python/platform and package indexes.
- `uv.lock`: the resolved dependency versions. Setup uses this lock without upgrading packages.

| Environment | Runs | Setup |
| --- | --- | --- |
| CPU | Throughput, TTFB, checkpointing and data preparation | Uses the pinned arena repository's `pyproject.toml` and single `uv.lock`. There is no second CPU lock here. |
| [plotting](plotting/pyproject.toml) | Original paper aggregation and figures | Installs into the plotting checkout's `.venv`. The plotting code comes from the pinned repository; its own dependency files are left untouched. |
| [training](training/pyproject.toml) | CUDA/TorchTitan training on Modal or SSH | Builds the Python 3.12, CUDA 13 image from this lock. Both image builders export temporary requirements from the lock before installing. |

Ordinary reproduction commands prepare these automatically. To intentionally
update dependencies, maintainers edit the relevant project and regenerate its lock:

```sh
uv lock --project environments/plotting --python 3.13
uv lock --project environments/training --python 3.12
```

Training targets Linux x86-64. Its Torch and matching Triton nightly builds use
the explicitly named CUDA package index. A dependency update changes the
experiment environment and requires a new campaign.
