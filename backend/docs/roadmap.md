# Roadmap

## v1 — Local collect + analyze (current)

The v1 CLI is complete and covers the full local loop.

### Completed

- `frx collect -- python train.py`
  - Wraps workload subprocess, injects env vars for SDK auto-persist
  - Background nvidia-smi poller writes `gpu_metrics.csv`
  - Imports artifacts from `--artifact-dir` (default `./frx-job-run`)
  - Generates `derived/summary.json` from SDK trace or profiler bundle
  - Produces a zip bundle with all artifacts
  - Prints captured file list with `[imported]` tags

- `frx analyze <run-dir>`
  - Loads `derived/summary.json` (preferred), falls back to raw trace or profiler bundle
  - Prints verdict, evidence, performance snapshot, and top recommendations
  - `--scope run|steady_state|auto` (default: steady_state when available)
  - `--json` for machine-readable output

- `frx doctor`
  - Checks Python, torch, CUDA, nvidia-smi, and internal modules
  - Exits non-zero if any check fails

- `frx smoke-test`
  - Writes a synthetic input-bound Chrome-format trace
  - Runs the full collect + analysis pipeline end-to-end
  - Verifies bundle layout, zip, and `input_bound` diagnosis

- Analysis pipeline (`analysis.py`)
  - `summarize_run_with_steady_state` — run + steady-state scopes
  - Bottleneck classifier: `input_bound`, `copy_bound`, `sync_bound`, `underutilized_gpu`, `memory_pressure`, `shape_instability`, `launch_bound`, `insufficient_telemetry`
  - `user_facing_bottleneck` — substitutes root-cause label when internal top signal is a symptom (e.g. `underutilized_gpu` → `input_bound`)
  - ROI-ranked recommendation engine with YAML rule + catalog files

- Profiler trace importer
  - Chrome-format `traceEvents` → SDK events
  - `ProfilerStep#N` step boundary detection; falls back to largest avg-duration `user_annotation`
  - `gpu_metrics.csv` (nvidia-smi) → `gpu_sample` events

- Frontend integration
  - `derived/summary.json` scores highest in bundle file picker (120 pts)
  - `user_facing_bottleneck` used for all display labels
  - `summary.json` BundlePill in the file indicator row

### Known gaps at v1

- No `frx upload` command — bundles must be uploaded to the web frontend manually
- Zip analysis not supported in `analyze` (must unzip first)
- Distributed (multi-GPU/multi-node) traces not yet classified
- nvidia-smi sampling is coarse (1s default); per-kernel GPU util requires profiler windows

## v2 — Cloud upload + web integration (planned)

- `frx upload <run-dir>` — push bundle to cloud storage; return a shareable URL
- Web frontend auto-fetches `derived/summary.json` from upload URL
- Auth token management (`frx login`)

## v3 — Richer telemetry (planned)

- Distributed bottleneck detection (NCCL, all-reduce stalls)
- Per-layer memory and compute breakdown from profiler windows
- Continuous profiling mode (rolling window, no manual re-run)
- `shape_instability` recommendations and bucket-input guidance
