# Architecture

## Overview

`frx` is a CLI tool that wraps a training workload, collects GPU telemetry, and produces a self-contained run bundle. The bundle can be analyzed locally or uploaded to the web frontend for interactive diagnosis.

```
workload process
    └── autopilot_telemetry SDK (Python)
            └── raw/trace.jsonl  (SDK events, JSONL)

frx collect (host process)
    ├── nvidia-smi poller   → gpu_metrics.csv
    ├── workload subprocess → optional_logs.txt
    ├── artifact import     → profiler/profiler_trace.json (Chrome JSON)
    └── analysis pipeline   → derived/summary.json
```

## Components

### CLI (`autopilot_telemetry/cli.py`)

Four subcommands:

| Command | Purpose |
| --- | --- |
| `collect` | Wrap and run a workload; produce a run bundle |
| `analyze` | Load a bundle and print a diagnosis report |
| `doctor` | Check that all runtime dependencies are present |
| `smoke-test` | Write a synthetic trace and verify the end-to-end pipeline |

### SDK (`autopilot_telemetry/sdk.py`, `profiler.py`)

Instruments a running PyTorch workload. Emits structured events to `raw/trace.jsonl` via env-var path injection (`FRX_RAW_TRACE_PATH`). Events cover step boundaries, DataLoader waits, H2D copies, phase spans (forward/backward/optimizer), sync waits, and shape snapshots.

### Analysis pipeline (`autopilot_telemetry/analysis.py`)

Pure-Python, no GPU required. Accepts the SDK event stream or a reconstructed event list from a Chrome-format profiler trace. Produces a `summary` dict with:

- `per_step` — per-step timing breakdown
- `run_summary` — aggregated run-level metrics
- `bottlenecks` — scored list of detected bottleneck types
- `diagnosis` — primary bottleneck, confidence, evidence, and recommendations

The `summarize_run_with_steady_state` entry point produces both a `run` scope (all steps) and a `steady_state` scope (steps after the warm-up window).

### Bottleneck classifier (`analysis.py:classify_bottlenecks`)

Scores these bottleneck types from the step timing data:

| Label | Signal |
| --- | --- |
| `input_bound` | DataLoader wait fraction ≥ 20% of step time |
| `copy_bound` | H2D copy fraction ≥ 15% of step time |
| `sync_bound` | Sync wait fraction ≥ 10% of step time |
| `underutilized_gpu` | GPU utilization < 35% (symptom — surfaced as root cause via `user_facing_bottleneck`) |
| `memory_pressure` | Peak memory ratio ≥ 90% |
| `shape_instability` | Shape volatility ratio ≥ 30% |
| `launch_bound` | Low GPU util + profiler windows exported, no dominant stall |
| `insufficient_telemetry` | No timing data and no GPU utilization samples |

`build_diagnosis_result` emits both `primary_bottleneck` (internal top signal) and `user_facing_bottleneck` (root-cause label, substituted when the internal top signal is a symptom like `underutilized_gpu`).

### Recommendation engine (`recommendations/engine.py`)

Rule + catalog system. Rules (YAML) match bottleneck labels and signal values; the catalog (YAML) defines actionable recommendations with effort, risk, and speedup estimates. ROI scores rank recommendations.

### Profiler trace importer (`cli.py:_events_from_profiler_bundle`)

Converts Chrome-format profiler traces (`traceEvents`, `ph=X`) into SDK events. Step boundaries are detected from `ProfilerStep#N` annotations; if absent, the heuristic falls back to the largest average-duration repeating `user_annotation`. Only `user_annotation` events are used for step-level timing (excludes `cpu_op`, `python_function` internals to avoid double-counting).

## Run bundle layout

```
runs/
  run-<id>/
    metadata.json          # run metadata, artifact manifest, warnings
    manifest.json          # file list, limited-data flag
    run_config.yaml        # collector config + detected environment
    gpu_metrics.csv        # nvidia-smi samples (util, memory, clock)
    optional_logs.txt      # combined stdout/stderr from workload
    raw/
      trace.jsonl          # SDK event stream (JSONL, one event per line)
    derived/
      summary.json         # pre-analyzed output of summarize_run_with_steady_state
    profiler/
      profiler_trace.json  # Chrome-format torch.profiler trace (imported)
  run-<id>.zip             # all of the above, zipped
```

## Data flow: collect

1. `collect` writes `run_config.yaml` and injects env vars into the workload process.
2. A background thread polls `nvidia-smi` at `--sample-interval-ms` (default 1000 ms) into `gpu_metrics.csv`.
3. The workload subprocess runs; stdout/stderr is tee'd to `optional_logs.txt`.
4. After the workload exits, `_import_workload_bundle_artifacts` copies artifacts from `--artifact-dir` (default `./frx-job-run`) into the bundle.
5. `_generate_derived_summary_from_trace` runs the analysis pipeline over `raw/trace.jsonl` and writes `derived/summary.json`.
6. If no SDK trace exists, `_generate_derived_summary_from_profiler_bundle` attempts the same from the imported Chrome-format trace.
7. `metadata.json` and `manifest.json` are written. The bundle is zipped.

## Data flow: analyze

1. `analyze <run-dir>` calls `_load_or_generate_summary`, which checks in order:
   - `derived/summary.json` (pre-analyzed, preferred)
   - `raw/trace.jsonl` (re-analyzes on the fly)
   - `profiler/profiler_trace.json` + `gpu_metrics.csv` (imports and analyzes)
2. `_print_analysis_report` renders verdict, evidence, performance snapshot, and top recommendations to stdout. Use `--json` for machine-readable output.

## Environment injection

`collect` sets these env vars on the workload subprocess:

| Variable | Value |
| --- | --- |
| `FRX_RUN_ID` | generated run ID |
| `FRX_JOB_NAME` | `--name` value |
| `FRX_OUTPUT_DIR` | run directory path |
| `FRX_RAW_TRACE_PATH` | `raw/trace.jsonl` absolute path |
| `FRX_DERIVED_SUMMARY_PATH` | `derived/summary.json` absolute path |
| `FRX_AUTO_PERSIST` | `1` |
| `FRX_SAMPLE_INTERVAL_MS` | sampling interval |
