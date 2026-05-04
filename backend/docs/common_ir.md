# Common IR

The Common IR is the normalized in-memory representation produced by the analysis pipeline. All downstream logic — bottleneck classification, recommendation engine, CLI report, web frontend — operates on this shape rather than on raw SDK events or Chrome trace events.

## Entry point

```python
from fournex.analysis import summarize_run_with_steady_state

summary = summarize_run_with_steady_state(events)
```

`events` is a list of SDK event dicts (from `raw/trace.jsonl` or reconstructed from a Chrome-format profiler trace).

## Top-level summary shape

```json
{
  "event_count": 28,
  "step_count": 4,
  "selector": { "policy": "default", "skip_first_n": 2, "last_k": null },
  "run": { ... },
  "steady_state": { ... },
  "scope_comparison": {
    "diagnosis_changed": true,
    "run_primary_bottleneck": null,
    "steady_state_primary_bottleneck": "copy_bound"
  }
}
```

Both `run` and `steady_state` are scope objects (see below). `selector` records which steps were excluded from the steady-state window.

## Scope object

```json
{
  "event_count": 28,
  "step_count": 4,
  "per_step": [ ... ],
  "run_summary": { ... },
  "bottlenecks": [ ... ],
  "diagnosis": { ... },
  "scope": { "name": "steady_state", "step_ids": [3, 4] }
}
```

### `per_step`

One record per completed training step, derived from SDK span events:

| Field | Type | Description |
| --- | --- | --- |
| `step_id` | int | Step sequence number |
| `status` | string | `ok` or `unknown` |
| `step_kind` | string or null | `train`, `eval`, etc. |
| `step_wall_time_ns` | int | Total step wall time |
| `dataloader_wait_time_ns` | int | Time blocked in DataLoader `__next__` |
| `h2d_copy_time_ns` | int | Host-to-device copy time |
| `forward_time_ns` | int | Forward pass duration |
| `backward_time_ns` | int | Backward pass duration |
| `optimizer_time_ns` | int | Optimizer step duration |
| `sync_wait_time_ns` | int | Explicit device synchronization time |
| `gpu_active_fraction_proxy` | float | `(fwd + bwd + opt) / step_wall_time` |
| `shape_signature` | string or null | Sorted `key:shape` pairs |
| `shape_changed` | bool | Whether shape changed from previous step |
| `batch_size` | int | From `shape_snapshot` event |
| `sequence_length` | int or null | From `shape_snapshot` event |
| `profiler_windows_exported` | int | Number of profiler window exports |

### `run_summary`

Aggregated run-level metrics:

| Field | Description |
| --- | --- |
| `average_gpu_utilization_pct` | Mean nvidia-smi GPU utilization across samples |
| `average_memory_utilization_pct` | Mean nvidia-smi memory utilization |
| `throughput_steps_per_sec` | Steps / total wall time |
| `memory_pressure_peak_ratio` | Peak `memory_used / memory_total` |
| `utilization_instability_pct` | `max(util) - min(util)` across samples |
| `step_time_avg_ns` | Mean step wall time |
| `step_time_max_ns` | Max step wall time |
| `shape_volatility_ratio` | Fraction of consecutive steps with different shapes |
| `dominant_stall_type` | `input_bound`, `copy_bound`, `sync_bound`, `compute_bound`, or `unknown` |
| `profiler_windows_exported` | Total profiler windows exported across all steps |

### `bottlenecks`

Sorted list (descending score) of detected bottleneck objects:

```json
{
  "label": "input_bound",
  "score": 0.825,
  "evidence": { "avg_dataloader_fraction": 0.825, "dominant_stall_type": "input_bound" },
  "worst_steps": [ { "step_id": 1, "value": 0.91 }, ... ]
}
```

Possible labels: `input_bound`, `copy_bound`, `sync_bound`, `underutilized_gpu`, `memory_pressure`, `shape_instability`, `launch_bound`, `insufficient_telemetry`.

### `diagnosis`

```json
{
  "primary_bottleneck": "underutilized_gpu",
  "user_facing_bottleneck": "input_bound",
  "secondary_bottlenecks": ["input_bound"],
  "confidence": { "level": "high", "score": 0.88, "reason": "..." },
  "evidence": { ... },
  "why": [ "Average DataLoader wait fraction is 0.825.", "..." ],
  "why_not_others": [ "copy_bound also triggered with score 0.180." ],
  "recommendations": [ ... ],
  "recommendation_bundles": [ ... ],
  "dominant_stall_type": "input_bound",
  "classifier_version": "0.2.0"
}
```

`primary_bottleneck` is the internal top-scoring signal. `user_facing_bottleneck` is the root-cause label shown to users — it replaces symptom labels (`underutilized_gpu`) with the underlying stall type (`input_bound`, `copy_bound`, `sync_bound`, `launch_bound`) when one is present.

### `recommendations`

Each recommendation object:

| Field | Description |
| --- | --- |
| `id` | Stable catalog ID (e.g. `rec_input_num_workers`) |
| `title` | Short action title |
| `priority` | `high`, `medium`, or `low` (from ROI score) |
| `score` / `roi_score` | 0–1 composite score |
| `tier` | `try_now`, `next`, or `advanced` |
| `confidence` | Bottleneck score that triggered this recommendation |
| `expected_impact` | `high`, `medium`, or `low` |
| `effort` | `config`, `low`, `medium`, `high`, or `custom_cuda` |
| `risk` | `low`, `medium`, or `high` |
| `category` | `input_pipeline`, `copy`, `synchronization`, etc. |
| `why` | Explanation sentence from the matching rule |
| `why_ranked` | List of ranking reason strings |
| `actions` | Ordered list of concrete implementation steps |
| `validation` | How to verify the fix worked |
| `risks` | Known caveats |
| `guardrails_applied` | e.g. `low_confidence_demoted`, `dependency_ordered` |
| `triggered_by` | Rule ID that matched |

## Source normalization

Both input sources are normalized to the same SDK event list before analysis:

| Source | Normalization path |
| --- | --- |
| `raw/trace.jsonl` | Read directly — already SDK events |
| `profiler/profiler_trace.json` | `_events_from_profiler_bundle` → Chrome `traceEvents` → SDK events |
| `gpu_metrics.csv` | `_gpu_metrics_csv_to_sdk_events` → `gpu_sample` SDK events |

The IR itself is source-agnostic once events are normalized.
