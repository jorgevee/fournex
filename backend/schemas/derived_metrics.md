# Derived Metrics

Phase 9 defines the first derived layer over the raw event stream.

## Per-step metrics

The initial reducer computes these fields per step:

| Field | Source events |
| --- | --- |
| `step_wall_time_ns` | `step_end.duration_ns` |
| `dataloader_wait_time_ns` | `dataloader_span` with `stage = next` |
| `h2d_copy_time_ns` | `memcpy_span` with `copy_kind = h2d` |
| `forward_time_ns` | `phase_span` with `phase_name = forward` |
| `backward_time_ns` | `phase_span` with `phase_name = backward` |
| `optimizer_time_ns` | `phase_span` with `phase_name = optimizer` |
| `sync_wait_time_ns` | `sync_wait.duration_ns` |
| `batch_size` | `shape_snapshot.payload.batch_size` |
| `sequence_length` | `shape_snapshot.payload.sequence_length` |
| `shape_signature` | normalized `shape_snapshot.payload.shapes` |
| `profiler_windows_exported` | `profiler_window` with `window_state = exported` |

## Per-run summary

The initial run summary computes:

| Field | Method |
| --- | --- |
| `average_gpu_utilization_pct` | average of `gpu_sample.payload.utilization_gpu_pct` |
| `average_memory_utilization_pct` | average of `gpu_sample.payload.utilization_mem_pct` |
| `throughput_steps_per_sec` | completed steps divided by total step wall time |
| `memory_pressure_peak_ratio` | max `memory_used_bytes / memory_total_bytes` across `gpu_sample` |
| `utilization_instability_pct` | max minus min GPU utilization |
| `step_time_avg_ns` | mean completed step time |
| `step_time_max_ns` | max completed step time |
| `shape_volatility_ratio` | fraction of adjacent steps whose shape signature changed |
| `profiler_windows_exported` | total exported profiler windows |
| `dominant_stall_type` | dominant aggregate among input, copy, sync, and compute buckets |

## Notes

* This layer is deterministic and computed only from emitted events.
* Phase 10 adds a transparent rules layer on top of these metrics.
* The native writer currently serializes payload values as strings, so reducers should tolerate both numeric and string numeric inputs.

## Bottleneck rules

The first rules engine emits ranked labels with evidence:

| Label | First-pass condition |
| --- | --- |
| `input_bound` | average DataLoader wait fraction >= 0.20 |
| `copy_bound` | average H2D copy fraction >= 0.15 |
| `sync_bound` | average sync wait fraction >= 0.10 |
| `underutilized_gpu` | average GPU utilization > 0 and < 35% |
| `memory_pressure` | peak memory usage ratio >= 0.90 |
| `shape_instability` | shape volatility ratio >= 0.30 |
| `launch_bound` | profiler windows exported, GPU utilization low, and input/copy/sync fractions all low |

Each classification carries:

* `label`
* `score`
* `evidence`

## Diagnosis result

The stored run summary now includes a thin diagnosis contract built from the ranked bottleneck list.

Fields:

* `primary_bottleneck`
* `secondary_bottlenecks`
* `confidence`
* `evidence`
* `why`
* `why_not_others`
* `recommendations`
* `dominant_stall_type`

Notes:

* `bottlenecks` remains the raw ranked rules output for tuning and debugging.
* `diagnosis` is the more stable user-facing summary shape for downstream API or UI use.

## Example artifact

A concrete stored summary example lives in `schemas/derived_summary_example.json`.

This example shows:

* a minimal `per_step` section
* the aggregate `run_summary`
* the raw ranked `bottlenecks` list
* the user-facing `diagnosis` object derived from that ranking

## Combined run plus steady-state artifact

Phase 8 also supports a combined report that carries both whole-run and steady-state summaries in one payload.

Top-level fields:

* `event_count`
* `step_count`
* `selector`
* `scope_comparison`
* `run`
* `steady_state`

Notes:

* `run` and `steady_state` each use the same summary shape as the single-scope artifact.
* The combined artifact is useful when the full run is mixed or ambiguous but the steady-state slice is still decisive.
* The storage helper for this artifact is `persist_run_with_steady_state_summary(...)`.
* `selector.policy` indicates whether the steady-state parameters came from backend defaults or explicit caller input.
* `scope_comparison` makes run-versus-steady-state diagnosis disagreement explicit for downstream consumers.

Example:

* `schemas/derived_run_with_steady_state_example.json`
