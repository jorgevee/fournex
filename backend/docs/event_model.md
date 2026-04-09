# Event Model

Phase 1 defines a single append-only event stream shared by Python and the native engine.

## Phase 1 goals

The v1 event model needs to be:

1. Stable enough that Python and C++ can emit the same record format.
2. Simple enough to serialize to JSONL without special translation layers.
3. Structured enough that Phase 9 derived metrics can be computed without heuristics over free-form text.

## Event envelope

Every event record uses the same top-level fields:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `schema_version` | string | yes | Starts at `0.1.0`. |
| `event_id` | string | yes | UUID or other per-event unique id. |
| `timestamp_ns` | integer | yes | Monotonic or normalized timestamp in nanoseconds. |
| `pid` | integer | yes | Process id of the emitter. |
| `tid` | integer | yes | Thread id of the emitter. |
| `job_id` | string | yes | Logical job name or stable job identifier. |
| `run_id` | string | yes | Unique id for one run. |
| `event_type` | string | yes | One of the Phase 1 event types below. |
| `event_source` | string | yes | `python_sdk` or `native_engine`. |
| `gpu_id` | integer or null | no | Set for GPU-scoped records. |
| `step_id` | integer or null | no | Set for step-scoped records. |
| `span_id` | string or null | no | Used for span-style events. |
| `parent_span_id` | string or null | no | Used when spans are nested. |
| `duration_ns` | integer or null | no | Required for completed spans. |
| `level` | string | yes | `debug`, `info`, `warning`, or `error`. |
| `payload` | object | yes | Event-specific typed content. |

## Event types

Phase 1 locks the first eleven event types:

| Event type | Source | Purpose |
| --- | --- | --- |
| `gpu_sample` | native | Coarse device utilization and memory samples. |
| `step_start` | python | Marks the beginning of a training or inference step. |
| `step_end` | python | Marks the end of a training or inference step. |
| `phase_span` | python | Timed semantic regions such as `forward` or `backward`. |
| `dataloader_span` | python | Time spent waiting for the next batch or collating it. |
| `memcpy_span` | python | Host-to-device or device-to-host copy region. |
| `shape_snapshot` | python | Batch shape and workload metadata. |
| `sync_wait` | python | Explicit host synchronization or blocking wait. |
| `profiler_window` | python | Metadata about sampled `torch.profiler` windows. |
| `system_info` | native | One-time run metadata about host and devices. |
| `warning_annotation` | python or native | Human-readable anomalies attached to the trace. |

## Payload contracts

### `gpu_sample`

Required payload fields:

| Field | Type |
| --- | --- |
| `utilization_gpu_pct` | number |
| `utilization_mem_pct` | number |
| `memory_used_bytes` | integer |
| `memory_total_bytes` | integer |

Optional payload fields:

* `temperature_c`
* `power_w`
* `sm_clock_mhz`
* `mem_clock_mhz`

### `step_start`

Required payload fields:

* `step_kind`

Optional payload fields:

* `global_step`
* `epoch`
* `mode`

### `step_end`

Required payload fields:

* `step_kind`
* `status`

Optional payload fields:

* `global_step`
* `throughput_items_per_sec`
* `loss`

### `phase_span`

Required payload fields:

* `phase_name`

Optional payload fields:

* `device`
* `stream_id`

### `dataloader_span`

Required payload fields:

* `stage`

Optional payload fields:

* `num_workers`
* `prefetch_factor`
* `pinned_memory`
* `batch_size`

### `memcpy_span`

Required payload fields:

* `copy_kind`

Optional payload fields:

* `src_device`
* `dst_device`
* `bytes`
* `non_blocking`

### `shape_snapshot`

Required payload fields:

* `batch_size`
* `shapes`

Optional payload fields:

* `sequence_length`
* `dtypes`
* `model_name`
* `precision_mode`
* `is_training`

### `sync_wait`

Required payload fields:

* `wait_kind`

Optional payload fields:

* `reason`
* `device`

### `profiler_window`

Required payload fields:

* `window_state`

Optional payload fields:

* `start_step`
* `end_step`
* `trace_path`
* `recorded_ops`

### `system_info`

Required payload fields:

* `hostname`
* `platform`

Optional payload fields:

* `cpu_count`
* `gpu_count`
* `driver_version`
* `cuda_runtime_version`
* `python_version`
* `pytorch_version`
* `devices`

### `warning_annotation`

Required payload fields:

* `code`
* `message`

Optional payload fields:

* `evidence`
* `suggested_action`

## Conventions

* `timestamp_ns` is recorded in nanoseconds.
* `duration_ns` is only set for closed spans and completed step events.
* `payload` keys use `snake_case`.
* Numeric percentages stay in the `0` to `100` range.
* Byte values use raw bytes, not MiB strings.
* Human text belongs in `warning_annotation.payload.message`, not in generic payload fields on other event types.

## JSONL record shape

The writer should emit one event per line with no outer array. Example:

```json
{
  "schema_version": "0.1.0",
  "event_id": "evt-000001",
  "timestamp_ns": 1712345678901234567,
  "pid": 12345,
  "tid": 12345,
  "job_id": "bert-train",
  "run_id": "run-20260407-001",
  "event_type": "phase_span",
  "event_source": "python_sdk",
  "gpu_id": 0,
  "step_id": 42,
  "span_id": "span-backward-42",
  "parent_span_id": "span-step-42",
  "duration_ns": 8432100,
  "level": "info",
  "payload": {
    "phase_name": "backward",
    "device": "cuda:0"
  }
}
```

## Phase 1 exit criteria

Phase 1 is complete when:

1. `tel_plan.md` defines the envelope, event set, and acceptance criteria.
2. `schemas/event_schema.json` validates the shared top-level contract and event-specific payload requirements.
3. Python and C++ stubs use the same event names and field names.
4. JSONL output can contain at least `system_info`, `gpu_sample`, `step_start`, `step_end`, and `phase_span` without ad hoc fields.
