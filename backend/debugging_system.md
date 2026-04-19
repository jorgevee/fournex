The core idea is: when something goes wrong, you want to answer three questions fast:
what failed, where it failed, and why it failed.

A good system usually has these layers:

1. Structured logging
Do not rely on random print statements. Log in a consistent shape.

Include fields like:

timestamp
service name
environment
request ID / trace ID
user ID or tenant ID when appropriate
operation name
severity
error code
stack trace

Good:

{
  "level": "error",
  "service": "payments-api",
  "trace_id": "a12f...",
  "request_id": "req_456",
  "operation": "charge_card",
  "user_id": "u_123",
  "error_code": "PAYMENT_GATEWAY_TIMEOUT",
  "message": "Gateway request timed out",
  "duration_ms": 3021
}

Bad:

something broke in payments

2. Correlation IDs and trace IDs
Every request should get an ID at the edge, then that ID should travel through all services, jobs, queues, and DB calls.

That lets you reconstruct a full path like:
frontend -> API gateway -> auth service -> order service -> payment service

Without this, distributed debugging becomes painful.

3. Distributed tracing
For multi-service systems, use tracing, not just logs.

Tracing shows:

which service was slow
where retries happened
where time was spent
which downstream call failed

Think of logs as detailed notes, and traces as the map.

Common approach:

create a root span per request
create child spans for DB queries, HTTP calls, cache calls, queue publish/consume, etc.
attach status and error info to spans


claude - 
  Here is what the system currently has and what is missing across your three layers:

  ---
  Layer 1 — Structured logging

  Already done in sdk.py::make_event(): every event carries event_id, timestamp_ns, pid, tid, run_id, job_id (service name), step_id, span_id, parent_span_id, level, and event_type. The warning_annotation
  event type already exists.

  What is missing:
  - No error_code field for machine-readable error classification
  - No structured warning events when the SDK itself is misconfigured (e.g., no GPU samples collected, native layer unavailable)

  Layer 2 — Correlation IDs

  Already done: run_id travels through all events, and span_id/parent_span_id support parent-child relationships at the event level. step_id is the correlation key within a run.

  What is missing:
  - The diagnosis output answers what but not where exactly — it does not say which specific step IDs were the worst offenders for the detected bottleneck. A user reading primary_bottleneck: "input_bound" with
   confidence 0.78 cannot currently answer "which 3 steps should I look at first in the raw trace?"

  Layer 3 — Distributed tracing / near-miss visibility

  Already partial: common_ir_analysis.py produces communication_bound and compute_bound step-level annotations. The classifier produces a ranked list but does not expose which rules almost fired — so a user
  cannot tell whether sync_bound was nowhere close or just barely below threshold.

  What is missing:
  - near_miss_rules in the diagnosis: labels that were close to threshold but did not fire, with how far off they were

  ---
  Proposed implementation — three additions

  1. worst_steps per classification (analysis.py)

  For each label that fires, attach the top-3 step IDs where that signal was most severe. For example, input_bound would include the 3 steps with the highest dataloader fraction. This directly answers where.

  {
    "label": "input_bound",
    "score": 0.325,
    "evidence": { "avg_dataloader_fraction": 0.325 },
    "worst_steps": [
      {"step_id": 14, "value": 0.51},
      {"step_id": 23, "value": 0.44},
      {"step_id": 8,  "value": 0.39}
    ]
  }

  2. near_miss_rules in diagnosis (analysis.py)

  After classification, compute which rules did not fire but came within 50% of their threshold. Report them with how close they were. This answers why not in a concrete way.

  "near_miss_rules": [
    { "label": "sync_bound", "threshold": 0.10, "observed": 0.07, "gap": 0.03 },
    { "label": "copy_bound", "threshold": 0.15, "observed": 0.09, "gap": 0.06 }
  ]

  3. SDK health warning events (sdk.py)

  After init(), add a deferred health check that fires after the first step completes. If no GPU samples were collected, emit a warning_annotation event with a structured error_code. This makes the
  "insufficient_telemetry" case self-reporting before the user even runs the classifier.

  {
    "event_type": "warning_annotation",
    "level": "warning",
    "payload": {
      "error_code": "NO_GPU_SAMPLES",
      "message": "No GPU utilization samples were recorded after step 1. Check that NVML is accessible.",
      "run_id": "run-abc123"
    }
  }

  ---
  What this does not include (intentionally)

  - No distributed tracing library integration (OpenTelemetry, Jaeger) — the existing span_id/parent_span_id model already covers the single-machine multi-phase case this system targets
  - No log aggregation backend — the JSONL trace format already works with any structured log ingestor
  - No schema changes to existing stored artifacts — worst_steps and near_miss_rules are additive

  ---
  Should I implement all three, or start with a specific one? worst_steps is the highest-value change (directly answers where with no design ambiguity). near_miss_rules is second. The SDK health event is the
  most self-contained.
