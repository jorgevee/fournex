# Bottleneck Classifier Plan

This file is the working execution plan for the first bottleneck classifier.

The current repository already has a usable first-pass implementation in `python/autopilot_telemetry/analysis.py`. The immediate goal is not to redesign that logic from scratch. The goal is to turn it into a stable, explainable, replayable product path with explicit validation.

## Status

* [x] Phase 1 problem framing captured in the initial design memo.
* [x] Phase 2 first-pass per-step metrics implemented in `python/autopilot_telemetry/analysis.py`.
* [x] Phase 3 first-pass run summary and ranked bottleneck labels implemented.
* [x] Phase 4 common IR bottleneck annotation bridge implemented in `python/autopilot_telemetry/common_ir_analysis.py`.
* [x] Phase 5 golden bottleneck corpus defined.
* [x] Phase 6 classifier regression tests implemented.
* [x] Phase 7 explanation contract hardened for storage and UI consumption.
* [x] Phase 8 initial phase-window and steady-state classification support implemented.
* [x] Phase 9 threshold tuning and taxonomy expansion implemented.

## Product goal for v1

Given one collected run, produce a concise bottleneck report that answers:

* What is the dominant bottleneck?
* What evidence supports that diagnosis?
* How confident should we be?
* What should the user try next?

The v1 target is not perfect hardware truth. The target is a deterministic diagnosis layer that is transparent, replayable, and useful on real traces.

## Current implementation baseline

The current rules layer already derives:

* per-step timing slices such as DataLoader wait, H2D copy, forward, backward, optimizer, and sync wait
* run-level summaries such as average GPU utilization, memory pressure peak ratio, shape volatility, and exported profiler windows
* ranked labels: `input_bound`, `copy_bound`, `sync_bound`, `underutilized_gpu`, `memory_pressure`, `shape_instability`, and `launch_bound`

This means the highest-leverage next work is:

* validate the current labels with replayable cases
* harden the result contract
* only then widen the taxonomy or scoring sophistication

## Output contract

The classifier output should stabilize around one primary diagnosis plus optional supporting diagnoses and evidence.

Example shape:

```json
{
  "primary_bottleneck": "input_bound",
  "secondary_bottlenecks": ["copy_bound"],
  "confidence": {
    "level": "medium",
    "score": 0.78
  },
  "evidence": {
    "avg_dataloader_fraction": 0.31,
    "avg_h2d_fraction": 0.08,
    "average_gpu_utilization_pct": 41.2
  },
  "why": [
    "DataLoader wait consumes a large fraction of step time",
    "GPU utilization stays low during steady-state steps"
  ],
  "recommendations": [
    "Increase DataLoader workers",
    "Enable pinned memory",
    "Prefetch batches"
  ],
  "classifier_version": "0.1.0"
}
```

The current implementation now emits a thin version of this contract through `diagnosis` in `summarize_run(...)` and persisted summary artifacts. Remaining work is mostly around versioning, richer contradiction handling, and future API/UI consumers.

## Execution strategy

### Phase 5: Define a validating corpus

Start with a small, explicit set of replayable cases. These should be synthetic or captured traces where the expected dominant label is not ambiguous.

Required first cases:

* `input_bound`
* `copy_bound`
* `sync_bound`
* `launch_bound`
* `memory_pressure`

Nice-to-have early cases:

* `underutilized_gpu`
* `shape_instability`
* `no_clear_dominant_bottleneck`

Deliverables:

* fixture format for replayable event streams
* one golden case file per expected bottleneck
* short notes on why each case should classify that way

Acceptance criteria:

1. Each golden case can be loaded without requiring live telemetry collection.
2. Each golden case has a clear expected primary label.
3. At least one mixed-signal case exists to exercise ambiguity handling.

Implemented artifacts:

* `tests/python/analysis_bottleneck_golden_cases.py`
* `tests/python/test_analysis_bottlenecks.py`
* `tests/python/test_storage_summary.py`

Current Phase 5 corpus coverage:

* implemented primary golden cases: `input_bound`, `copy_bound`, `sync_bound`, `launch_bound`, `memory_pressure`
* implemented additional primary golden case: `shape_instability`
* implemented ambiguity and degradation cases: `mixed_signal`, `sparse_telemetry`
* implemented persisted-summary checks for the `diagnosis` contract

Remaining Phase 5 follow-ups:

* decide whether `underutilized_gpu` should get its own standalone golden case instead of being covered only through mixed and launch-adjacent scenarios
* add short inline notes per fixture explaining the intended diagnosis and known caveats

### Phase 6: Add regression tests around the existing rules

Treat the classifier as product logic, not helper glue.

Test classes:

* positive classification tests
* negative classification tests
* ordering tests for ranked outputs
* missing telemetry tests
* mixed-signal tests

Suggested test file:

* `tests/python/test_analysis_bottlenecks.py`

Acceptance criteria:

1. The current rules in `python/autopilot_telemetry/analysis.py` are covered by replay-based tests.
2. The tests lock expected labels and core evidence fields.
3. Rule changes that alter classifications require explicit fixture or threshold updates.

Current status:

* replay-based classifier tests are implemented and passing in `tests/python/test_analysis_bottlenecks.py`
* persisted-summary tests are implemented and passing in `tests/python/test_storage_summary.py`
* confidence-level expectations are now covered for representative single-signal and mixed-signal cases

### Phase 7: Harden the explanation contract

The current output is a ranked list of `{label, score, evidence}` items. That is enough internally, but it is still too thin for downstream UI or API use.

Add:

* primary versus secondary diagnosis selection
* confidence level derivation
* explanation bullets
* contradiction or ambiguity notes
* recommendation mapping separate from diagnosis logic
* classifier and ruleset version fields

Acceptance criteria:

1. A stored summary artifact can be rendered into a user-facing report without reinterpreting raw traces.
2. Explanation content is deterministic from the same event stream.
3. Recommendation mapping remains separate from raw rule scoring.

Current status:

* `summarize_run(...)` now returns both raw ranked `bottlenecks` and a user-facing `diagnosis` object
* `persist_run_summary(...)` stores the `diagnosis` contract in the derived summary artifact
* `schemas/derived_metrics.md` and `schemas/derived_summary_example.json` document the current summary shape

Remaining Phase 7 follow-ups:

* add classifier or ruleset version fields
* decide whether `copy_bound` and similar labels remain internal names or get aliased to more user-facing wording
* refine contradiction bullets so they communicate exclusions, not only alternate triggered labels

### Phase 8: Add phase-window classification

The current logic is run-level only. That is enough for the first implementation, but bottlenecks often move during warmup, steady-state, evaluation, and checkpoint windows.

Scope:

* classify steady-state windows first
* keep run-level summary classification
* avoid startup and checkpoint noise by default

Acceptance criteria:

1. The reducer can classify a selected subset of steps or windows.
2. Run-level and steady-state outputs can disagree without breaking the result contract.
3. The default report clearly states which scope was classified.

Current status:

* `summarize_step_scope(...)` supports classification of an explicit subset of step IDs
* `select_steady_state_step_ids(...)` supports explicit steady-state selection with warmup skipping and optional last-`K` trimming
* `summarize_steady_state(...)` produces a steady-state scoped summary
* `summarize_run_with_steady_state(...)` returns both whole-run and steady-state summaries in one artifact
* the combined artifact now reports selector metadata and `scope_comparison` so run-versus-steady-state diagnosis disagreement is explicit
* `persist_run_with_steady_state_summary(...)` stores the combined Phase 8 artifact
* `schemas/derived_run_with_steady_state_example.json` and `schemas/derived_metrics.md` document the current combined report shape

Remaining Phase 8 follow-ups:

* decide whether `skip_first_n = 2` should remain the default steady-state policy or become configurable per workload type
* add support for named scopes beyond `run` and `steady_state`, such as `warmup`, `eval`, or checkpoint windows
* decide whether the storage layer should persist both single-scope and combined artifacts by default or keep the richer Phase 8 report opt-in
* add a small source-of-truth helper or schema note for `scope_comparison` semantics so downstream consumers know how to interpret disagreement

### Phase 9: Tune thresholds and expand the taxonomy

Do this only after the golden corpus and explanation contract exist.

Candidate additions:

* `host_device_transfer_overhead` as a more user-facing replacement or alias for `copy_bound`
* `distributed_communication_bottleneck`
* `memory_bandwidth_bound`
* `batch_size_too_small`
* `insufficient_telemetry`
* `mixed_bottleneck`

Acceptance criteria:

1. New labels are backed by explicit features and tests.
2. Existing labels do not silently regress.
3. Any label rename is reflected consistently in common IR, schemas, and examples.

## First validating milestone

The first milestone should be narrow:

**Given replayed event streams for five golden cases, the backend produces a stable ranked bottleneck result with expected primary label and evidence fields.**

Build order:

1. Define golden event fixtures.
2. Add regression tests for `summarize_run(...)`.
3. Freeze the current label set and thresholds in the tests.
4. Only after the tests pass, reshape the output contract for confidence and recommendations.

This is the best first proving step because it validates the current implementation before we make it more sophisticated.

Status:

* completed

## File ownership and likely touch points

Primary code paths:

* `python/autopilot_telemetry/analysis.py`
* `python/autopilot_telemetry/common_ir_analysis.py`
* `python/autopilot_telemetry/storage.py`

Primary docs and schemas:

* `schemas/derived_metrics.md`
* `schemas/common_ir_examples.md`
* `docs/common_ir.md`

Primary tests to add:

* `tests/python/test_analysis_bottlenecks.py`
* `tests/python/test_storage_summary.py`
* fixture helpers under `tests/python/`

## Things to avoid

Do not:

* widen the taxonomy before validation exists
* jump to ML-based classification
* couple recommendations directly into threshold logic
* classify startup or checkpoint noise as the dominant steady-state issue
* rename labels casually without updating schemas and common IR examples

## Immediate next actions

1. Decide whether `underutilized_gpu` should get its own standalone golden fixture instead of being covered only through mixed and launch-adjacent scenarios.
2. Add short inline notes per fixture explaining the intended diagnosis and known caveats.
3. Decide whether the current default steady-state selector policy should remain `skip_first_n = 2`, `last_k = null`, or become configurable by workload type.
4. Decide whether current labels like `copy_bound` are internal-only names or user-facing contract names.
5. Add classifier and ruleset version fields to the persisted `diagnosis` contract.
6. Decide whether the combined run-plus-steady-state artifact should become the default persisted derived summary or remain a separate opt-in report.
