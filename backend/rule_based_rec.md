Here’s the step-by-step process for **Phase 2, Step 6: Build a rules-based recommendation layer** for your GPU Performance Autopilot.

---

# Goal

Take the bottleneck classifier output and convert it into:

1. **likely fixes**
2. **clear explanations**
3. **confidence / priority**
4. **safe, repeatable recommendations**

This layer should behave like:

> “We detected input pipeline starvation with high confidence. The most likely fixes are increasing dataloader workers, enabling pinned memory, and prefetching batches.”

The key is: **deterministic, explainable, low-risk recommendations first**.

---

# Step 1: Define the recommendation engine contract

Before writing rules, define exactly what goes in and what comes out.

## Input

The recommendation engine should consume a normalized “diagnostic summary” from Step 5, not raw traces directly.

Example input:

```json
{
  "run_id": "abc123",
  "bottlenecks": [
    {
      "type": "input_pipeline_bound",
      "score": 0.87,
      "evidence": {
        "gpu_active_pct": 42,
        "cpu_wait_pct": 38,
        "host_to_device_gap_ms": 6.4,
        "dataloader_idle_gaps": true
      }
    },
    {
      "type": "kernel_launch_overhead_bound",
      "score": 0.72,
      "evidence": {
        "small_kernel_count": 18423,
        "median_kernel_us": 8.1,
        "launches_per_step": 2900
      }
    }
  ],
  "environment": {
    "framework": "pytorch",
    "distributed": false,
    "gpu_type": "A100",
    "num_gpus": 1,
    "mixed_precision": false
  }
}
```

## Output

Each recommendation should be structured.

Example:

```json
{
  "recommendations": [
    {
      "id": "rec_input_pipeline_num_workers",
      "title": "Increase DataLoader workers",
      "priority": "high",
      "confidence": 0.86,
      "expected_impact": "high",
      "applies_to": ["input_pipeline_bound"],
      "why": [
        "GPU active time is low",
        "CPU wait time is high",
        "Trace shows batch delivery gaps before compute"
      ],
      "actions": [
        "Increase num_workers",
        "Benchmark worker counts 4, 8, 16",
        "Monitor step-time variance and GPU utilization"
      ],
      "risks": [
        "Too many workers can increase CPU contention or RAM pressure"
      ],
      "validation": [
        "Check whether idle gaps shrink",
        "Check whether GPU active % rises"
      ]
    }
  ]
}
```

This schema matters a lot. It turns recommendations into a **product surface**, not just freeform text.

---

# Step 2: Create a recommendation taxonomy

Do not jump straight into rules. First define the universe of recommendation types.

A good initial taxonomy:

## Input pipeline recommendations

* increase dataloader workers
* enable pinned memory
* increase prefetch factor
* move preprocessing off main thread
* cache / tokenize / decode ahead of time
* reduce per-sample Python work
* switch to faster storage or streaming strategy

## Kernel execution recommendations

* fuse small ops
* enable `torch.compile`
* use CUDA Graphs
* batch small operations
* write Triton/custom kernels for hot paths
* reduce Python overhead around launches

## Memory efficiency recommendations

* switch to mixed precision
* change tensor layout
* reduce redundant reads/writes
* improve operator fusion
* use in-place ops where safe
* reduce activation footprint / checkpoint selectively

## Synchronization recommendations

* remove `.item()` in hot path
* avoid frequent `.cpu()` / `.numpy()`
* reduce explicit device syncs
* batch metric logging
* move logging/checkpoint logic off critical path

## Multi-GPU recommendations

* overlap communication with compute
* adjust gradient bucket sizes
* inspect all-reduce frequency
* improve sharding strategy
* rebalance work across GPUs
* reduce synchronization barriers
* use larger per-device batch when possible

## Occupancy / launch config recommendations

* increase batch size
* improve thread/block sizing
* reduce register pressure
* increase per-kernel work granularity

## Memory pressure / fragmentation recommendations

* reduce allocation churn
* reuse buffers
* preallocate workspace
* reduce tensor shape variability
* checkpoint or offload strategically

This becomes your internal **recommendation catalog**.

---

# Step 3: Build a recommendation catalog file

Implement the first version as static config, not code scattered everywhere.

For example:

```yaml
- id: rec_input_pipeline_num_workers
  title: Increase DataLoader workers
  category: input_pipeline
  applies_to:
    - input_pipeline_bound
  prerequisites:
    framework: [pytorch]
  impact: high
  effort: low
  safety: medium
  action_templates:
    - "Increase num_workers and benchmark step time."
    - "Try 4, 8, and 16 workers depending on CPU cores."
  validation_templates:
    - "GPU active % should increase."
    - "Batch-ready gaps should shrink."
  caveats:
    - "Too many workers can increase CPU contention."
```

Why this matters:

* easier to add/edit recommendations
* easier to test
* easier to support multiple frameworks later
* separates **policy** from **engine logic**

---

# Step 4: Define rule inputs and derived signals

Do not write rules directly against dozens of raw metrics. First compute **derived signals**.

Examples:

## Utilization signals

* `low_gpu_activity = gpu_active_pct < 60`
* `high_cpu_wait = cpu_wait_pct > 20`
* `high_h2d_gap = median_h2d_gap_ms > threshold`

## Kernel shape signals

* `many_small_kernels = small_kernel_count > X && median_kernel_us < Y`
* `high_launch_density = launches_per_step > Z`

## Memory behavior signals

* `memory_bound = dram_bw_util_high && sm_util_low_relative`
* `excessive_memory_traffic = bytes_moved_per_flop high`

## Synchronization signals

* `sync_heavy = sync_events_per_step high`
* `host_roundtrip_hot = frequent item/cpu/numpy transfers`

## Distributed signals

* `poor_scaling = multi_gpu_efficiency < threshold`
* `comm_dominant = comm_time_pct > threshold`
* `straggler_gpu = gpu_step_variance high`

## Pressure signals

* `fragmented = alloc_free_churn high`
* `oom_risk = memory_near_capacity && fragmentation high`

These derived signals are what your rules should depend on.

---

# Step 5: Design the rule format

Each rule should be explicit and testable.

A good rule shape:

```json
{
  "id": "rule_input_starvation_01",
  "if": [
    "bottleneck.type == input_pipeline_bound",
    "bottleneck.score >= 0.7",
    "signals.low_gpu_activity == true",
    "signals.high_cpu_wait == true"
  ],
  "then": [
    "rec_input_pipeline_num_workers",
    "rec_input_pipeline_pinned_memory",
    "rec_input_pipeline_prefetch"
  ],
  "priority_boost": 0.2,
  "confidence_formula": "base_bottleneck_score * 0.9",
  "explanation_template": [
    "GPU activity is low while CPU wait is high.",
    "This pattern usually indicates the input pipeline is starving the device."
  ]
}
```

Keep rules:

* simple
* composable
* deterministic
* easy to inspect

Avoid giant nested `if/else` code early on.

---

# Step 6: Start with 10–20 high-value rules only

Do not try to cover everything on day one. Build the first rules around the most obvious and high-ROI cases.

Recommended initial set:

## Input pipeline

1. low GPU activity + high CPU wait
2. low GPU activity + visible batch gaps
3. high host-to-device transfer stalls

## Kernel launch overhead

4. many tiny kernels per step
5. high launch count + low per-kernel duration

## Memory-bound compute

6. high memory bandwidth use + low compute saturation
7. excessive tensor movement / layout conversion overhead

## Synchronization

8. frequent sync events
9. frequent host roundtrips (`.item()`, `.cpu()`, logging)

## Multi-GPU

10. communication time dominates step
11. poor scaling with more GPUs
12. imbalance across devices

## Under-occupancy / batching

13. low occupancy + small batch
14. low arithmetic intensity + underfilled kernels

## Memory pressure

15. allocation churn / fragmentation
16. repeated near-OOM pressure

That is enough for a strong v1.

---

# Step 7: Add recommendation ranking logic

A run may trigger many recommendations. You need ranking so the output feels useful, not noisy.

Rank by:

## 1. Bottleneck score

If the bottleneck classifier is highly confident, recommendations tied to it get boosted.

## 2. Evidence strength

More matching signals = higher priority.

Example:

* low GPU activity only → medium
* low GPU activity + CPU wait + visible gaps → high

## 3. Expected impact

Some fixes tend to matter more:

* dataloader starvation fix: often high impact
* tiny kernel fusion: often high impact
* tiny logging cleanup: maybe medium

## 4. Effort

Prefer high-impact, low-effort fixes near the top.

## 5. Safety

Prefer safer recommendations first.

A simple ranking formula:

```text
recommendation_score =
  0.40 * bottleneck_confidence +
  0.25 * evidence_strength +
  0.20 * expected_impact +
  0.10 * low_effort_bonus +
  0.05 * safety_bonus
```

No ML needed here.

---

# Step 8: Generate explanation text from evidence

This is extremely important.

Every recommendation should answer:

1. **What did we observe?**
2. **What does it likely mean?**
3. **Why this fix?**
4. **How do you verify it worked?**

Example:

> GPU active time is only 42%, while CPU wait accounts for 38% of step time. We also see repeated gaps before kernels begin. This usually means the input pipeline is not preparing batches fast enough. Start by increasing DataLoader workers, enabling pinned memory, and testing prefetch settings.

That is much better than:

> Recommendation: tune dataloader.

---

# Step 9: Attach concrete actions, not vague advice

Each recommendation should include specific actions the user can take.

Bad:

* optimize dataloader

Good:

* benchmark `num_workers` at 4, 8, and 16
* enable `pin_memory=True`
* increase `prefetch_factor`
* move CPU-heavy transforms out of `__getitem__`
* profile batch ready-time before and after changes

The more actionable, the more useful the system becomes.

---

# Step 10: Add “validation instructions” to every recommendation

This makes the product feel rigorous and reduces hallucination risk.

For each recommendation, say how to verify success.

Examples:

## Input pipeline fix validation

* GPU active % rises
* batch gaps shrink
* step time becomes more stable

## Small-kernel fix validation

* launches per step decrease
* average kernel duration increases
* end-to-end step time drops

## Sync fix validation

* host synchronization count drops
* CPU-side stalls shrink
* GPU timeline becomes less fragmented

## Multi-GPU fix validation

* comm/compute overlap improves
* scaling efficiency rises
* step-time skew across ranks decreases

This makes the system feel like engineering, not guesswork.

---

# Step 11: Add guardrails and suppression rules

This is where recommendation quality goes up a lot.

Examples:

## Suppression examples

* Do not recommend more dataloader workers if CPU is already saturated.
* Do not recommend mixed precision if it is already enabled.
* Do not recommend CUDA Graphs if the workload is highly dynamic and shapes vary heavily.
* Do not recommend larger batch size if memory headroom is near zero.
* Do not recommend torch.compile repeatedly if already enabled and graph breaks are severe.
* Do not recommend multi-GPU overlap tactics for single-GPU jobs.

These suppression rules reduce dumb output.

---

# Step 12: Support recommendation bundles

Users usually do not want 14 separate suggestions. Group related fixes.

Example bundle:

## “Input Pipeline Optimization”

* Increase dataloader workers
* Enable pinned memory
* Increase prefetch factor
* Reduce CPU transforms in hot path

Another:

## “Kernel Overhead Reduction”

* Enable `torch.compile`
* Fuse pointwise ops
* Consider Triton for hot sequences
* Use CUDA Graphs if shapes are stable

This makes output much more readable.

---

# Step 13: Build the first recommendation engine in code

Implementation layers:

## Layer A: signal extractor

Converts raw metrics/classifier output into boolean/numeric signals.

## Layer B: rule evaluator

Runs rules against signals and emits candidate recommendations.

## Layer C: ranker

Scores and orders candidates.

## Layer D: explanation generator

Builds human-readable output from templates + evidence.

## Layer E: formatter

Produces UI/API response.

Simple architecture:

```text
trace/profile
   ↓
normalized metrics
   ↓
bottleneck classifier
   ↓
signal extractor
   ↓
rules engine
   ↓
candidate recommendations
   ↓
rank + suppress + bundle
   ↓
final report
```

---

# Step 14: Write golden test cases

This step is mandatory.

Create curated synthetic or real profiles and assert expected outputs.

Examples:

## Case 1: dataloader starvation

Expected top recs:

* increase workers
* pinned memory
* prefetch

Expected non-recs:

* mixed precision
* gradient bucketing

## Case 2: many tiny kernels

Expected top recs:

* torch.compile
* op fusion
* CUDA Graphs

Expected non-recs:

* dataloader tuning

## Case 3: sync-heavy trace

Expected top recs:

* remove `.item()`
* reduce `.cpu()`
* batch logging

Golden tests are how you keep behavior repeatable.

---

# Step 15: Create an evaluation rubric for recommendation quality

You need a way to judge whether the recommendation layer is good.

Use 5 dimensions:

## 1. Correctness

Does the recommendation match the observed bottleneck?

## 2. Specificity

Is it actionable, not generic?

## 3. Explanation quality

Does it clearly connect evidence to advice?

## 4. Safety

Could it mislead the user into harmful changes?

## 5. Priority quality

Are the most important fixes shown first?

Review outputs manually against this rubric before shipping.

---

# Step 16: Add confidence labels carefully

Do not overclaim.

Good labels:

* high confidence
* moderate confidence
* exploratory suggestion

Example:

* “High confidence: input pipeline starvation”
* “Moderate confidence: kernel launch overhead”
* “Exploratory: CUDA Graphs may help if shapes are stable”

This keeps trust high.

---

# Step 17: Separate “diagnosis” from “recommendation”

This is important product design.

Diagnosis:

* “Your job appears input-pipeline bound.”

Recommendation:

* “Increase DataLoader workers, pinned memory, and prefetching.”

Validation:

* “Reprofile and verify that idle gaps shrink.”

This structure is clean and feels trustworthy.

---

# Step 18: Build a small recommendation report UI format

A strong output format:

## Top issue

Input pipeline starvation

## Why we think this

* GPU active time: 42%
* CPU wait time: 38%
* repeated batch delivery gaps

## Recommended actions

1. Increase DataLoader workers
2. Enable pinned memory
3. Increase prefetch factor

## Expected impact

High

## How to verify

* GPU active % rises
* step time drops
* idle gaps shrink

This can later become dashboard cards or CLI output.

---

# Step 19: Track recommendation outcomes

Even before ML, start collecting feedback data.

For each recommendation:

* was it shown?
* was it accepted?
* did the user apply it?
* did performance improve?
* by how much?
* did the bottleneck score drop on next run?

This becomes the future training data for a smarter recommender later.

---

# Step 20: Keep v1 narrow and high precision

The biggest risk is trying to be too broad too early.

For v1, optimize for:

* fewer recommendations
* clearer explanations
* higher precision
* lower hallucination risk

It is much better to say:

> “We found 2 high-confidence fixes.”

than

> “Here are 17 possible optimizations.”

---

# Recommended development order

Here is the practical build order I’d use:

## Phase A: foundations

1. define recommendation schema
2. define recommendation catalog
3. define derived signals
4. define rule format

## Phase B: first engine

5. implement signal extraction
6. implement 10–20 high-value rules
7. implement ranking
8. implement suppression
9. implement explanation templates

## Phase C: quality

10. build golden test cases
11. review outputs manually on real traces
12. tighten thresholds and suppress noisy advice

## Phase D: productization

13. bundle related recommendations
14. expose via API/UI
15. track outcome feedback

---

# What v1 should probably cover first

If I were narrowing to the highest-value first slice, I would start with these 5 recommendation families:

1. **Input pipeline starvation**
2. **Tiny kernel / launch overhead**
3. **Sync-heavy host behavior**
4. **Memory-bound kernels**
5. **Poor multi-GPU scaling**

That is already enough to create a compelling system.

---

# Example of a full v1 recommendation

## Diagnosis

Input pipeline bottleneck

## Confidence

High

## Evidence

* GPU active time is 42%
* CPU wait time is 38%
* visible gaps before compute begins
* host-to-device transfer stalls are elevated

## Recommendations

1. Increase DataLoader workers
2. Enable pinned memory
3. Increase prefetch factor
4. Move expensive preprocessing off the critical path

## Why

The GPU is frequently idle waiting for the next batch. This usually means data preparation or transfer is too slow.

## How to verify

* GPU active time should rise above current baseline
* step-time variance should drop
* idle gaps between batches should shrink

---

# One important design principle

Do **not** frame the engine as:

> “the optimizer knows the fix.”

Frame it as:

> “given this profile pattern, these are the most likely and safest first fixes.”

That keeps the system honest and robust.

---

If you want, I can turn this into the next layer and give you a concrete **JSON/YAML schema for rules + recommendation objects**, along with **10 starter rules for v1**.
