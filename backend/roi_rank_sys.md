Here’s a practical step-by-step plan for **Phase 2, Step 7: Rank recommendations by expected ROI**.

The goal is to turn a pile of detected fixes into an ordered list of **what the user should do first**.

You want the system to answer:

**“Which actions are most worth doing, right now, for this workload?”**

---

## Status

Use this section as the working checklist. When a task lands in code and has passing tests, change `[ ]` to `[x]`.

* [x] Confirm current recommendation output contract and identify compatibility constraints.
* [ ] Extend catalog recommendations with ROI fields: speedup estimate, effort, safety/risk, prerequisites, and validation metadata.
* [x] Implement normalized scoring helpers for speedup, confidence, cost savings, ease, and safety.
* [x] Replace the current ranking formula with the weighted ROI formula.
* [x] Add guardrails: confidence floor, risk demotion, dependency ordering, and duplicate suppression.
* [x] Add tier assignment: `try_now`, `next`, and `advanced`.
* [x] Expose explanation fields that show why each recommendation ranked where it did.
* [x] Add or update golden tests for score ordering, tiering, guardrails, and compatibility with existing recommendation output.
* [ ] Add recommendation outcome logging schema for attempted/applied/user-dismissed results.

## Getting started

The repo already has a first-pass recommendation engine in `python/autopilot_telemetry/recommendations/engine.py` and coverage in `tests/python/test_recommendation_engine.py`. Start by preserving the existing public fields (`id`, `title`, `priority`, `score`, `confidence`, `expected_impact`, `effort`, `category`, `why`, `actions`, `validation`, `risks`, `triggered_by`) while adding the richer ROI fields behind them.

Suggested first implementation slice:

1. Add pure scoring helpers to the recommendation engine or a new small ranker module.
2. Backfill default ROI metadata from the existing catalog values so old catalog entries still work.
3. Keep `score` as the final ROI score for API compatibility.
4. Add a `tier` field without removing `priority`.
5. Update tests to assert ordering and tier behavior before tuning weights.

# Goal of Step 7

Take the recommendation candidates from Step 6 and rank them using a consistent ROI model based on:

* estimated speedup
* implementation effort
* confidence
* cost savings
* blast radius / risk

The output should feel like:

1. Increase dataloader workers from 4 to 12
2. Enable pinned memory
3. Try `torch.compile` on the forward pass
4. Replace repeated small ops with a fused path

That is what makes the product actually useful.

---

# Step 1: Define what “ROI” means in your system

Before writing ranking code, define the objective clearly.

For your first version, ROI should mean:

**high expected performance gain, high confidence, low effort, low risk**

A good recommendation is not just “big theoretical speedup.”
It is:

* likely to help
* cheap enough to try
* safe enough to attempt
* meaningful enough to matter

So your ranking function should prefer:

* medium-sized, very safe wins over
* giant but speculative or dangerous changes

Example:

* **Enable pinned memory** may beat
* **Rewrite a custom CUDA kernel**

even if the rewrite has a bigger ultimate upside, because pinned memory is faster to test and much safer.

---

# Step 2: Define the recommendation object

Each recommendation should already exist as a structured object before ranking.

Example schema:

```json
{
  "recommendation_id": "rec_enable_pinned_memory",
  "title": "Enable pinned memory",
  "category": "input_pipeline",
  "description": "Pinned host memory can reduce host-to-device transfer overhead and improve batch delivery speed.",
  "rationale": "GPU active time is low while CPU wait and H2D transfer overhead are elevated.",
  "estimated_speedup_pct_min": 3,
  "estimated_speedup_pct_max": 12,
  "implementation_effort": "low",
  "confidence": 0.84,
  "risk_level": "low",
  "blast_radius": "small",
  "cost_savings_score": 0.62,
  "prerequisites": [],
  "validation_metric": "GPU active %, step time, H2D copy time"
}
```

This is important because ranking works best on structured fields, not prose.

---

# Step 3: Standardize each scoring dimension

Now convert each ranking factor into a normalized score.

Use a common range such as:

* **0.0 to 1.0**
* where higher is better for beneficial factors
* and higher is worse for penalty factors unless inverted

Recommended dimensions:

## Positive factors

* expected speedup
* confidence
* cost savings

## Negative factors

* implementation effort
* risk / blast radius

You can either:

* invert negative factors into “ease” and “safety”
* or treat them as penalties

I recommend the first approach because it makes the formula cleaner.

Example normalized fields:

```json
{
  "speedup_score": 0.78,
  "confidence_score": 0.84,
  "cost_savings_score": 0.62,
  "ease_score": 0.90,
  "safety_score": 0.88
}
```

---

# Step 4: Create explicit rubrics for each factor

Do not let these scores be vague. Define rules.

## 4.1 Estimated speedup score

Translate the predicted speedup range into a score.

Example rubric:

* <2% → 0.10
* 2–5% → 0.25
* 5–10% → 0.45
* 10–20% → 0.70
* 20–35% → 0.90
* > 35% → 1.00

Or use a smooth function instead of buckets.

Important: use **credible expected speedup**, not best-case marketing speedup.

For example:

* dataloader worker tuning may be 5–20%
* pinned memory may be 3–12%
* kernel fusion on many tiny ops may be 10–30%
* multi-GPU communication fixes may be workload-specific

Keep estimates conservative.

---

## 4.2 Implementation effort score

Effort should reflect how hard it is for the user to try and validate.

Example rubric:

* config-only change → 1.00
* 1-line code change → 0.90
* small local code refactor → 0.70
* moderate pipeline change → 0.45
* invasive model/runtime rewrite → 0.20
* custom CUDA/Triton engineering → 0.10

Examples:

* increase dataloader workers → high ease
* enable pinned memory → high ease
* try `torch.compile` → medium-high ease
* rewrite repeated ops as fused kernel → low ease

This factor is critical because users usually want the easiest wins first.

---

## 4.3 Confidence score

Confidence should come from evidence quality.

Example inputs to confidence:

* strength of bottleneck classifier score
* amount of corroborating evidence
* how direct the causal link is
* historical success rate of this recommendation on similar workloads

Example rubric:

* weak evidence / speculative → 0.30
* moderate evidence → 0.55
* strong multi-signal evidence → 0.80
* highly repeatable pattern match → 0.95

For example:

**low GPU active + high CPU wait + dataloader gaps + slow batch readiness**
should give high confidence for input-pipeline recommendations.

---

## 4.4 Cost savings score

This translates technical speedup into financial value.

You do not need perfect dollar estimates at first. Start simple.

Possible inputs:

* GPU hourly cost
* number of GPUs
* workload runtime
* job frequency
* cluster utilization

Simple idea:

```text
cost_savings ≈ expected_speedup * gpu_spend_exposure
```

A 10% speedup on:

* an A100 single-GPU hobby run is not the same as
* a 128-GPU training fleet

So recommendations should rank higher when they affect expensive workloads.

You can normalize this into a 0–1 score.

Example:

* small local job → 0.20
* daily multi-GPU training pipeline → 0.80
* large always-on inference service → 0.95

---

## 4.5 Safety score from blast radius / risk

Risk is not just “could break.”
It includes:

* correctness risk
* reproducibility risk
* production stability risk
* rollback difficulty
* scope of systems affected

Example rubric:

* safe, reversible tuning knob → 1.00
* isolated code path change → 0.75
* behavior-changing runtime optimization → 0.55
* broad architecture change → 0.25
* distributed / production-critical invasive change → 0.10

Examples:

* increase workers → very safe
* pinned memory → very safe
* `torch.compile` → moderate safety
* rewriting ops or changing sharding/communication → lower safety

---

# Step 5: Start with a simple weighted scoring formula

Do not overcomplicate v1.

A good first formula:

```text
ROI score =
0.30 * speedup_score
+ 0.20 * confidence_score
+ 0.20 * cost_savings_score
+ 0.15 * ease_score
+ 0.15 * safety_score
```

This is simple, explainable, and easy to tune.

Why this works:

* speedup matters most
* confidence and cost matter a lot
* ease and safety keep the system practical

You can later adjust weights by customer segment.

For example:

* research users may tolerate more risk
* enterprise production users may care more about safety

---

# Step 6: Add hard penalties and guardrails

A plain weighted score is not enough.

Some recommendations should be pushed down or flagged even if their raw ROI looks high.

Add guardrails like:

## 6.1 Confidence floor

If confidence < 0.4:

* do not show in top recommendations
* or label as exploratory

## 6.2 Risk cap

If risk is high and speedup is uncertain:

* demote heavily

## 6.3 Dependency rules

Some recommendations should only rank if prerequisites are satisfied.

Example:

* “fuse repeated small ops with Triton” should not appear above
* “try `torch.compile`”
  if the easier option has not been tried.

## 6.4 Duplicate suppression

If two recommendations are very similar:

* cluster or combine them

Example:

* “increase workers”
* “increase prefetch factor”
* “enable pinned memory”

These may belong in one grouped recommendation block under **Improve input pipeline throughput**.

---

# Step 7: Introduce recommendation tiers

Do not just give one long ranked list.

Group them into tiers.

Suggested tiers:

## Try now

High ROI, low effort, low risk

## Next

Good ROI, moderate effort or uncertainty

## Advanced

Potentially large gains, but higher effort or risk

This makes the output much more usable.

Example:

### Try now

* Increase dataloader workers from 4 to 12
* Enable pinned memory
* Increase prefetch factor

### Next

* Try `torch.compile` on the forward pass
* Batch small CPU-to-GPU transfers

### Advanced

* Replace repeated small ops with a fused path
* Rework distributed gradient bucketing

That presentation is much better than a raw score table alone.

---

# Step 8: Make the ranking explainable

Every ranked recommendation should answer:

* why this is being suggested
* why it is ranked here
* what evidence supports it
* how hard it is to try
* how to validate whether it helped

Example format:

```json
{
  "title": "Increase dataloader workers from 4 to 12",
  "roi_score": 0.86,
  "rank": 1,
  "why_ranked_high": [
    "Strong evidence of input pipeline starvation",
    "Low implementation effort",
    "Low operational risk",
    "Likely to improve GPU utilization quickly"
  ],
  "expected_impact": "medium",
  "effort": "low",
  "risk": "low",
  "confidence": "high",
  "validate_by": [
    "step time",
    "GPU active %",
    "batch ready latency"
  ]
}
```

This is important because users need to trust the ranking.

---

# Step 9: Build a recommendation ranking pipeline

Implementation flow should be:

## Input

Diagnostic summary + recommendation candidates

## Pipeline

1. compute feature-level scores per recommendation
2. apply weights
3. apply penalties / guardrails
4. suppress duplicates
5. enforce dependencies / ordering logic
6. sort descending by final ROI score
7. group into tiers
8. generate human-readable output

Conceptually:

```text
diagnostics
→ recommendation candidates
→ score normalization
→ weighted ROI calculation
→ safety/confidence penalties
→ dependency resolution
→ dedup/grouping
→ final ranked output
```

---

# Step 10: Encode domain-specific ranking heuristics

Some heuristics are worth hardcoding from the start.

## 10.1 Prefer safe knobs before rewrites

If two recommendations have similar expected gain:

* prefer configuration changes
* then runtime switches
* then local code changes
* then invasive rewrites

## 10.2 Prefer fixes that directly match strongest bottleneck

If input pipeline bound score is very high:

* input recommendations should outrank generic compute optimizations

## 10.3 Prefer broad leverage fixes

A fix that improves every training step should outrank one that helps a rare phase.

## 10.4 Prefer reversible experiments

Users like things they can A/B quickly.

This is why:

* changing workers
* enabling pinned memory
* toggling `torch.compile`

should often surface near the top.

---

# Step 11: Add “action templates” for the final output

Your output should not just say what to do.
It should be phrased as a concrete action.

Good examples:

* Increase dataloader workers from 4 to 12
* Enable pinned memory in the training dataloader
* Increase batch prefetch depth
* Try `torch.compile` on the forward pass
* Batch repeated small host-device transfers
* Replace repeated small ops with a fused execution path
* Reduce synchronization caused by `.item()` and host-side waits
* Tune gradient bucket sizes to improve overlap

This is the right level of specificity.

Not too vague:

* “optimize data loading”

Not too deep:

* “rewrite the dataloader state machine with lock-free ring buffers”

---

# Step 12: Define what the user-facing output should include

For each recommendation, show:

* rank
* action title
* expected impact
* effort
* confidence
* risk
* why it was suggested
* how to validate it

Example:

## 1. Increase dataloader workers from 4 to 12

**Expected impact:** Medium
**Effort:** Low
**Confidence:** High
**Risk:** Low

Why:

* GPU active time is low
* CPU wait time is elevated
* batch delivery gaps suggest input starvation

Validate with:

* higher GPU active %
* lower step time
* reduced batch-ready latency

---

## 2. Enable pinned memory

**Expected impact:** Low to Medium
**Effort:** Low
**Confidence:** High
**Risk:** Low

Why:

* host-to-device transfers are contributing to batch latency
* pinned memory is a low-risk optimization for input staging

Validate with:

* reduced H2D copy time
* improved end-to-end step throughput

---

## 3. Try `torch.compile` on the forward pass

**Expected impact:** Medium
**Effort:** Medium
**Confidence:** Medium
**Risk:** Medium

Why:

* trace shows many small repeated kernels
* model may benefit from graph capture / fusion opportunities

Validate with:

* reduced kernel launch count
* lower step time
* stable numerics and correctness

---

## 4. Replace repeated small ops with a fused path

**Expected impact:** High
**Effort:** High
**Confidence:** Medium
**Risk:** Medium to High

Why:

* kernel launch overhead is a major bottleneck
* repeated tiny operators are creating inefficient execution

Validate with:

* fewer launches
* lower CPU launch overhead
* lower end-to-end latency

---

# Step 13: Log outcomes so ranking gets smarter over time

This step is very important.

Once users try recommendations, log:

* was it attempted
* did it help
* actual speedup
* effort/time to implement
* regressions or rollback
* workload type
* hardware type

This later lets you improve ranking with real-world evidence.

Eventually, your confidence and speedup estimates become grounded in:

* model family
* GPU type
* batch size regime
* distributed setup
* workload category

That becomes a strong moat.

---

# Step 14: Keep v1 deterministic

Do not jump to learned ranking too early.

For the first version:

* rules generate recommendations
* weighted formula ranks them
* thresholds and guardrails control safety

That gives you:

* explainability
* repeatability
* low hallucination risk
* easier debugging

Later, you can learn better priors from historical outcomes.

---

# Step 15: Suggested v1 implementation order

Build this in the following sequence:

## Part A

Create the structured recommendation schema

## Part B

Define normalized rubrics for:

* speedup
* effort
* confidence
* cost savings
* safety

## Part C

Implement the weighted ROI formula

## Part D

Add penalties, floors, and dependency rules

## Part E

Add tiering:

* Try now
* Next
* Advanced

## Part F

Generate clean user-facing ranked recommendations

## Part G

Log recommendation outcomes for future tuning

---

# Good v1 principle

A great first version does **not** need perfect prediction.

It only needs to consistently rank:

* easy, likely wins near the top
* speculative, risky rewrites lower down

If the user sees:

1. Increase dataloader workers from 4 to 12
2. Enable pinned memory
3. Try `torch.compile` on the forward pass
4. Replace repeated small ops with a fused path

and that order usually feels right, then the system is already doing valuable work.

---

# Final mental model

Step 6 answers:

**“What could fix this?”**

Step 7 answers:

**“What should the user try first?”**

That ranking layer is what turns diagnostics into action.

If you want, I can next turn this into a concrete **JSON schema + scoring formula + pseudocode** for implementation.
