## Step 9: Add “safe autopilot” actions

### Implementation Progress

- [x] **1.** Define safe autopilot action schema — `autopilot/actions.py` (AutopilotAction, CandidateConfig, TrialResult, PromotionThresholds)
- [x] **2.** Create Autopilot Action schema — dataclasses with tier, env_vars, rollback, risk fields
- [x] **3.** Safety tiers — TIER_SAFE=0, TIER_VALIDATED=1, TIER_RISKY=2 enforced in tuners
- [x] **4.** Experiment runner — `autopilot/runner.py` (ExperimentRunner: baseline → candidates → trials → winner)
- [x] **5.** Benchmark window — FRX_TUNE_WARMUP_STEPS / FRX_TUNE_MAX_STEPS injected per trial
- [x] **6.** Correctness guards — `autopilot/guards.py` (exit code, step count, throughput, memory ratio, step time regression)
- [x] **7.** Dataloader autopilot — `autopilot/tuners/dataloader.py` (num_workers, pin_memory, prefetch_factor, persistent_workers grid)
- [x] **8.** Batch size tuner — `autopilot/tuners/batch_size.py` (1.25×, 1.5×, 2× candidates with OOM guard)
- [x] **9.** Mixed precision autopilot — `autopilot/tuners/mixed_precision.py` (bf16 preferred, fp16 fallback, GPU detection)
- [x] **13.** Staged search — `autopilot/tuners/__init__.py` (Stage 1: dl → Stage 2: bs → Stage 3: amp)
- [x] **14.** Promotion rules — PromotionThresholds (min_speedup=8%, max_memory=90%, step_time_regression=10%)
- [x] **16.** User-facing report — `autopilot/report.py` (winner, env vars to apply, rollback config)
- [x] **17.** Apply/rollback modes — `recommend_only` by default; env vars printed; --apply flag planned for v2
- [x] **19.** First shippable version — `frx tune train.py --safe --max-trials 12` CLI command live
- [ ] **15.** Postgres experiment storage — planned for v2
- [ ] **10.** torch.compile tuner — planned for v2
- [ ] **11.** CUDA Graphs tuner — planned for v2
- [ ] **12.** Env var tuner — planned for v2
- [ ] **18.** Full MVP order items 8–12 — planned for v2

---

### Goal

Move from:

> “You should increase dataloader workers.”

to:

> “I tested 8 safe configurations. `num_workers=8`, `pin_memory=True`, and AMP improved throughput by 23% with no accuracy regression.”

This is the first version of real autopilot.

---

# 1. Define what “safe autopilot” means

Safe actions must be:

**Configurable**
No code rewrite required at first.

**Reversible**
Can return to baseline instantly.

**Measurable**
Must prove speedup with before/after runs.

**Low blast radius**
Should not silently change model quality, numerical behavior, or training semantics.

Good first targets:

```text
batch size
mixed precision
dataloader workers
pin_memory
prefetch_factor
persistent_workers
torch.compile
CUDA graphs
environment variables
distributed runtime knobs
```

Avoid early:

```text
kernel rewrites
custom Triton generation
automatic model graph surgery
optimizer/loss changes
distributed sharding changes without user approval
```

---

# 2. Create an Autopilot Action schema

Every action should be represented as a structured object.

Example:

```json
{
  "action_id": "enable_amp",
  "type": "mixed_precision",
  "description": "Enable torch autocast with fp16/bf16",
  "risk": "medium",
  "reversible": true,
  "requires_user_approval": false,
  "preconditions": [
    "cuda_available",
    "model_forward_is_amp_safe"
  ],
  "expected_benefit": "higher tensor core utilization",
  "rollback": {
    "disable_amp": true
  }
}
```

Your system should not “just try stuff.”
It should try known actions with explicit guardrails.

---

# 3. Split actions into safety tiers

## Tier 0: Very safe

These are mostly runtime/data loading config changes.

```text
num_workers
pin_memory
prefetch_factor
persistent_workers
CUDA_VISIBLE_DEVICES ordering
PYTORCH_CUDA_ALLOC_CONF
NCCL debug/env diagnostics
```

Use these first.

## Tier 1: Usually safe, but needs validation

```text
batch size changes
mixed precision
torch.compile
gradient accumulation adjustment
CUDA graphs for static-shape inference/training loops
```

These need correctness checks.

## Tier 2: Higher risk

```text
distributed runtime tuning
bucket sizes
activation checkpointing
memory format changes
channels_last
optimizer changes
custom kernels
```

These should require user approval in early versions.

---

# 4. Build the experiment runner

This is the core of safe autopilot.

The runner should:

1. Capture baseline.
2. Generate candidate configs.
3. Run each config for a small controlled window.
4. Measure throughput, latency, GPU utilization, memory, errors.
5. Validate correctness.
6. Pick the winner.
7. Save a report.
8. Optionally apply the best config.

Basic flow:

```text
baseline run
   ↓
generate safe candidates
   ↓
run candidate A
run candidate B
run candidate C
   ↓
compare against baseline
   ↓
choose winner
   ↓
recommend or apply
```

---

# 5. Define the benchmark window

Do not measure the first few steps only.

Use:

```text
warmup_steps: 5-10
measure_steps: 20-100
repeat_count: 2-3
```

For inference:

```text
warmup requests
fixed input shapes
latency percentiles
throughput
memory peak
error rate
```

For training:

```text
samples/sec
step time
GPU active %
loss sanity
memory peak
OOM events
data wait %
```

---

# 6. Add correctness guards

This is what makes it “safe.”

For training, check:

```text
loss is finite
no NaNs/Infs
loss does not explode versus baseline
optimizer step succeeds
gradients are finite
memory usage stays below threshold
```

For inference, check:

```text
output shape matches
dtype is acceptable
max output difference below tolerance
latency improved
no runtime errors
```

Example validation result:

```json
{
  "candidate": "amp_bf16",
  "passed": true,
  "throughput_change": 0.18,
  "max_output_diff": 0.0007,
  "memory_change": -0.21
}
```

---

# 7. Start with dataloader autopilot

This is the best first autopilot module because it is low-risk and reversible.

Tune:

```text
num_workers: [0, 2, 4, 8, 12, 16]
pin_memory: [true, false]
persistent_workers: [true, false]
prefetch_factor: [2, 4, 8]
```

Guardrails:

```text
do not exceed CPU core limit
do not exceed memory budget
skip prefetch_factor when num_workers=0
rollback if data wait worsens
```

Output:

```text
Best config:
num_workers=8
pin_memory=True
persistent_workers=True
prefetch_factor=4

Result:
+19% samples/sec
-42% dataloader wait
+3% CPU usage
No memory pressure detected
```

---

# 8. Add batch size tuner

Batch size has huge ROI but needs more care.

Strategy:

```text
start from current batch size
try 1.25x, 1.5x, 2x
stop on OOM
track throughput and memory
optionally adjust gradient accumulation
```

Guardrails:

```text
max GPU memory usage < 90%
no OOM
loss finite
same effective batch size if gradient accumulation changes
```

Example:

```text
Current batch size: 32
Tested: 32, 48, 64, 96

Winner:
batch_size=64

Impact:
+31% throughput
GPU memory: 7.1GB / 8GB
No instability detected
```

---

# 9. Add mixed precision autopilot

Try:

```text
fp32 baseline
amp fp16
amp bf16
```

Prefer bf16 when supported.

Validation:

```text
loss finite
output diff within tolerance
grad scaler stable for fp16
no NaNs/Infs
```

For training, do not auto-apply unless validation passes over enough steps.

Output:

```text
AMP bf16 passed validation.
Throughput improved by 22%.
Peak memory decreased by 18%.
Recommended: enable autocast(dtype=torch.bfloat16).
```

---

# 10. Add torch.compile autopilot

Try conservative modes first:

```python
torch.compile(model, mode="default")
torch.compile(model, mode="reduce-overhead")
torch.compile(model, mode="max-autotune")
```

Guardrails:

```text
skip if dynamic shapes are unstable
skip if compile time is too high
rollback on graph breaks/errors
measure separately after compile warmup
```

Track:

```text
compile time
steady-state speedup
graph breaks
error rate
memory increase
```

Important: report compile overhead separately.

```text
torch.compile improved steady-state throughput by 17%, but added 42 seconds of compile time.
Recommended for long-running jobs, not short runs.
```

---

# 11. Add CUDA Graphs only for safe cases

CUDA Graphs are powerful but need strict preconditions.

Use only when:

```text
static shapes
stable memory addresses
no data-dependent control flow
no CPU sync inside captured region
fixed batch size
repeatable execution path
```

Start with inference first.

Output:

```text
CUDA Graph capture was safe for this inference path.
Latency improved from 4.8ms to 3.6ms.
Recommended for fixed-shape serving.
```

---

# 12. Add environment variable tuning

Start with reversible env changes.

Examples:

```text
PYTORCH_CUDA_ALLOC_CONF
CUDA_MODULE_LOADING=LAZY
NCCL_SOCKET_NTHREADS
NCCL_NSOCKS_PERTHREAD
NCCL_BUFFSIZE
OMP_NUM_THREADS
MKL_NUM_THREADS
```

Be careful: env vars can be workload-specific.

The system should say:

```text
This change affects only the benchmark subprocess.
It was not applied globally.
```

Run experiments in isolated subprocesses so env changes do not pollute the parent process.

---

# 13. Build a search strategy

Do not brute-force everything.

Use staged search:

```text
Stage 1: dataloader
Stage 2: batch size
Stage 3: mixed precision
Stage 4: torch.compile
Stage 5: CUDA graphs
Stage 6: combined best config
```

Why staged?

Because testing all combinations explodes fast.

Example:

```text
6 dataloader configs
4 batch sizes
3 precision modes
3 compile modes

Full search = 216 configs
Staged search = maybe 16-25 configs
```

Much better.

---

# 14. Add promotion rules

A candidate should only win if it clears minimum thresholds.

Example:

```json
{
  "min_speedup": 0.08,
  "max_memory_pct": 0.90,
  "max_accuracy_delta": 0.001,
  "max_error_rate": 0,
  "require_no_nan": true
}
```

Do not recommend tiny noisy improvements.

Example:

```text
Candidate improved throughput by 2.1%, but below promotion threshold.
Not recommended.
```

---

# 15. Store experiment results

Use Postgres.

Tables:

```text
runs
baseline_metrics
autopilot_experiments
candidate_configs
candidate_results
validation_results
winning_configs
rollback_records
```

This becomes your product data flywheel.

You can later answer:

```text
For ResNet-style vision models on RTX 4090, pin_memory + 8 workers wins 72% of the time.
```

That is valuable.

---

# 16. Create the user-facing report

The report should be extremely clear.

Example:

```text
Autopilot tested 12 safe configurations.

Winner:
batch_size=64
num_workers=8
pin_memory=True
mixed_precision=bf16

Result:
+27% samples/sec
-18% peak memory
-41% dataloader wait
No NaNs detected
Loss stayed within expected range

Confidence:
High

Applied:
No — recommendation only

Rollback:
Use original config:
batch_size=32
num_workers=4
pin_memory=False
precision=fp32
```

---

# 17. Add apply/rollback modes

Support three modes:

```text
recommend_only
apply_once
apply_and_persist
```

For MVP, default to:

```text
recommend_only
```

Then later:

```text
--autopilot apply-once
```

Eventually:

```text
--autopilot apply --write-config
```

Never silently persist changes early.

---

# 18. MVP implementation order

Build in this order:

```text
1. Autopilot action schema
2. Experiment runner
3. Baseline capture
4. Dataloader tuner
5. Batch size tuner
6. Mixed precision tuner
7. Report generator
8. Postgres experiment storage
9. torch.compile tuner
10. CUDA Graphs safe-mode tuner
11. Env var tuner
12. Distributed tuning
```

---

# 19. First shippable version

Your first safe autopilot should do only this:

```text
dataloader workers
pin_memory
prefetch_factor
batch size
mixed precision
```

That is enough for a real product moment.

CLI example:

```bash
gpu-autopilot tune train.py --safe --max-trials 12
```

Output:

```text
Tested 12 safe configs.
Best config improved throughput by 23%.
No correctness issues detected.
Saved report to runs/autopilot_report.json.
```

---

# 20. The key product principle

The product should not feel like magic.

It should feel like:

> “A careful CUDA/PyTorch performance engineer ran a controlled experiment for me and found the best safe config.”

That is the wedge.
Use **mostly Python**, with **C++ only where needed**.

Best split:

```text
Python: 80–90%
C++: 10–20%
C: almost none
```

### For Step 9 specifically

**Python should own:**

```text
experiment runner
config generation
PyTorch integration
dataloader tuning
batch size tuning
AMP/mixed precision
torch.compile trials
CUDA Graph safety checks
report generation
Postgres logging
CLI/API
```

Python is best because the knobs are mostly PyTorch/runtime-level knobs.

**C++ should own:**

```text
low-level telemetry sampling
NVML/DCGM bindings if needed
CUDA event timing helpers
CUPTI/Nsight-style trace collection later
low-overhead background collector
```

Use C++ when overhead matters or when NVIDIA APIs are easier/cleaner from native code.

**C should not be the main language.**

Use C only for:

```text
tiny ABI layer
low-level compatibility wrapper
embedding into other runtimes
```

### Recommended architecture

```text
Python Autopilot Controller
  ├── generates configs
  ├── runs trials
  ├── validates correctness
  ├── chooses winner
  └── writes report/database

C++ Telemetry Engine
  ├── GPU metrics
  ├── memory stats
  ├── kernel timing
  ├── low-overhead sampling
  └── exposes Python bindings
```

### MVP choice

For the first version, do:

```text
100% Python
```

Then add C++ once you hit telemetry overhead or need CUPTI/deeper CUDA integration.

So the practical answer:

> Build Step 9 in Python first. Add C++ for the telemetry/profiling engine later. Do not start in pure C or pure C++.
