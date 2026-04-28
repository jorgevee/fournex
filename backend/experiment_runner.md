## Step 10: Build an experiment runner

This is the point where your GPU optimization product becomes an **active optimizer**, not just a profiler or recommendation engine.

## Implementation progress

- [x] Phase 1: Manual local benchmark runner foundation.
- [x] Add a stable `TrialConfig` abstraction.
- [x] Add `LocalTrialExecutor` for isolated local trial execution.
- [x] Persist per-trial `config.yaml`, `metrics.json`, `stdout.log`, and `stderr.log`.
- [x] Persist experiment-level JSON and Markdown reports.
- [x] Add focused regression tests for local trial artifacts and report persistence.
- [x] Phase 2: Candidate generation from a bottleneck diagnosis.
- [x] Add focused routing for `input_bound`, `copy_bound`, `launch_bound`, `memory_pressure`, and `underutilized_gpu`.
- [x] Add `frx tune --bottleneck` override for manual candidate focus.
- [x] Auto-detect candidate focus from the baseline summary diagnosis when present.
- [x] Phase 3: Safety validation before running candidates.
- [x] Reject unsafe candidates before execution and record skipped trial artifacts.
- [x] Validate risky action policy, CUDA availability, dynamic-shape CUDA Graph usage, compile support, precision quality checks, and batch-size memory headroom.
- [x] Phase 4: Explicit benchmark window controls beyond env injection.
- [x] Add first-class benchmark window config with warmup, measurement, repeat count, and timeout fields.
- [x] Persist `benchmark_window.json` per trial and include window policy in `config.yaml`.
- [x] Prefer explicit `measurement_window` metrics when per-step data is available.
- [x] Phase 5: Quality regression detection.
- [x] Extract loss and quality metrics from measurement-window summaries.
- [x] Reject trials with NaN/Inf loss, final-loss regression, loss divergence, or output drift beyond tolerance.
- [x] Persist quality metrics in per-trial `metrics.json` and reports.
- [ ] Phase 6: Noise-aware comparison.
- [ ] Phase 7: SQLite persistence.
- [ ] Phase 8: CI integration.
- [ ] Phase 9: Cluster integration.

The experiment runner should answer:

> “Given this workload and bottleneck diagnosis, what safe configuration changes can we test, how do we compare them fairly, and when do we stop?”

The core job is not just running experiments. The core job is running **trustworthy experiments**.

---

# 1. Define the experiment runner’s responsibility

The runner should manage the full lifecycle:

```text
Input:
  baseline workload
  telemetry diagnosis
  allowed optimization actions
  user safety policy

Process:
  generate candidate configs
  execute controlled trials
  collect metrics
  compare against baseline
  detect regressions
  stop bad trials early

Output:
  ranked results
  confidence score
  performance deltas
  quality/regression warnings
  reproducible run record
```

Example output:

```text
Tested 8 valid configurations.

Best safe result:
  dataloader_workers: 8
  pin_memory: true
  prefetch_factor: 4
  mixed_precision: unchanged

Throughput improved: +23.4%
Step time reduced: -18.9%
GPU active time improved: +31.2%
Loss curve: no regression detected
Numerics: passed tolerance checks
Confidence: medium-high
```

---

# 2. Start with a narrow experiment scope

Do **not** start by optimizing everything.

For v1, support safe, reversible knobs:

```text
Dataloader:
  num_workers
  prefetch_factor
  pin_memory
  persistent_workers

Batching:
  batch_size candidates
  gradient accumulation adjustment

Precision:
  fp32 baseline
  autocast mixed precision
  bf16/fp16 when available

Runtime:
  torch.compile on/off
  CUDA graphs when safe
  environment variables
  thread settings
```

Avoid early support for risky changes like:

```text
model architecture changes
optimizer changes
loss function changes
aggressive quantization
custom kernel replacement
distributed sharding rewrites
```

Those can come later.

---

# 3. Design the core architecture

A good v1 architecture looks like this:

```text
autopilot/
  runner/
    experiment_planner.py
    candidate_generator.py
    trial_executor.py
    benchmark_controller.py
    early_stopper.py
    result_comparator.py
    regression_detector.py
    report_generator.py

  adapters/
    local_runner.py
    slurm_runner.py
    kubernetes_runner.py
    ci_runner.py

  workload/
    workload_spec.py
    config_patch.py
    reproducibility.py

  metrics/
    metric_schema.py
    telemetry_collector.py
    quality_checks.py

  storage/
    sqlite_store.py
    postgres_store.py
    artifact_store.py
```

The main abstraction should be:

```python
ExperimentPlan
Trial
TrialConfig
TrialResult
ComparisonReport
```

Do not let the runner become a pile of scripts.

---

# 4. Create a workload specification format

You need a standard way to describe what is being tested.

Example:

```yaml
workload:
  name: resnet50_training
  framework: pytorch
  entrypoint: train.py
  command:
    - python
    - train.py
    - --epochs
    - "1"

baseline:
  config:
    batch_size: 64
    num_workers: 4
    pin_memory: false
    precision: fp32

constraints:
  max_trials: 12
  max_runtime_minutes: 30
  min_warmup_steps: 10
  benchmark_steps: 100
  require_quality_checks: true

metrics:
  primary: samples_per_second
  secondary:
    - step_time_ms
    - gpu_utilization
    - memory_allocated
    - dataloader_wait_pct
    - loss
```

This becomes your experiment contract.

---

# 5. Build the candidate config generator

The generator takes:

```text
baseline config
detected bottleneck
hardware info
user constraints
known unsafe rules
```

Then it emits candidate configs.

Example for input pipeline starvation:

```python
[
  {"num_workers": 2, "pin_memory": True},
  {"num_workers": 4, "pin_memory": True},
  {"num_workers": 8, "pin_memory": True},
  {"num_workers": 8, "pin_memory": True, "prefetch_factor": 4},
  {"num_workers": 12, "pin_memory": True, "prefetch_factor": 4},
]
```

Example for small-kernel overhead:

```python
[
  {"torch_compile": True},
  {"torch_compile": True, "compile_mode": "reduce-overhead"},
  {"cuda_graphs": "try_if_static_shapes"},
]
```

Example for memory-bound workload:

```python
[
  {"precision": "bf16"},
  {"precision": "fp16"},
  {"channels_last": True},
]
```

The candidate generator should have guardrails.

For example:

```text
Do not increase batch size if memory headroom < 20%.
Do not use CUDA graphs if shapes are dynamic.
Do not test fp16 if loss already contains NaN/Inf.
Do not enable torch.compile if workload uses unsupported dynamic behavior.
Do not change precision without quality checks.
```

---

# 6. Separate “safe knobs” from “risky knobs”

This matters a lot.

## Safer knobs

These usually affect performance without changing training math much:

```text
num_workers
pin_memory
prefetch_factor
persistent_workers
torch.set_num_threads
CUDA allocator env vars
logging/profiling overhead
compile flags
```

## Medium-risk knobs

These can affect correctness, numerics, or convergence:

```text
batch size
gradient accumulation
mixed precision
bf16/fp16
torch.compile
CUDA graphs
distributed bucket size
```

## High-risk knobs

Avoid for v1 autopilot:

```text
optimizer changes
learning rate changes
loss function changes
architecture changes
quantization
kernel rewrites
custom fused kernels
```

Your product should label actions by risk.

Example:

```text
Low-risk:
  Increase dataloader workers from 4 to 8

Medium-risk:
  Enable bf16 autocast

High-risk:
  Replace attention implementation
```

---

# 7. Build the local trial executor first

Start local.

A trial executor should:

```text
create isolated trial directory
write config patch
set environment variables
run command
collect stdout/stderr
collect telemetry
collect benchmark metrics
save artifacts
return structured TrialResult
```

Example structure:

```text
runs/
  experiment_2026_04_27_001/
    baseline/
      config.yaml
      metrics.json
      trace.json
      stdout.log
      stderr.log

    trial_001/
      config.yaml
      metrics.json
      trace.json
      stdout.log
      stderr.log

    trial_002/
      config.yaml
      metrics.json
      trace.json
      stdout.log
      stderr.log

    report.json
    report.md
```

Every trial should be reproducible.

---

# 8. Add clean comparison controls

This is one of the hardest parts.

You need to make trials comparable.

## Control randomness

For PyTorch:

```python
import random
import numpy as np
import torch

seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

Optionally:

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

But be careful: full determinism can reduce performance and may not represent real production speed.

So support two modes:

```text
deterministic_benchmark: cleaner comparison, less realistic
production_like_benchmark: more realistic, more noisy
```

## Control data

Use the same:

```text
dataset
data subset
data order
batch count
sequence lengths
augmentation settings
```

For early v1, use a fixed benchmark window:

```text
warmup: 10-20 steps
measure: 50-200 steps
repeat: 2-5 times if possible
```

## Control system load

Record:

```text
GPU model
driver version
CUDA version
PyTorch version
CPU model
RAM
other GPU processes
GPU temperature
power limit
clock state
container image
git commit
```

If another process is using the GPU, mark the result as contaminated.

---

# 9. Implement warmup and measurement windows

Do not measure from step 0.

A good benchmark format:

```text
startup phase:
  import model, load data, allocate memory

warmup phase:
  10-30 steps ignored

measurement phase:
  50-500 steps measured

cooldown:
  collect final telemetry, save logs
```

Why?

Because first steps include:

```text
CUDA context creation
memory allocation
kernel autotuning
JIT compilation
torch.compile graph capture
data loader startup
cache warming
```

If you include those in normal throughput, you will lie to yourself.

---

# 10. Add repeated trials for noisy measurements

GPU performance is noisy.

Cloud GPUs are very noisy.

Do not compare one run against one run and declare victory.

For each candidate:

```text
run 1 baseline
run 1 trial
run 2 baseline
run 2 trial
optional run 3 baseline
optional run 3 trial
```

Better comparison pattern:

```text
baseline A
trial 1
baseline B
trial 1 repeat
baseline C
```

This helps detect drift.

The result comparator should compute:

```text
mean
median
p50
p90
standard deviation
confidence interval
relative improvement
regression probability
```

For v1, you can use a simple rule:

```text
Only call a trial "better" if:
  improvement > minimum threshold
  and no quality regression
  and no stability regression
  and result is larger than noise band
```

Example:

```text
Throughput +3% with ±5% noise: inconclusive
Throughput +23% with ±4% noise: likely improvement
Throughput +18% but loss diverged: unsafe
```

---

# 11. Define your core metrics

You need three metric classes.

## Performance metrics

```text
samples_per_second
tokens_per_second
step_time_ms
iteration_time_ms
GPU active %
dataloader wait %
CPU-GPU sync time
kernel launch count
memory bandwidth proxy
GPU memory allocated
GPU memory reserved
```

## Stability metrics

```text
OOM count
NaN/Inf count
crashes
timeouts
variance in step time
thermal throttling
clock throttling
unexpected process exit
```

## Quality metrics

```text
training loss
validation loss
accuracy
perplexity
gradient norm
parameter norm
output difference vs baseline
numerical tolerance checks
```

This is essential because:

```text
faster training != better training
```

A candidate that gives +40% throughput but damages convergence is not a win.

---

# 12. Add quality regression checks

This is where your runner becomes serious.

For training jobs, compare:

```text
loss curve
initial loss
final loss
loss slope
NaN/Inf occurrence
gradient norms
validation metric if available
```

Basic v1 checks:

```text
final_loss must not be worse than baseline by more than tolerance
loss must not diverge
no NaN/Inf
gradient norm must remain within sane range
```

Example:

```python
def passes_quality_gate(baseline, trial):
    if trial.nan_count > 0:
        return False

    if trial.final_loss > baseline.final_loss * 1.05:
        return False

    if trial.loss_slope > baseline.loss_slope * 0.5:
        return False

    return True
```

For inference jobs, compare:

```text
outputs against golden examples
max absolute error
relative error
classification match rate
latency
throughput
memory
```

Example:

```text
Mixed precision result:
  throughput: +31%
  max output diff: 0.0032
  tolerance: 0.005
  status: pass
```

---

# 13. Be very careful with batch size changes

Batch size is dangerous.

Increasing batch size may improve throughput but change optimization dynamics.

Example:

```text
baseline:
  batch_size = 64
  lr = 1e-4

trial:
  batch_size = 256
  lr = 1e-4
```

This may improve GPU utilization but change convergence.

So batch-size experiments should be marked as **medium risk** or **high risk** unless you preserve effective training behavior.

Safer approach:

```text
increase batch size
adjust gradient accumulation
preserve effective batch size when possible
```

Example:

```text
baseline:
  micro_batch_size = 32
  grad_accum = 4
  effective_batch_size = 128

trial:
  micro_batch_size = 64
  grad_accum = 2
  effective_batch_size = 128
```

That is much safer than changing the effective batch size.

---

# 14. Be careful with mixed precision

Mixed precision is a huge source of real speedups, but also quality bugs.

You need checks for:

```text
NaNs
Infs
loss scaling issues
gradient underflow
gradient overflow
accuracy drift
output drift
unstable convergence
```

For mixed precision trials, require:

```text
quality gate
numerics gate
fallback path
clear warning
```

Example result:

```text
bf16 autocast:
  throughput: +28%
  memory usage: -21%
  loss delta: +0.7%
  NaNs: 0
  status: safe candidate

fp16 autocast:
  throughput: +35%
  memory usage: -24%
  loss delta: +18%
  overflow events: 7
  status: rejected
```

This kind of output is very valuable.

---

# 15. Add early stopping for bad trials

Bad trials waste time and money.

Stop early when:

```text
OOM occurs
process crashes
loss becomes NaN
loss explodes
step time is much worse than baseline
GPU memory exceeds threshold
trial exceeds max runtime
dataloader stalls
no progress detected
```

Example policy:

```yaml
early_stop:
  on_oom: true
  on_nan_loss: true
  max_step_time_regression_pct: 50
  max_runtime_minutes: 10
  min_steps_before_perf_judgment: 20
```

Important: do not stop too early during warmup.

A `torch.compile` trial may look bad initially because compilation adds overhead. So classify trials:

```text
normal trial
compile trial
cuda graph trial
distributed trial
```

Each one needs different warmup rules.

---

# 16. Build a regression recorder

Every bad trial is useful data.

Record:

```text
config
error type
stack trace
hardware
framework version
failure phase
telemetry before failure
user workload metadata
```

Regression categories:

```text
performance_regression
quality_regression
numerical_regression
memory_regression
stability_regression
compatibility_failure
timeout
oom
crash
```

Example:

```json
{
  "trial_id": "trial_007",
  "status": "rejected",
  "regression_type": "quality_regression",
  "candidate": {
    "precision": "fp16"
  },
  "performance_delta": "+34.8%",
  "final_loss_delta": "+22.1%",
  "nan_count": 0,
  "reason": "Loss degraded beyond 5% tolerance."
}
```

This becomes part of your product data flywheel.

---

# 17. Support three execution backends

Eventually, the experiment runner should support:

```text
local machine
cluster jobs
CI benchmark pipelines
```

But build them in this order.

---

## Backend 1: Local runner

This is v1.

It runs experiments on the current machine.

Use it for:

```text
single-node PyTorch training
inference scripts
developer workflows
local GPU benchmarks
```

Interface:

```python
class LocalTrialExecutor:
    def run(self, trial_config: TrialConfig) -> TrialResult:
        ...
```

---

## Backend 2: CI runner

This is v2.

Useful for:

```text
benchmarking PRs
detecting performance regressions
testing model-serving changes
tracking infra changes
```

For CI, you need:

```text
fixed benchmark scripts
fixed datasets or synthetic inputs
versioned configs
pass/fail thresholds
artifact upload
trend tracking
```

Example CI output:

```text
PR benchmark result:
  throughput: -7.8%
  threshold: -5.0%
  status: failed
  suspected cause: dataloader wait increased
```

This could become a strong developer product.

---

## Backend 3: Cluster runner

This is v3.

Support:

```text
Slurm
Kubernetes
Ray
custom job queue
cloud batch services
```

For cluster jobs, the runner should submit jobs, not directly run everything itself.

Architecture:

```text
Experiment controller
  -> writes TrialSpec
  -> submits job
  -> monitors status
  -> collects artifacts
  -> compares results
```

Trial statuses:

```text
PENDING
RUNNING
SUCCEEDED
FAILED
STOPPED_EARLY
REJECTED
INCONCLUSIVE
```

---

# 18. Add a trial state machine

Do not use ad hoc booleans.

Use a clear state machine:

```text
CREATED
VALIDATED
QUEUED
RUNNING
WARMING_UP
MEASURING
COMPLETED
FAILED
STOPPED_EARLY
ANALYZED
ACCEPTED
REJECTED
INCONCLUSIVE
```

This matters when you support clusters and CI.

---

# 19. Store every result in a database

Use Postgres eventually, SQLite for local dev.

Tables:

```text
experiments
trials
trial_configs
trial_metrics
trial_events
trial_artifacts
regressions
hardware_snapshots
software_snapshots
comparison_reports
```

Example schema:

```sql
experiments
  id
  workload_name
  created_at
  baseline_config
  objective
  status

trials
  id
  experiment_id
  trial_index
  config
  status
  started_at
  ended_at
  exit_code

trial_metrics
  trial_id
  metric_name
  value
  unit
  step
  timestamp

regressions
  trial_id
  regression_type
  severity
  reason
  evidence
```

This is critical for learning over time.

---

# 20. Build the comparator carefully

The comparator should not simply sort by throughput.

It should rank by safe ROI:

```text
performance improvement
confidence
quality pass/fail
stability
risk
implementation complexity
cost impact
```

Example scoring:

```python
score = (
    speedup_score * 0.35
    + confidence_score * 0.25
    + quality_score * 0.20
    + stability_score * 0.10
    + risk_score * 0.10
)
```

But make hard gates first:

```text
If quality regression: reject.
If NaN/Inf: reject.
If OOM: reject.
If throughput improvement is within noise: inconclusive.
If memory use exceeds limit: reject.
```

Then rank only the surviving candidates.

---

# 21. Generate human-readable experiment reports

The report should explain what happened.

Example:

```text
Experiment summary

Baseline:
  batch_size: 64
  num_workers: 4
  pin_memory: false
  throughput: 820 samples/sec

Best candidate:
  batch_size: 64
  num_workers: 8
  pin_memory: true
  throughput: 1012 samples/sec

Improvement:
  +23.4% throughput
  -18.9% step time
  +29.1% GPU active time

Quality:
  final loss delta: +0.3%
  NaNs: 0
  status: pass

Rejected trials:
  fp16 autocast:
    +35% throughput but loss degraded by 18%

  num_workers=16:
    dataloader became unstable, higher p90 step time

Recommendation:
  Apply num_workers=8 and pin_memory=true.
```

This is the product moment.

---

# 22. Add safety modes

You should have several levels of automation.

## Mode 1: Observe only

```text
Run benchmark.
Do not modify config.
Produce recommendation.
```

## Mode 2: Suggest experiments

```text
Generate candidate configs.
Ask user to approve.
```

## Mode 3: Run experiments

```text
Execute trials.
Compare results.
Recommend winner.
```

## Mode 4: Apply safe config

```text
Apply low-risk changes automatically.
Save rollback config.
```

## Mode 5: Closed-loop autopilot

```text
Continuously optimize under constraints.
```

For now, build up to Mode 3.

Do not jump straight to full autopilot.

---

# 23. Python vs C++ split

For this experiment runner, most of the system should be **Python**.

C++ should be used only where low-level telemetry or low-overhead instrumentation is needed.

## Build in Python

Use Python for:

```text
experiment planning
candidate generation
config patching
trial orchestration
subprocess management
benchmark control
metric aggregation
comparison logic
regression detection
report generation
database writes
cluster/CI adapters
PyTorch integration
```

Why?

Because Python gives you:

```text
fast iteration
easy PyTorch integration
easy config handling
easy process control
easy DB integration
easy experiment logic
```

This layer will change constantly. Do not bury it in C++.

---

## Build in C++ only when needed

Use C++ for:

```text
low-overhead telemetry agent
CUDA event timing helpers
NVML/DCGM wrappers
CUPTI-based tracing later
native memory/launch instrumentation
optional LD_PRELOAD-style hooks
```

C++ makes sense for:

```text
low-level runtime data
minimal overhead collection
framework-independent telemetry
future production daemon pieces
```

But do **not** build the experiment planner in C++.

That would slow you down.

---

## Practical split

```text
Python:
  80-90% of experiment runner

C++:
  10-20% low-level telemetry/runtime hooks
```

Recommended split:

```text
Python controller
  ├── generates configs
  ├── launches trials
  ├── collects Python/PyTorch metrics
  ├── calls native telemetry module
  ├── compares results
  └── writes reports

C++ telemetry module
  ├── NVML polling
  ├── CUDA event timing
  ├── CUPTI later
  └── low-overhead process metrics
```

Expose C++ to Python via:

```text
pybind11
C ABI + ctypes
gRPC/local socket if daemonized
```

For your current product stage, I would use:

```text
Python + pybind11 C++ extension
```

---

# 24. MVP implementation order

Build it in this order.

---

## Phase 1: Manual local benchmark runner

Status: **completed in MVP form**.

Goal:

```text
Run baseline and one candidate locally.
Compare throughput and loss.
```

Implement:

- [x] TrialConfig
- [x] TrialResult
- [x] LocalTrialExecutor
- [x] basic metrics.json
- [x] basic report.md

Support:

- [x] num_workers
- [x] pin_memory
- [x] batch_size candidate plumbing
- [x] mixed_precision candidate plumbing

---

## Phase 2: Candidate generation

Status: **completed in MVP form**.

Goal:

```text
Automatically generate 5-10 configs from a bottleneck diagnosis.
```

Input:

```json
{
  "primary_bottleneck": "input_pipeline_bound",
  "baseline_config": {
    "num_workers": 4,
    "pin_memory": false,
    "batch_size": 64
  }
}
```

Output:

```json
[
  {"num_workers": 8},
  {"num_workers": 8, "pin_memory": true},
  {"num_workers": 12, "pin_memory": true},
  {"num_workers": 8, "pin_memory": true, "prefetch_factor": 4}
]
```

Implemented routing:

- [x] `input_bound` / `input_pipeline_bound` -> dataloader worker, pin memory, prefetch candidates.
- [x] `copy_bound` -> pinned-memory-focused dataloader candidates.
- [x] `launch_bound` / `small_kernel_overhead` -> `torch.compile` and CUDA Graph candidates when validated actions are allowed.
- [x] `memory_pressure` / `memory_bound` -> allocator candidates, then mixed precision when validated actions are allowed.
- [x] `underutilized_gpu` -> batch size, mixed precision, and runtime candidates when validated actions are allowed.

---

## Phase 3: Safety validation

Status: **completed in MVP form**.

Goal:

```text
Reject unsafe candidates before running them.
```

Checks:

```text
[x] memory headroom
[x] GPU availability
[x] static vs dynamic shapes
[x] precision support
[x] known incompatible options
[ ] max runtime policy beyond per-trial timeout
[x] user policy
```

---

## Phase 4: Benchmark windows

Status: **completed in MVP form**.

Goal:

```text
Separate warmup from measurement.
```

Add:

```text
[x] warmup_steps
[x] measurement_steps
[x] repeat_count recorded in benchmark window
[x] timeout
```

Implemented:

- [x] `BenchmarkWindow` validates warmup, measurement, repeat, and timeout settings.
- [x] Runner and local executor share the same benchmark window object.
- [x] Trial artifacts include `benchmark_window.json`.
- [x] Derived summaries include a `measurement_window` scope when `per_step` data is present.
- [x] Metric extraction prefers `measurement_window`, then `steady_state`, then full run.
- [ ] Executing multiple repeats is deferred to Phase 6 noise-aware comparison.

---

## Phase 5: Regression detection

Status: **completed in MVP form**.

Goal:

```text
Reject trials that improve speed but damage quality.
```

Add:

```text
[x] loss comparison
[x] NaN/Inf checks
[x] output tolerance checks
[x] OOM detection via exit/guard failures
[ ] broader stability checks beyond current step-time and memory guards
```

Implemented:

- [x] `step_end.payload.loss` is preserved in derived `per_step` summaries.
- [x] `measurement_window.run_summary` includes loss summary fields when losses are present.
- [x] `TrialResult.quality_metrics` stores extracted quality evidence.
- [x] Quality gates reject final-loss regression, trial loss divergence, NaN/Inf losses, and reported output drift.
- [x] CLI exposes `--max-final-loss-regression`, `--max-loss-divergence`, `--output-abs-tolerance`, and `--allow-nonfinite-loss`.

---

## Phase 6: Noise-aware comparator

Goal:

```text
Stop declaring tiny noisy gains as wins.
```

Add:

```text
multiple repeats
median comparison
variance estimate
minimum improvement threshold
confidence labels
```

Confidence levels:

```text
high
medium
low
inconclusive
```

---

## Phase 7: Database persistence

Goal:

```text
Every experiment is stored and queryable.
```

Start with SQLite.

Then move to Postgres.

---

## Phase 8: CI integration

Goal:

```text
Run benchmark experiments on code changes.
```

Support:

```text
GitHub Actions
Buildkite
GitLab CI
self-hosted GPU runners
```

---

## Phase 9: Cluster integration

Goal:

```text
Submit trials to Slurm/Kubernetes/Ray.
```

This comes later. Do not start here.

---

# 25. Minimal v1 code abstraction

Something like this:

```python
@dataclass
class TrialConfig:
    name: str
    patch: dict
    env: dict
    risk_level: str


@dataclass
class TrialResult:
    name: str
    status: str
    metrics: dict
    regressions: list
    artifacts_path: str


class ExperimentRunner:
    def __init__(self, executor, comparator, store):
        self.executor = executor
        self.comparator = comparator
        self.store = store

    def run(self, plan):
        baseline = self.executor.run(plan.baseline)

        results = []
        for candidate in plan.candidates:
            if not candidate.is_safe():
                continue

            result = self.executor.run(candidate)
            results.append(result)

            if result.status in {"oom", "nan", "timeout"}:
                continue

        report = self.comparator.compare(baseline, results)
        self.store.save(plan, baseline, results, report)

        return report
```

Keep it boring at first.

---

# 26. What this should integrate with from previous steps

Your Step 10 runner should consume outputs from previous components.

```text
Telemetry collector
  -> gives raw performance data

Common schema / performance IR
  -> normalizes metrics

Bottleneck classifier
  -> identifies likely problem

Recommendation engine
  -> suggests candidate fixes

ROI ranker
  -> prioritizes candidates

Safe autopilot actions
  -> defines allowed config changes

Experiment runner
  -> tests them safely
```

The flow becomes:

```text
profile workload
detect bottleneck
generate fixes
rank by ROI
run experiments
compare results
recommend winner
optionally apply config
```

That is a real product loop.

---

# 27. Biggest hidden traps

## Trap 1: Fake performance wins

Example:

```text
Trial skips validation.
Trial uses fewer batches.
Trial changes data order.
Trial uses smaller inputs.
Trial silently disables augmentation.
```

Prevent this with config hashing and data controls.

---

## Trap 2: Faster but worse model

Example:

```text
fp16 improves throughput but loss diverges later.
batch size improves utilization but hurts convergence.
torch.compile changes numerics subtly.
```

Prevent this with quality gates.

---

## Trap 3: Measuring compile overhead incorrectly

`torch.compile` may look worse if you include compile time.

Measure both:

```text
cold-start performance
steady-state performance
```

Both matter, but they answer different questions.

---

## Trap 4: Overfitting to benchmark window

A config may look good for 100 steps but bad for long training.

So label results properly:

```text
short benchmark win
not proven full-training win
```

---

## Trap 5: Cluster noise

Cloud GPUs vary due to:

```text
neighbor load
thermal state
power limits
network noise
storage noise
CPU contention
```

So cluster mode needs repeated trials and contamination detection.

---

# 28. Recommended build stack

For your current project, I’d use:

```text
Python:
  Typer or Click for CLI
  Pydantic for schemas
  SQLAlchemy for DB
  SQLite locally
  Postgres later
  Rich for terminal reports
  Jinja2 for Markdown/HTML reports
  PyTorch profiler integration
  subprocess/asyncio for local execution

C++:
  pybind11
  NVML wrapper
  CUDA event helpers
  CUPTI later

Storage:
  local filesystem for artifacts
  SQLite for MVP
  Postgres + object storage later

Cluster later:
  Slurm adapter
  Kubernetes Job adapter
  Ray adapter
```

---

# 29. Good v1 CLI experience

Example:

```bash
gpu-autopilot experiment run \
  --workload workload.yaml \
  --bottleneck input_pipeline_bound \
  --max-trials 8 \
  --benchmark-steps 100 \
  --quality-gate loss
```

Example output:

```text
Baseline:
  throughput: 812 samples/sec
  step time: 78.8 ms
  dataloader wait: 41%

Running 6 candidate configs...

[1/6] workers=8 pin_memory=true
  throughput: 994 samples/sec
  status: pass

[2/6] workers=12 pin_memory=true
  throughput: 1001 samples/sec
  p90 step time worsened
  status: inconclusive

[3/6] workers=8 pin_memory=true prefetch=4
  throughput: 1034 samples/sec
  status: pass

Best safe config:
  workers=8
  pin_memory=true
  prefetch_factor=4

Estimated improvement:
  +27.3% throughput
```

---

# 30. Final recommendation

Build the experiment runner mostly in **Python**, with a small C++ telemetry extension.

Your ideal split:

```text
Python:
  experiment logic
  orchestration
  config mutation
  benchmark execution
  result comparison
  quality gates
  DB/reporting
  cluster/CI adapters

C++:
  low-overhead GPU telemetry
  NVML/DCGM/CUPTI integrations
  CUDA timing helpers
```

The most important thing is not raw engineering complexity. It is **trust**.

Your product wins if it can safely say:

```text
I tested 8 valid configurations.
3 were rejected.
2 were inconclusive.
1 improved throughput by 23% with no detected quality regression.
Here is the exact config and evidence.
```

That is the moment where this becomes a real optimizer.
