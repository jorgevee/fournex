# Fournex

**Open-source GPU performance profiler, bottleneck analyzer, and LLM optimization brief generator for PyTorch and CUDA.**

[![PyPI](https://img.shields.io/pypi/v/fournex)](https://pypi.org/project/fournex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Fournex finds GPU performance bottlenecks and turns profiler evidence into ranked, actionable recommendations. It supports PyTorch training-loop telemetry, PTX static analysis, CUDA source inspection, Nsight Compute CSV ingestion, before/after CUDA comparisons, and LLM-ready optimization briefs.

---

## Install

```bash
pip install fournex
```

Requires Python 3.10+. A CUDA-capable GPU is needed for live profiling; PTX, CUDA source, and imported Nsight Compute CSV analysis run without one.

---

## 60-second demo

### Profile a CUDA kernel

```bash
# One command: run NCU and get a full detailed report
frx profile -- ./my_kernel_app

# Or analyze an existing NCU CSV
frx profile --ncu ncu_report.csv
```

Example output:

```text
====================================================================
  Fournex - CUDA Performance Profile
  Source  : ncu_report.csv
  Kernels : 3
  Confidence: high
====================================================================

VERDICT
  Primary bottleneck : memory_bandwidth_bound
  Also detected      : l1_cache_thrashing, uncoalesced_access

MEASURED METRICS
  Status  Metric                           Value        Threshold hint
  ----------------------------------------------------------------
  [!!]  DRAM Throughput                  87.4%        high >= 80% -> memory bandwidth bound
  [ok]  Tensor Core Utilization          62.1%        low < 10% -> underutilized TC units
  [!!]  L1 Hit Rate                      31.2%        low < 40% -> L1 cache thrashing
  [ok]  L2 Hit Rate                      72.8%        low < 50% -> L2 cache thrashing
  [!!]  Load Sectors/Request             9.3          high > 4 -> uncoalesced global loads (ideal = 1)
  [ok]  Issue Slot Utilization           78.4%        low < 40% -> low ILP / underutilized SMs

RECOMMENDATIONS (3)

  ----------------------------------------------------------------
  1. [HIGH] Improve global memory access coalescing
     Tier: next   Score: 0.84   Triggered by: rule_ncu_uncoalesced_access

     Why:
       Global load transactions average more than 4 sectors per request...

     Actions:
       1. Ensure adjacent threads access adjacent memory addresses (stride-1).
       2. Restructure array-of-structs (AoS) layouts to struct-of-arrays (SoA).
       3. Align buffers to 128-byte cache line boundaries.

     Validate:
       ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum_per_request,\
                     dram__throughput.avg.pct_of_peak_sustained_elapsed \
           --csv ./report.csv ./your_app
       <-- Load sectors/request: was 9.3; drops toward 1-4 after coalescing
       <-- DRAM throughput %: was 87.4%; decreases as fewer lines fetched

NEXT STEPS
  Re-run after changes:  frx profile -- <your_workload_command>
```

### Get an LLM-ready optimization brief

```bash
# Profile + generate a paste-ready brief for your LLM in one step
frx explain ncu_report.csv --src kernel.cu --out explain_output/

# Pipe directly to clipboard (macOS / Windows)
frx explain ncu_report.csv --src kernel.cu --prompt-only | pbcopy
frx explain ncu_report.csv --src kernel.cu --prompt-only | clip
```

Produces three files in `explain_output/`:

| File | Purpose |
|------|---------|
| `frx_summary.txt` | Human-readable bottleneck narrative with evidence and ranked recommendations |
| `frx_llm_prompt.txt` | Paste-ready prompt for any LLM with guardrails, evidence, speedup estimates, and validation targets |
| `frx_evidence.json` | Structured data for tools and agents |

Example `frx_summary.txt`:

```text
GPU Performance Summary
=======================

Source     : ncu_report.csv
Kernel     : kernel.cu
Layers     : ncu, source

PRIMARY ISSUE
  Inefficient global memory access  [confirmed, high severity]
  Confirmed by: ncu, source

EVIDENCE
  [NCU] DRAM Throughput: 87.4%  [SATURATED]
  [NCU] Load Sectors/Request: 9.3  [SATURATED]
  [NCU] L1 Cache Hit Rate: 31.2%  [POOR]
  [Source] strided_or_pitched access detected (line 43)

ROOT CAUSE
  Uncoalesced global memory access pattern. Adjacent threads in a warp
  access non-adjacent memory locations, causing 9x more DRAM transactions
  than necessary.

WHAT TO FIX FIRST
  1. [HIGH] Improve global memory access coalescing  (est. 10-25% speedup)
  2. [MEDIUM] Use shared memory tiling for global loads  (est. 20-40% speedup)

MISSING DATA
  PTX not analyzed
  -> Collect PTX with: nvcc -ptx kernel.cu -o kernel.ptx
```

Example `frx_llm_prompt.txt` (abridged):

```text
## CUDA Kernel Optimization Request

Rules for your response:
- Do NOT rewrite the entire kernel unless absolutely necessary
- Suggest the minimal targeted change that addresses the identified bottleneck
- Preserve correctness -- flag any correctness risks in your suggestions

**PRIMARY BOTTLENECK:** Inefficient Global Memory Access
**Confidence:** confirmed (2 layers confirm: ncu, source)
**Severity:** high

**EXPECTED IMPROVEMENT:**
Fix: Improve global memory access coalescing  [HIGH, Tier: next]
Estimated speedup: 10-25%
Global load transactions average more than 4 sectors per request, indicating
that adjacent threads in a warp are not accessing adjacent memory addresses.

Validation targets (re-check these after applying the fix):
  <-- Load sectors/request: was 9.3; drops toward 1-4 after coalescing
  <-- DRAM throughput %: was 87.4%; decreases as fewer cache lines are fetched

Re-profile with:
  ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum_per_request,... \
      --csv after_fix.csv ./your_app

**KERNEL SOURCE** (`kernel.cu`):
[source code here]

**SPECIFIC QUESTION:**
Why does this access pattern cause uncoalesced loads? What is the minimal
change to ensure consecutive threads access consecutive addresses?
```

### Benchmark before and after

```bash
# Compile, time, and compare two kernels
frx bench bad_linear_layer.cu good_linear_layer.cu --arch sm_120 --runs 5

# Also run NCU on both and report bottleneck changes
frx bench bad_linear_layer.cu good_linear_layer.cu --arch sm_120 --with-ncu
```

Example output:

```text
BENCH: bad_linear_layer.cu -> good_linear_layer.cu
===================================================
Compile:   OK (sm_120)

Timing (5 runs, 2 warmup, wall-clock):
  Before:  12.3 ms  [min 11.9  max 12.8  sd 0.3]
  After:    8.6 ms  [min  8.4  max  8.9  sd 0.2]
  Speedup: 1.43x

Bottleneck changes (NCU):
  RESOLVED:  memory_bandwidth_bound  (was 0.82)
  RESOLVED:  warp_stall_sync         (was 0.60)

No new bottlenecks introduced.

Verdict: improved  (2 resolved, 0 new)
```

### Analyze a training run

```bash
frx collect --name my-run -- python train.py
frx analyze runs/run-a1b2c3d4e5f6
```

Example output:

```text
--------------------------------------------------------
  GPU Autopilot - Run Analysis
  Run  : run-a1b2c3d4e5f6
  Scope: steady_state  (42 steps)
--------------------------------------------------------

VERDICT
  Primary Bottleneck : DataLoader / input pipeline
  Confidence         : high (0.91)
  Reason             : DataLoader consumed 74% of average step time

TOP RECOMMENDATIONS
  1. Increase DataLoader num_workers
  2. Tune prefetching / persistent workers
  3. Move expensive CPU transforms closer to the GPU path
```

### Compare before and after

```bash
frx analyze --before before.ptx --after after.ptx
frx analyze --before before.csv --after after.csv
```

Use `--json` for automation:

```bash
frx profile --ncu ncu_report.csv --json
frx analyze --before before.ptx --after after.ptx --json
```

---

## What Fournex Detects

### Training-level bottlenecks

| Label | Signal |
|---|---|
| `input_bound` | DataLoader / input pipeline consumes a large share of step time |
| `copy_bound` | Host-to-device transfer time is material |
| `sync_bound` | Synchronization wait is material |
| `underutilized_gpu` | GPU utilization is low without another dominant stall |
| `memory_pressure` | GPU memory pressure is high |
| `shape_instability` | Batch or tensor shapes vary enough to disrupt steady execution |
| `launch_bound` | Many small kernels / low utilization pattern |
| `insufficient_telemetry` | Not enough trace data to make a reliable call |

### NCU kernel bottlenecks

Reported by `frx profile` and `frx analyze ncu_report.csv`.

| Label | Signal |
|---|---|
| `memory_bandwidth_bound` | High DRAM throughput plus memory stalls |
| `warp_stall_memory` | Memory-related warp stalls dominate |
| `warp_stall_sync` | Barrier/synchronization stalls dominate |
| `l1_cache_thrashing` | L1 cache hit rate below 40% |
| `l2_cache_thrashing` | L2 cache hit rate below 50% |
| `uncoalesced_access` | Global load sectors per request exceed 4 (ideal = 1) |
| `tensor_core_underutilized` | Tensor core utilization is low |
| `low_issue_efficiency` | Issue slot utilization is low |
| `low_warp_scheduler_utilization` | Warp scheduler is underutilized |
| `occupancy_limited_by_registers` | Occupancy is limited by register pressure |
| `occupancy_limited_by_shared_memory` | Occupancy is limited by shared memory usage |
| `occupancy_limited` | Occupancy is low (other cause) |
| `insufficient_ncu_data` | NCU CSV had no parseable performance metrics |

### CUDA source findings

Reported by `frx analyze kernel.cu` and `frx compare`. Architecture-aware thresholds apply when `--gpu-model` is set.

**Memory**

| Code | Signal |
|---|---|
| `uncoalesced_access` | Strided array index pattern (alias-aware — catches `idx = tid * stride; src[idx]`) |
| `no_shared_memory_tiling` | Nested loop with global memory reads, no shared memory |
| `missing_vectorized_loads` | Coalesced scalar loads where `float4` / `int4` would reduce transaction count |

**Synchronization**

| Code | Signal |
|---|---|
| `unnecessary_syncthreads` | `__syncthreads()` with no visible shared memory write before it |
| `conditional_syncthreads` | `__syncthreads()` inside a thread-conditional branch (potential deadlock) |
| `sync_inside_tight_loop` | Three or more sync points inside a loop body |
| `warp_level_sync_misuse` | `__syncwarp()` combined with shared memory without `__syncthreads()` |

**Control flow**

| Code | Signal |
|---|---|
| `warp_divergence` | `threadIdx.x % N` or `threadIdx.x & N` in an `if`-condition |
| `excessive_branching` | More than 6 conditional branches in one kernel |
| `bounds_check_inside_hot_loop` | Four or more bounds checks inside a `for` loop |

**Occupancy**

| Code | Signal |
|---|---|
| `high_register_pressure` | More than 20 live local scalar variables in one kernel |
| `poor_block_size` | Block size below 32 threads or not a multiple of 32 |
| `low_theoretical_occupancy` | Estimated occupancy below 25% at the configured block size |

**Tensor cores**

| Code | Signal |
|---|---|
| `fp32_only_matmul` | GEMM pattern with FP32 only and no tensor core intrinsics |
| `missing_wmma_mma_path` | GEMM pattern with FP16 data but no `wmma` / `mma` intrinsics |
| `dimensions_not_tensor_core_friendly` | Shared tile dimension greater than 8 and not a multiple of 16 |

### PTX static findings

Reported by `frx analyze kernel.ptx` and `frx profile --ptx kernel.ptx`.

| Code | Signal |
|---|---|
| `register_spills_detected` | PTX local-memory spill loads/stores detected |
| `high_global_memory_ratio` | Global-memory-heavy instruction mix with little shared-memory use |
| `no_shared_memory_usage` | No shared memory ops despite significant global memory traffic |
| `fp64_detected` | FP64 arithmetic or data movement detected |
| `tensor_core_intrinsics_used` | `wmma` or `mma` instructions present (informational) |
| `loop_detected` | Back-edge branches detected (informational — aids register analysis) |

---

## CLI Reference

```bash
# CUDA profiling -- one command from workload to full report
frx profile -- ./my_kernel_app                               # run NCU + report
frx profile --preset memory -- python train.py               # memory-focused preset
frx profile --ncu ncu_report.csv                             # analyze saved CSV
frx profile --ptx kernel.ptx                                 # PTX static analysis
frx profile --ncu report.csv --json                          # JSON output

# LLM-ready optimization brief
frx explain ncu_report.csv                                   # NCU only
frx explain ncu_report.csv --src kernel.cu                   # + static analysis layer
frx explain ncu_report.csv --src kernel.cu --out ./brief/    # write to directory
frx explain ncu_report.csv --src kernel.cu --prompt-only     # print prompt to stdout
frx explain ncu_report.csv --gpu-model rtx5060               # architecture-aware

# Compile and benchmark two kernels side-by-side
frx bench before.cu after.cu                                 # compile + time
frx bench before.cu after.cu --arch sm_120                   # explicit GPU arch (recommended)
frx bench before.cu after.cu --arch sm_120 --with-ncu        # + bottleneck diff
frx bench before.cu after.cu --runs 10 --warmup 3            # custom timing params
frx bench before.cu after.cu --build-flags "-DBUILD_EXEC -O3" # extra nvcc flags
frx bench before.cu after.cu --json                          # JSON output

# Collect and analyze PyTorch training telemetry
frx collect --name <name> [--out <dir>] -- python train.py
frx analyze <run-dir-or-zip> [--scope run|steady_state|auto] [--json]

# Analyze CUDA evidence files
frx analyze kernel.ptx [--json]
frx analyze kernel.cu [--gpu-model A100] [--json]
frx analyze ncu_report.csv [--json]

# Generate Nsight Compute commands (manual two-step alternative to frx profile)
frx ncu-command --list
frx ncu-command memory --output ncu_memory.csv -- ./my_kernel_app
frx ncu-command full --kernel-name regex:my_kernel --output ncu_full.csv -- ./my_kernel_app

# Compare two CUDA source files (kernel review / before-after)
frx compare baseline.cu optimized.cu                              # source-only
frx compare baseline.cu optimized.cu --gpu-model H100             # architecture-aware thresholds
frx compare baseline.cu optimized.cu --with-ptx                   # + PTX (requires nvcc)
frx compare baseline.cu optimized.cu --with-ncu                   # + runtime NCU (requires nvcc + ncu)
frx compare baseline.cu optimized.cu --ncu-a a.csv --ncu-b b.csv  # pre-existing NCU CSVs
frx compare baseline.cu optimized.cu --json                       # JSON output

# Compare with multiple evidence layers per side
frx analyze \
  --before-source before.cu --before-ptx before.ptx --before-ncu before.csv \
  --after-source after.cu  --after-ptx after.ptx  --after-ncu after.csv

# Utilities
frx doctor
frx smoke-test
frx tune --safe --max-trials 12 -- python train.py
```

---

## SDK Instrumentation

For richer per-step telemetry, instrument your training loop:

```python
import fournex as frx

frx.init(job_name="resnet-baseline")

for step, batch in enumerate(dataloader):
    with frx.step_context(step=step, batch=batch, model=model):
        with frx.phase("forward", step=step):
            loss = model(batch)
        with frx.phase("backward", step=step):
            loss.backward()
        with frx.phase("optimizer", step=step):
            optimizer.step()
```

Without SDK instrumentation, `frx collect` can still wrap the process, sample `nvidia-smi`, and import optional PyTorch profiler traces.

---

## PyTorch Profiler Integration

Export a Chrome-format trace from `torch.profiler` into `frx-job-run/profiler_trace.json`; `frx collect` imports it automatically.

```python
from torch.profiler import ProfilerActivity, profile

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train()

prof.export_chrome_trace("frx-job-run/profiler_trace.json")
```

```bash
frx collect --name prof-run -- python train.py
frx analyze runs/<run-id>
```

---

## CUDA Workflow

### frx profile -- one-command bottleneck report

`frx profile` is the recommended entry point for kernel-level analysis. It runs Nsight Compute on your workload and immediately prints a structured report with evidence, ranked recommendations, and validation steps.

```bash
# Run NCU on a workload and report
frx profile -- ./my_kernel_app
frx profile --preset memory -- python train.py

# Scope profiling to specific kernels
frx profile --kernel-name gemm --launch-skip 2 --launch-count 10 -- ./app

# Save the NCU CSV for later re-analysis
frx profile --out ncu_report.csv -- ./my_kernel_app
frx profile --ncu ncu_report.csv          # re-analyze without re-running
```

When `ncu` is not on PATH, `frx profile` prints the equivalent manual commands so you can collect and import the CSV separately.

### frx explain -- LLM-ready optimization brief

`frx explain` turns an NCU CSV (plus optional CUDA source) into three output files:

- `frx_summary.txt` — human-readable bottleneck narrative: primary issue, evidence, root cause, what to fix, missing data
- `frx_llm_prompt.txt` — paste-ready LLM prompt with guardrails, bottleneck-specific question, speedup estimates, and validation targets
- `frx_evidence.json` — structured JSON for tools and agents

```bash
frx explain ncu_report.csv --src kernel.cu --out explain_output/

# The prompt already contains everything the LLM needs:
#   - Primary bottleneck and confidence level
#   - Evidence from profiler + source analysis
#   - Expected speedup range
#   - Current measured values (was X, expect Y)
#   - Exact NCU command to verify the fix
#   - The kernel source code
#   - A bottleneck-specific targeted question

# Pipe directly to an LLM via clipboard:
frx explain ncu_report.csv --src kernel.cu --prompt-only | clip
```

The workflow:
1. `frx profile --ncu ncu_report.csv` -- identify bottleneck
2. `frx explain ncu_report.csv --src kernel.cu --prompt-only | clip` -- generate brief
3. Paste `frx_llm_prompt.txt` into Claude, ChatGPT, or any LLM
4. Apply the suggestion, recompile
5. `frx bench before.cu after.cu --arch sm_120 --with-ncu` -- validate speedup

### frx bench -- compile and benchmark two kernels

`frx bench` compiles two `.cu` files, times both with warmup runs discarded, and reports the speedup. With `--with-ncu` it also profiles both and shows which bottlenecks were resolved or introduced.

```bash
frx bench before.cu after.cu --arch sm_120 --runs 5
frx bench before.cu after.cu --arch sm_120 --with-ncu
```

Timing is wall-clock subprocess timing — correct when the kernel calls `cudaDeviceSynchronize()` before exit. Pass `--arch` matching your GPU (e.g., `sm_120` for RTX 5060, `sm_90` for H100) to ensure NCU can profile the binary rather than JIT-compiled code.

### Metric presets

| Preset | Use when | Covers |
|---|---|---|
| `memory` | DRAM bandwidth, cache, or coalescing issues | DRAM throughput, L1/L2 hit rates, sectors/request, memory stalls |
| `tensor` | GEMM/convolution/AMP performance | Tensor core utilization, issue utilization, occupancy |
| `occupancy` | Launch config or resource limits reduce active warps | Achieved occupancy, registers/thread, shared memory |
| `stalls` | Warp stall breakdown is needed | Memory, sync, scoreboard, dispatch, and scheduler stalls |
| `full` | Broadest Fournex CUDA diagnosis | Union of all presets (default) |

### PTX static analysis

Generate PTX and analyze it without a GPU:

```bash
nvcc -ptx kernel.cu -o kernel.ptx
frx analyze kernel.ptx
# or
frx profile --ptx kernel.ptx
```

### Architecture-aware scoring

Pass `--gpu-model` to apply thresholds tuned to the target architecture:

```bash
frx analyze kernel.cu --gpu-model h100
frx explain ncu_report.csv --gpu-model rtx5060
```

| GPU | SM version | Notes |
|---|---|---|
| RTX 30xx (Ampere consumer) | sm_86 | |
| A100 | sm_80 | |
| RTX 40xx (Ada) | sm_89 | |
| H100 | sm_90 | Larger shared memory limits; Hopper wgmma alignment (x64) |
| RTX 50xx (Blackwell consumer) | sm_120 | pc-sampling metrics not available in NCU |
| B100 / B200 | sm_100 | |

### Analysis depth

Fournex's diagnostic confidence scales with the evidence available. At every level, metrics that cannot be confirmed are explicitly marked unavailable rather than omitted or fabricated.

| Evidence level | What Fournex detects |
|---|---|
| Source only | 16 structural CUDA antipatterns: spurious sync, strided access (alias-aware), missing bounds guards, warp divergence, tensor core alignment, occupancy risks |
| Source + `--gpu-model` | Architecture-aware thresholds: shared memory limits, register pressure cutoffs, tensor core alignment requirements tuned to the target SM generation |
| Source + PTX | Compiler-confirmed signals: global vs. shared load counts, register pressure, spill detection, instruction mix |
| Source + NCU | Measured hardware bottlenecks: DRAM throughput, cache hit rates, warp stall breakdown, sectors per request -- with current values shown in each validation step |
| Source + PTX + NCU | Cross-layer reconciliation: source intent vs. compiled code vs. runtime behavior, with per-finding confidence labels and specific NCU metrics needed to upgrade low-confidence diagnoses |

### frx compare -- kernel review in one command

```bash
frx compare bad_linear_layer.cu good_linear_layer.cu
```

```text
==================================================================
  frx compare
  A: bad_linear_layer.cu
  B: good_linear_layer.cu
  Evidence: CUDA source
==================================================================

Winner: good_linear_layer.cu  (score 1.000 vs 0.814)

Resolved in B:
  missing bounds guard
  spurious __syncthreads()

Improved in B:
  launch configuration  (+0.10)
  synchronization overhead  (+0.25)

Root causes in A:
  !! Inefficient global memory access  [medium - source]
  !  Excessive synchronization  [medium - source]

-- Missing evidence -------------------------------------------

  Inefficient global memory access  [low-medium -> medium-high if confirmed]
    - Global load sectors per request
        l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum_per_request
        > 4 sectors/request confirms uncoalesced access
    - DRAM throughput
        dram__throughput.avg.pct_of_peak_sustained_elapsed
        high % confirms memory bandwidth pressure

    Run:
      ncu --metrics l1tex__t_sectors...,dram__throughput... \
          --csv ./report.csv ./your_kernel

    Or collect everything:
      ncu --set full --csv ./report.csv ./your_kernel

==================================================================
```

Add `--with-ptx` (requires `nvcc`) to unlock register efficiency scores. Add `--ncu-a`/`--ncu-b` with pre-existing NCU CSVs to measure runtime stalls, DRAM throughput, and tensor core utilization.

### Demo: bad vs. good linear layer

`backend/demos/` contains a runnable before/after comparison of a naive vs. tiled matrix-multiply kernel:

```bash
# Source-only (no GPU or compiler required)
python backend/demos/demo_01_bad_vs_good_linear_layer.py

# With PTX for compiler-level evidence (requires nvcc)
python backend/demos/demo_01_bad_vs_good_linear_layer.py --with-ptx
```

Example output:

```text
FINDINGS RESOLVED IN B
  !  [medium] missing_obvious_bounds_guard
  !  [medium] unnecessary_syncthreads

EVIDENCE TABLE
  Finding                       Source  PTX   NCU   Confidence
  missing_obvious_bounds_guard  yes     n/a   n/a   medium
  unnecessary_syncthreads       yes     n/a   n/a   medium
  strided_or_pitched (removed)  yes     n/a   n/a   medium
  shared_memory_tiling (added)  yes     n/a   n/a   high

SCORECARD
  launch efficiency    A: [##################  ]  0.90
                       B: [####################]  1.00  (+0.10)  <-- winner
  sync efficiency      A: [###############     ]  0.75
                       B: [####################]  1.00  (+0.25)  <-- winner

VERDICT
  Winner:   good_linear_layer.cu
  Score A:  0.814    Score B:  1.000    Delta: +0.186
```

### CUDA antipattern zoo

`demos/cuda_zoo/` contains four pairs of bad/good kernels, each isolating one classic GPU performance antipattern. No GPU or compiler required -- `frx compare` runs static analysis only.

| Pair | Antipattern | Static rule triggered |
|---|---|---|
| `01_uncoalesced` | Strided global memory access | `uncoalesced_access` |
| `02_matmul_notiled` | Naive FP32 GEMM, no shared memory | `fp32_only_matmul`, `no_shared_memory_tiling` |
| `03_excess_sync` | Redundant `__syncthreads()` inside loop | `sync_inside_tight_loop` |
| `04_register_pressure` | 32 live local scalars in one kernel | `high_register_pressure` |

```bash
# Analyse all pairs
cd demos/cuda_zoo
.\run_zoo.ps1          # PowerShell

# Or analyse a single pair
frx compare 01_uncoalesced/bad.cu 01_uncoalesced/good.cu
```

---

## REST API

Fournex also ships a FastAPI backend for integration use cases.

```bash
cd backend
uvicorn api:app --reload
```

### CUDA static inspection -- `POST /cuda/static-inspect`

```bash
curl -X POST http://127.0.0.1:8000/cuda/static-inspect \
  -H "Content-Type: application/json" \
  -d '{"files": [{"filename": "kernel.cu", "content": "<source>"}], "gpu_model": "A100"}'
```

### PTX analysis -- `POST /ptx/analyze`

```bash
curl -X POST http://127.0.0.1:8000/ptx/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": "<ptx text>", "filename": "kernel.ptx"}'
```

### Nsight Compute analysis -- `POST /ncu/analyze`

```bash
curl -X POST http://127.0.0.1:8000/ncu/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": "<csv text>", "filename": "report.csv"}'
```

### Implementation comparison -- `POST /compare`

```bash
curl -X POST http://127.0.0.1:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
    "a": {"label": "baseline", "ptx": "<ptx text>", "ncu_csv": "<csv>"},
    "b": {"label": "optimized", "ptx": "<ptx text>", "ncu_csv": "<csv>"}
  }'
```

---

## Development Setup

```bash
git clone https://github.com/jorgevee/fournex.git
cd fournex
pip install fournex
frx doctor
frx smoke-test
```

Run tests:

```bash
pytest backend/tests/python/
```

---

## Contributing

Pull requests are welcome. Open an issue first to discuss proposed changes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions.

---

## License

MIT - see [LICENSE](LICENSE).
