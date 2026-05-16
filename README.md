# Fournex

**Open-source GPU performance profiler and bottleneck analyzer for PyTorch and CUDA.**

[![PyPI](https://img.shields.io/pypi/v/fournex)](https://pypi.org/project/fournex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Fournex finds GPU performance bottlenecks and turns profiler evidence into ranked, actionable recommendations. It supports PyTorch training-loop telemetry, PTX static analysis, CUDA source inspection, Nsight Compute CSV ingestion, and before/after CUDA comparisons.

---

## Install

```bash
pip install fournex
```

Requires Python 3.10+. A CUDA-capable GPU is needed for live profiling; PTX, CUDA source, and imported Nsight Compute CSV analysis run without one.

---

## 60-second demo

### Profile a CUDA workload

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
     Tier: advanced   Score: 0.84   Triggered by: rule_ncu_uncoalesced_access

     Why:
       Global load transactions average more than 4 sectors per request...

     Actions:
       1. Ensure adjacent threads access adjacent memory addresses (stride-1).
       2. Restructure array-of-structs (AoS) layouts to struct-of-arrays (SoA).
       3. Align buffers to 128-byte cache line boundaries.

     Validation:
       - DRAM throughput % should decrease after improving coalescing.
       - L2 hit rate should improve as fewer cache lines are fetched.
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

### PTX static findings

Reported by `frx analyze kernel.ptx` and `frx profile --ptx kernel.ptx`.

| Code | Signal |
|---|---|
| `register_spills_detected` | PTX local-memory spill loads/stores detected |
| `high_global_memory_ratio` | Global-memory-heavy instruction mix with little shared-memory use |
| `fp64_detected` | FP64 arithmetic or data movement detected |

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

---

## CLI Reference

```bash
# CUDA profiling — one command from workload to full report
frx profile -- ./my_kernel_app                               # run NCU + report
frx profile --preset memory -- python train.py               # memory-focused preset
frx profile --ncu ncu_report.csv                             # analyze saved CSV
frx profile --ptx kernel.ptx                                 # PTX static analysis
frx profile --ncu report.csv --json                          # JSON output

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

# Compare two versions
frx analyze --before before.ptx --after after.ptx [--json]
frx analyze --before before.csv --after after.csv [--json]

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

### frx profile — one-command bottleneck report

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

### Before/after comparison

```bash
frx analyze --before baseline.ptx --after optimized.ptx
frx analyze --before baseline.csv --after optimized.csv
```

For the strongest comparison, provide source, PTX, and NCU CSV for both sides:

```bash
frx analyze \
  --before-source baseline.cu --before-ptx baseline.ptx --before-ncu baseline.csv \
  --after-source optimized.cu --after-ptx optimized.ptx --after-ncu optimized.csv
```

---

## REST API

Fournex also ships a FastAPI backend for integration use cases.

```bash
cd backend
uvicorn api:app --reload
```

### CUDA static inspection — `POST /cuda/static-inspect`

```bash
curl -X POST http://127.0.0.1:8000/cuda/static-inspect \
  -H "Content-Type: application/json" \
  -d '{"files": [{"filename": "kernel.cu", "content": "<source>"}], "gpu_model": "A100"}'
```

### PTX analysis — `POST /ptx/analyze`

```bash
curl -X POST http://127.0.0.1:8000/ptx/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": "<ptx text>", "filename": "kernel.ptx"}'
```

### Nsight Compute analysis — `POST /ncu/analyze`

```bash
curl -X POST http://127.0.0.1:8000/ncu/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": "<csv text>", "filename": "report.csv"}'
```

### Implementation comparison — `POST /compare`

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
