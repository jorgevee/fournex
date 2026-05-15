# Fournex

**Open-source GPU performance profiler and bottleneck analyzer for PyTorch and CUDA.**

[![PyPI](https://img.shields.io/pypi/v/fournex)](https://pypi.org/project/fournex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Fournex helps developers find GPU performance bottlenecks and turn profiler evidence into ranked recommendations. It supports PyTorch training-loop telemetry, PTX static analysis, CUDA source inspection, Nsight Compute CSV ingestion, and before/after CUDA comparisons.

---

## Install

```bash
pip install fournex
```

Requires Python 3.10+. A CUDA-capable GPU is recommended for runtime profiling; PTX, CUDA source, and imported Nsight Compute CSV analysis can run without live GPU access.

---

## 60-second demo

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

EVIDENCE
  - avg_dataloader_fraction=0.74 exceeds threshold 0.20
  - avg_step_wall_time_ns=482000000 (482 ms per step)

PERFORMANCE SNAPSHOT
  Avg GPU Utilization : 23.4%
  Avg Step Time       : 482.000 ms
  Throughput          : 2.1 steps/sec
  Dominant Stall      : dataloader

TOP RECOMMENDATIONS
  1. Increase DataLoader num_workers
  2. Tune prefetching / persistent workers
  3. Move expensive CPU transforms closer to the GPU path
```

### Analyze CUDA evidence directly

```bash
frx analyze kernel.ptx
frx analyze kernel.cu
frx analyze ncu_report.csv
```

### Compare before and after

```bash
frx analyze --before before.ptx --after after.ptx
frx analyze --before before.csv --after after.csv
```

Use JSON for automation:

```bash
frx analyze ncu_report.csv --json
frx analyze --before before.ptx --after after.ptx --json
```

JSON output is wrapped as:

```json
{
  "mode": "ptx",
  "result": {}
}
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

### CUDA / kernel-level bottlenecks

| Label | Signal |
|---|---|
| `ptx_register_spills` | PTX local-memory spill loads/stores detected |
| `ptx_high_register_pressure` | High per-thread register use |
| `ptx_global_memory_heavy` | Global-memory-heavy PTX with little shared-memory use |
| `ptx_fp64_usage` | FP64 arithmetic or FP64 data movement detected |
| `ptx_branch_divergence_risk` | High conditional branch density |
| `memory_bandwidth_bound` | High DRAM throughput plus memory stalls in Nsight Compute data |
| `warp_stall_memory` | Memory-related warp stalls dominate |
| `warp_stall_sync` | Barrier/synchronization stalls dominate |
| `cache_thrashing` | Low L1 or L2 cache hit rate |
| `tensor_core_underutilized` | Tensor core utilization is low where tensor cores may apply |
| `low_issue_efficiency` | Issue slot utilization is low |
| `insufficient_ncu_data` | NCU CSV had no parseable performance metrics |

---

## CLI Reference

```bash
# Collect and analyze PyTorch training telemetry
frx collect --name <name> [--out <dir>] -- python train.py
frx analyze <run-dir-or-zip> [--scope run|steady_state|auto] [--json]

# Analyze CUDA evidence files
frx analyze kernel.ptx [--json]
frx analyze kernel.cu [--gpu-model A100] [--json]
frx analyze ncu_report.csv [--json]

# Generate Nsight Compute collection commands
frx ncu-command --list
frx ncu-command memory --output ncu_memory.csv -- ./my_kernel_app
frx ncu-command full --kernel-name regex:my_kernel --output ncu_full.csv -- ./my_kernel_app

# Compare two versions
frx analyze --before before.ptx --after after.ptx [--json]
frx analyze --before before.csv --after after.csv [--json]

# Compare with multiple evidence layers per side
frx analyze \
  --before-source before.cu --before-ptx before.ptx --before-ncu before.csv \
  --after-source after.cu --after-ptx after.ptx --after-ncu after.csv

# Utilities
frx doctor
frx smoke-test
frx tune --safe --max-trials 12 -- python train.py
```

`--baseline` and `--optimized` are still accepted as deprecated aliases for NCU before/after comparison. Prefer `--before` and `--after`.

Full documentation: **[fournex.com/docs](https://fournex.com/docs)**

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

### PTX analysis

Generate PTX with your CUDA toolchain, then analyze it:

```bash
nvcc -ptx kernel.cu -o kernel.ptx
frx analyze kernel.ptx
```

PTX analysis is static. It flags likely risks such as register spills, high register pressure, global-memory-heavy instruction mix, FP64 use, branch divergence risk, and atomics.

### Nsight Compute CSV analysis

Export a CSV from Nsight Compute and pass it to Fournex:

```bash
frx ncu-command full --output ncu_report.csv -- ./my_kernel_app
# prints:
# ncu --csv --target-processes all --metrics <fournex metrics> ./my_kernel_app > ncu_report.csv

frx analyze ncu_report.csv
```

NCU analysis uses measured runtime counters, so its confidence is higher when hardware counters are available. On some WSL2/WDDM setups, NCU hardware counters may be inaccessible.

Fournex ships metric presets for common CUDA investigations:

| Preset | Use when | Includes |
|---|---|---|
| `memory` | DRAM bandwidth, cache, or memory stalls are suspected | DRAM throughput, L1/L2 hit rates, memory stall reasons |
| `tensor` | GEMM/convolution/AMP performance is suspected | Tensor core utilization, issue utilization, achieved occupancy |
| `occupancy` | Launch config or resource pressure may limit active warps | Achieved occupancy, block size, registers/thread, shared memory |
| `stalls` | You need warp stall reason breakdown | Memory, sync, scoreboard, dispatch, and scheduler stall metrics |
| `full` | You want the broadest Fournex CUDA diagnosis | Union of all presets |

Examples:

```bash
frx ncu-command memory --output ncu_memory.csv -- ./my_kernel_app
frx ncu-command tensor --kernel-name regex:gemm --output ncu_tensor.csv -- ./my_kernel_app
frx ncu-command occupancy --launch-skip 5 --launch-count 20 --output ncu_occ.csv -- ./my_kernel_app
frx ncu-command stalls --target-processes all --output ncu_stalls.csv -- python train.py
```

`frx analyze ncu_report.csv` validates the CSV before reporting. If key metrics are missing, Fournex prints warnings such as "Memory diagnosis may be incomplete" or "Occupancy diagnosis may be incomplete." Malformed CSVs fail with actionable errors for missing kernel, metric, or value columns.

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

### CUDA static inspection - `POST /cuda/static-inspect`

```bash
curl -X POST http://127.0.0.1:8000/cuda/static-inspect \
  -H "Content-Type: application/json" \
  -d '{"files": [{"filename": "kernel.cu", "content": "<source>"}], "gpu_model": "A100"}'
```

### PTX analysis - `POST /ptx/analyze`

```bash
curl -X POST http://127.0.0.1:8000/ptx/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": "<ptx text>", "filename": "kernel.ptx"}'
```

### Nsight Compute analysis - `POST /ncu/analyze`

```bash
curl -X POST http://127.0.0.1:8000/ncu/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": "<csv text>", "filename": "report.csv"}'
```

### Implementation comparison - `POST /compare`

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
pip install -e backend/python
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

## Coming Soon

- `frx upload` - push run bundles to the cloud for shareable analysis URLs
- Automated config optimization
- Distributed and multi-GPU workload support

---

## License

MIT - see [LICENSE](LICENSE).
