# Fournex

**GPU performance profiler, bottleneck analyzer, and LLM optimization brief generator for PyTorch and CUDA.**

[![PyPI](https://img.shields.io/pypi/v/fournex)](https://pypi.org/project/fournex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Fournex tells you **why your GPU is slow and exactly what to fix**. It ingests Nsight Compute CSVs, PyTorch training telemetry, CUDA source, and PTX — then produces ranked, evidence-backed recommendations and paste-ready LLM briefs.

---

## Install

```bash
pip install fournex
```

Python 3.10+. A CUDA GPU is needed for live profiling; NCU CSV, PTX, and source analysis run without one.

---

## Quick start

```bash
# First time? Let Fournex detect your setup and show what to run:
frx init

# Profile a kernel + generate LLM brief in one step
frx profile --ncu ncu_report.csv --explain

# Collect training telemetry + generate LLM brief in one step
frx collect --name my-run --explain -- python train.py
```

Example `frx profile` output:

```
VERDICT
  Primary bottleneck : memory_bandwidth_bound
  Also detected      : l1_cache_thrashing, uncoalesced_access

MEASURED METRICS
  [!!] DRAM Throughput          87.4%   high >= 80% → memory bandwidth bound
  [!!] L1 Hit Rate              31.2%   low < 40%  → L1 cache thrashing
  [!!] Load Sectors/Request      9.3    high > 4   → uncoalesced global loads

RECOMMENDATIONS
  1. [HIGH] Improve global memory access coalescing
     Ensure adjacent threads access adjacent addresses (stride-1).
     Restructure AoS → SoA layouts. Align buffers to 128-byte boundaries.

     Validate:
       ncu --metrics l1tex__t_sectors...,dram__throughput... --csv after.csv ./app
       <-- Load sectors/request: was 9.3; drops toward 1-4 after coalescing
```

---

## What it detects

| Layer | What Fournex finds |
|---|---|
| **NCU CSV** | DRAM bottlenecks, warp stalls, cache thrashing, uncoalesced access, tensor core underutilization, occupancy limits |
| **PyTorch telemetry** | DataLoader stalls, H2D copy overhead, sync-bound steps, launch fragmentation, shape instability |
| **CUDA source** | 16 structural antipatterns — strided access, warp divergence, spurious sync, register pressure, tensor core alignment |
| **PTX** | Register spills, global-memory-heavy instruction mix, FP64 usage, missing shared memory |
| **Cross-layer** | Reconciled confidence labels: source intent vs. compiled code vs. runtime behavior |

### Framework Abstraction Tax

When profiler telemetry is available, Fournex also reports a **Framework Abstraction Tax** — a 0–100 score for how much GPU idle is attributable to framework/runtime overhead (launch fragmentation, Python dispatch, missing graph capture) vs. hardware limits or the data pipeline:

```
FRAMEWORK ABSTRACTION TAX
  Score              : 74/100 (high)
  Contributors:
   - Kernel launch fragmentation
   - Missing graph capture (opportunity) (inferred)
   - Unfused elementwise operations (opportunity) (inferred)
```

Contributors marked `(inferred)` are opportunities Fournex reasons about from existing signals — not assertions that a feature is disabled.

---

## Core commands

```bash
# Kernel profiling
frx profile -- ./my_app                          # run NCU + report
frx profile --ncu report.csv                     # analyze saved CSV
frx profile --ncu report.csv --gpu-model h100    # architecture-aware roofline
frx profile --ncu report.csv --arch-profile h100-overrides.yaml  # custom specs

# LLM brief — auto-detects CSV (kernel) or run directory (training)
frx profile --ncu report.csv --explain          # profile + brief in one command
frx explain report.csv --src kernel.cu --out ./brief/
frx explain runs/my-run --out ./brief/
frx explain report.csv --prompt-only | clip     # pipe to clipboard

# Training telemetry
frx collect --name <name> -- python train.py
frx analyze <run-dir> [--scope run|steady_state|auto] [--json]

# Before/after comparison
frx compare baseline.cu optimized.cu --gpu-model h100
frx compare baseline.cu optimized.cu --ncu-a a.csv --ncu-b b.csv
frx analyze --before before.csv --after after.csv

# Benchmark two kernels
frx bench before.cu after.cu --arch sm_120 --with-ncu

# Utilities
frx init                                            # guided setup + SDK snippet
frx init --patch train.py                           # auto-add SDK lines to your script
frx ncu-command full --output report.csv -- ./app   # print NCU command
frx doctor                                          # detailed environment check
```

### `--gpu-model` and `--arch-profile`

Fournex auto-detects your GPU when PyTorch is available (`torch.cuda.get_device_name()`). Pass `--gpu-model` explicitly to override or when running without PyTorch:

```bash
frx profile --ncu report.csv --gpu-model h100
frx explain report.csv --gpu-model rtx5060
```

Use `--arch-profile` to override hardware specs with a YAML file (useful for custom hardware or pre-production SKUs):

```yaml
# h100-sxm.yaml
profiles:
  h100:
    peak_fp32_tflops: 60.0
    peak_memory_bw_gbps: 3900.0
```

Supported GPU families: RTX 30xx (sm_86), A100 (sm_80), RTX 40xx (sm_89), H100 (sm_90), RTX 50xx / B100 / B200 (sm_120 / sm_100).

---

## LLM workflow

`frx explain` works the same way for both CUDA kernels and PyTorch training runs — same three output files, same paste-into-LLM step.

**CUDA kernel:**
1. `frx profile --ncu report.csv --explain` — identify bottleneck + generate brief in one step
2. Paste `frx_llm_prompt.txt` into Claude / ChatGPT — get targeted fix suggestion
3. Apply, recompile
4. `frx bench before.cu after.cu --arch sm_120 --with-ncu` — validate

**PyTorch training run:**
1. `frx collect --name my-run --explain -- python train.py` — collect telemetry + generate brief in one step
2. Paste `frx_llm_prompt.txt` into Claude / ChatGPT — get targeted fix suggestion
3. Apply fix
4. `frx collect --name after-fix -- python train.py && frx analyze --before runs/my-run --after runs/after-fix` — validate

The brief includes: primary bottleneck, Framework Abstraction Tax (when relevant), per-phase timing breakdown, top recommendations with validation steps, and a bottleneck-specific question for the LLM.

---

## SDK instrumentation

For per-step PyTorch telemetry:

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

Without SDK instrumentation, `frx collect` still wraps the process, samples `nvidia-smi`, and imports PyTorch profiler Chrome traces automatically.

---

## REST API

```bash
cd backend && uvicorn api:app --reload
```

| Endpoint | Purpose |
|---|---|
| `POST /analyze` | PyTorch telemetry events → bottleneck report |
| `POST /ncu/analyze` | NCU CSV text → bottleneck report |
| `POST /ptx/analyze` | PTX text → static findings |
| `POST /cuda/static-inspect` | CUDA source → antipattern findings |
| `POST /compare` | Two evidence bundles → scorecard |
| `POST /reconcile` | Multi-layer evidence → reconciled diagnoses |

---

## Development

```bash
git clone https://github.com/jorgevee/fournex.git
cd fournex && pip install fournex
frx doctor && frx smoke-test
pytest backend/tests/python/
```

---

## License

MIT — see [LICENSE](LICENSE).
