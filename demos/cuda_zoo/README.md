# CUDA Antipattern Zoo

Four pairs of bad/good CUDA kernels. Each pair isolates one classic GPU
performance antipattern, compiles with `nvcc`, and is analysed by `frx compare`.

---

## Pairs

| Dir | Antipattern | Bad finding | Good outcome |
|-----|-------------|-------------|--------------|
| `01_uncoalesced` | Strided global memory access | `uncoalesced_access` | Stride-1 coalesced copy |
| `02_matmul_notiled` | Naive GEMM, no shared memory | `fp32_only_matmul`, `no_shared_memory_tiling` | TILE=16 shared-memory GEMM |
| `03_excess_sync` | `__syncthreads()` in reduction loop | `sync_inside_tight_loop` | Warp shuffle reduction |
| `04_register_pressure` | 32 live local scalars | `high_register_pressure` | Split into two 16-var passes |

---

## Requirements

- CUDA toolkit ≥ 11.0 (`nvcc` on PATH)
- Python environment with `fournex` installed (`pip install -e backend/python`)
- `frx` CLI available (installed with fournex)

---

## Run all pairs

```powershell
.\run_zoo.ps1
```

Or run a single pair manually:

```powershell
cd 01_uncoalesced
nvcc -O2 -o bad  bad.cu
nvcc -O2 -o good good.cu
frx compare bad.cu good.cu
```

---

## What to look for in `frx compare` output

**01_uncoalesced**
- `bad.cu` should flag `uncoalesced_access` — strided index pattern detected
- `good.cu` should have no memory access findings

**02_matmul_notiled**
- `bad.cu` should flag `fp32_only_matmul` (FP32 matmul, no TC intrinsics)
  and `no_shared_memory_tiling` (loop + global accesses, no `__shared__`)
- `good.cu` should have neither (uses `__shared__` tile)

**03_excess_sync**
- `bad.cu` should flag `sync_inside_tight_loop` (≥3 syncs, has loop)
- `good.cu` should not (only 1 `__syncthreads__` outside the warp loop)

**04_register_pressure**
- `bad.cu` should flag `high_register_pressure` (>20 local vars)
- `good.cu` should not (each kernel ≤16 local vars)

---

## Case-study harness (no GPU required)

Each pair ships a `case_study.yaml` manifest, so the whole diagnose → fix →
re-check → validate loop runs from the repo on any machine — no CUDA toolkit or
GPU needed (it uses static source analysis). It emits a reproducible proof bundle.

```bash
# list the available case studies
frx case-study list

# run one and write an artifact bundle (+ a GitHub-ready README.md)
frx case-study run uncoalesced_global_loads --emit-readme
```

The command exits non-zero if validation fails, so it doubles as a regression
gate. Each run validates three things against the manifest:

- the expected finding is **detected** in `bad.cu`,
- it is **resolved** in `good.cu`, and
- **no new** finding is introduced.

Artifacts land in `artifacts/case_studies/<name>/`:
`case_study.txt` (transcript), `diagnosis.txt`, `llm_brief.txt`, `evidence.json`,
`compare.json`, `validation.json`, and optionally `README.md`.

To layer hardware-counter evidence on top, capture before/after Nsight Compute
CSVs and point the manifest's `before_ncu` / `after_ncu` at them.

---

## Static-only vs. with NCU data

`frx compare` runs static analysis only. To add runtime confirmation:

```powershell
# profile bad.cu binary, then compare with NCU data
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,`
             l1tex__t_sector_hit_rate.pct,`
             sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active `
    --csv ./report.csv ./bad
frx profile --ncu report.csv
```
