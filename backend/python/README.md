# Fournex

**Open-source GPU performance profiler and bottleneck analyzer for PyTorch.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/jorgevee/fournex/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Fournex wraps your training script, collects GPU telemetry, and tells you exactly what is slowing it down — with ranked, actionable recommendations.

## Install

```bash
pip install fournex
```

## Quick start

```bash
# Profile your workload
frx collect --name my-run -- python train.py

# Analyze and get recommendations
frx analyze runs/run-<id>

# Check your environment
frx doctor

# Validate the pipeline end-to-end
frx smoke-test
```

## Detected bottleneck types

| Label | Signal |
|---|---|
| `input_bound` | DataLoader wait ≥ 20% of step time |
| `copy_bound` | H2D transfer ≥ 15% of step time |
| `sync_bound` | Sync wait ≥ 10% of step time |
| `underutilized_gpu` | GPU utilization < 35% |
| `memory_pressure` | Peak memory ratio ≥ 90% |
| `shape_instability` | Shape volatility ≥ 30% |
| `launch_bound` | Low utilization + profiler windows, no dominant stall |
| `insufficient_telemetry` | No timing or GPU utilization data |

## Safe config benchmarking

```bash
frx tune --safe --max-trials 12 -- python train.py
```

Fournex sweeps DataLoader and runtime configs, benchmarks each one, and recommends the fastest safe candidate — without changing your code.

Interrupted or repeated tune runs can reuse completed trial artifacts:

```bash
frx tune --resume runs/tune-<id> -- python train.py
```

`--resume` reuses a trial only when the saved `config.yaml`, `benchmark_window.json`, and `metrics.json` match the current workload command and benchmark settings.

## CUDA static analysis

`frx analyze` accepts `.cu` and `.cuh` files directly and reports kernel-level antipatterns, launch configuration issues, and occupancy estimates without requiring a GPU:

```bash
frx analyze kernel.cu
frx analyze kernel.cu --gpu-model RTX4090
frx analyze kernel.cu --output-json
```

Detected antipatterns span five categories across 22 rules:

| Category | Example findings |
|---|---|
| Memory | `uncoalesced_access`, `no_shared_memory_tiling`, `missing_vectorized_loads` |
| Synchronization | `unnecessary_syncthreads`, `conditional_syncthreads`, `sync_inside_tight_loop` |
| Control flow | `warp_divergence`, `excessive_branching`, `bounds_check_inside_hot_loop` |
| Occupancy | `large_static_shared_memory`, `possible_shared_memory_bank_conflict`, `high_register_pressure` |
| Tensor cores | `fp32_only_matmul`, `missing_wmma_mma_path`, `dimensions_not_tensor_core_friendly` |

## PTX and Nsight Compute analysis

`frx analyze` also accepts PTX assembly files and Nsight Compute CSV exports:

```bash
frx analyze kernel.ptx
frx analyze profile.csv
```

## Kernel comparison

Compare two CUDA source files across all available evidence layers:

```bash
# Source-only diff
frx compare baseline.cu optimized.cu

# With PTX and NCU profiling (requires nvcc + ncu on PATH)
frx compare baseline.cu optimized.cu --with-ptx --with-ncu

# Supply pre-collected NCU CSVs
frx compare baseline.cu optimized.cu --ncu-a baseline.csv --ncu-b optimized.csv
```

The comparison report scores each kernel across memory, compute, occupancy, and instruction-mix dimensions, with a verdict and reconciled diagnosis from all available layers.

## Recommendation validation commands

Every recommendation includes an NCU command to confirm the fix worked:

```
Validate:
  ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum_per_request \
      --csv ./report.csv ./your_app
  <-- Global load sectors/request: drops toward 1-4 (target: 4.0)
```

## Adding antipattern rules

Rules live in `fournex/cuda_rules/` as YAML files, one per finding:

```yaml
id: my_new_rule
scope: kernel          # kernel | launch | occupancy
category: memory
severity: medium       # high | medium | low
confidence: medium
message: "Description with optional {signal_name} interpolation."
conditions:
  strided_or_pitched: true    # bool equality
  for_count_gte: 1            # numeric >=
recommendations:
  - rec_ncu_improve_coalescing
ncu_signals:
  sectors_per_request_gt: 4.0
architecture_overrides: {}
```

Drop the file into the matching category subfolder — the engine picks it up automatically with no code changes required.

Available signals for `scope: kernel` conditions include: `has_shared`, `has_sync`, `has_tc`, `has_fp16`, `has_matmul`, `has_loop`, `has_thread_indexing`, `has_bounds_guard`, `strided_or_pitched`, `likely_coalesced_1d`, `vectorized`, `bank_conflict_risk`, `tc_unfriendly_dims`, `warp_divergence_pattern`, `conditional_syncthreads_pattern`, `sync_count`, `for_count`, `global_access_count`, `branch_count`, `bounds_check_count`, `local_var_count`, `max_shared_bytes`.

## Links

- [GitHub](https://github.com/jorgevee/fournex)
- [Documentation](https://fournex.com/docs)
- [Website](https://www.fournex.com)
