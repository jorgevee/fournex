# Fournex

**Open-source GPU performance profiler and bottleneck analyzer for PyTorch.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fournex/fournex/blob/main/LICENSE)
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

## Links

- [GitHub](https://github.com/fournex/fournex)
- [Documentation](https://fournex.com/docs)
- [Website](https://fournex.com)
