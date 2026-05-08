# Fournex

**Open-source GPU performance profiler and bottleneck analyzer for PyTorch.**

[![PyPI](https://img.shields.io/pypi/v/fournex)](https://pypi.org/project/fournex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

Fournex wraps your training script, collects GPU telemetry, and tells you exactly what is slowing it down — dataloader starvation, H2D copy overhead, kernel launch bottlenecks, memory pressure, and more — with ranked, actionable recommendations.

---

## Install

```bash
pip install fournex
```

Requires Python 3.10+. A CUDA-capable GPU is recommended; CPU-only mode works for trace analysis.

---

## 60-second demo

**Step 1 — Profile your workload**

```bash
frx collect --name my-run -- python train.py
```

```
frx collect completed
Run bundle: runs/run-a1b2c3d4e5f6
Zip bundle: runs/run-a1b2c3d4e5f6.zip

Captured (6 files):
  metadata.json
  run_config.yaml
  gpu_metrics.csv
  optional_logs.txt
  raw/trace.jsonl
  derived/summary.json
```

**Step 2 — Analyze**

```bash
frx analyze runs/run-a1b2c3d4e5f6
```

```
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
  Avg Memory Util     : 18.1%
  Peak Memory Pressure: 0.21
  Avg Step Time       : 482.000 ms
  Throughput          : 2.1 steps/sec
  Dominant Stall      : dataloader

TOP RECOMMENDATIONS (3 of 5)

  1. [HIGH] Increase DataLoader num_workers
     Effort: config  |  Risk: low  |  Score: 0.91
     DataLoader is the dominant bottleneck. More workers will overlap
     data loading with GPU compute.
     Actions:
       - Set num_workers=8 (or os.cpu_count() // 2)
       - Benchmark with num_workers=4, 8, 12 to find the sweet spot

  2. [HIGH] Enable pin_memory
     Effort: config  |  Risk: low  |  Score: 0.88
     Pinned memory enables faster host-to-device transfers for
     DataLoader output tensors.
     Actions:
       - Add pin_memory=True to DataLoader

  3. [MEDIUM] Move transforms to GPU
     Effort: medium  |  Risk: low  |  Score: 0.71
     CPU-side augmentation pipelines are a common source of
     DataLoader starvation.
     Actions:
       - Replace torchvision CPU transforms with GPU equivalents (v2 API)
```

---

## Detected bottleneck types

| Label | Signal |
|---|---|
| `input_bound` | DataLoader wait ≥ 20% of step time |
| `copy_bound` | H2D transfer ≥ 15% of step time |
| `sync_bound` | Sync wait ≥ 10% of step time |
| `underutilized_gpu` | GPU utilization < 35% |
| `memory_pressure` | Peak memory ratio ≥ 90% |
| `shape_instability` | Shape volatility ratio ≥ 30% |
| `launch_bound` | Low utilization + profiler windows, no dominant stall |
| `insufficient_telemetry` | No timing or GPU utilization data |

---

## CLI reference

```
frx collect --name <name> [--out <dir>] -- python train.py
frx analyze <run-dir> [--scope run|steady_state|auto] [--json]
frx doctor
frx smoke-test
```

Full documentation: **[fournex.com/docs](https://fournex.com/docs)**

---

## SDK instrumentation (optional)

For richer per-step telemetry, instrument your training loop:

```python
from fournex import AutopilotSession

with AutopilotSession(job_name="resnet-baseline") as session:
    for batch in dataloader:
        with session.step():
            with session.phase("forward"):
                loss = model(batch)
            with session.phase("backward"):
                loss.backward()
            optimizer.step()
```

Without the SDK, `frx collect` still works using `nvidia-smi` sampling and an optional PyTorch profiler trace import.

---

## PyTorch profiler integration

Export a Chrome-format trace from `torch.profiler` and pass it to `frx collect`:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train()

prof.export_chrome_trace("frx-job-run/profiler_trace.json")
```

```bash
frx collect --name prof-run -- python train.py
# frx automatically imports frx-job-run/profiler_trace.json
```

---

## Development setup

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

## Coming soon

- `frx upload` — push run bundles to the cloud for shareable analysis URLs
- Automated config optimization (private beta — [join the waitlist](https://fournex.com))
- Distributed (multi-GPU) workload support

---

## License

MIT — see [LICENSE](LICENSE).
