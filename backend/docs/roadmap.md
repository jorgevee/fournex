# Roadmap

## v0.1 - Local collect + analyze (current)

### Shipped

- `frx collect -- python train.py` - wraps workload, samples GPU metrics, generates run bundle
- `frx analyze <run-dir-or-zip>` - bottleneck verdict, evidence, and recommendations
- `frx doctor` - environment dependency check
- `frx smoke-test` - end-to-end pipeline validation

- Bottleneck classifier: `input_bound`, `copy_bound`, `sync_bound`, `underutilized_gpu`, `memory_pressure`, `shape_instability`, `launch_bound`, `insufficient_telemetry`
- Rule-based recommendation engine (YAML rules + catalog)
- PyTorch profiler trace importer (Chrome JSON format)

### Known gaps

- No `frx upload` - bundles must be uploaded to the web UI manually
- Distributed (multi-GPU) traces not yet classified
- nvidia-smi sampling is coarse (1 s default); per-kernel GPU utilization requires profiler traces

## Coming soon

- Cloud upload and shareable run URLs
- Automated config optimization (private beta)
- Distributed workload support
- Continuous profiling mode
