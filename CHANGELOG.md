# Changelog

All notable changes to **Fournex** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Note: entries prior to adopting this changelog were reconstructed from git tags and
> release history, so historical version boundaries are approximate.

## [Unreleased]

## [0.2.9] - 2026-06-16

### Added
- **Case-study harness** (`frx case-study run|list`): turns a bad/good CUDA
  kernel pair into a validated, reproducible optimization proof. Runs on static
  source analysis (no GPU/CUDA toolkit required), validates detected→resolved→
  no-regression against a `case_study.yaml` manifest, and emits an artifact
  bundle (transcript, diagnosis, LLM brief, evidence/compare/validation JSON,
  optional README). Ships manifests for the four `demos/cuda_zoo` antipatterns.
- Global `-v`/`--verbose` and `--debug` flags: structured logging to stderr.
- This changelog.

### Fixed
- `fournex.__version__` is now single-sourced from installed package metadata
  (`importlib.metadata`) instead of a hardcoded literal, fixing a drift where
  `__init__.py` reported `0.2.7` while `pyproject.toml` was `0.2.8`.
- **Telemetry durability:** the Python event trace now streams to disk during
  the run (flushed periodically) when artifacts will be persisted, so a hard
  crash/OOM/kill no longer loses all buffered telemetry. The canonical trace is
  still rewritten at clean exit; `flush()` now flushes the stream. Event buffer
  writes are lock-guarded for multi-thread safety.
- **nvidia-smi sampler resilience:** a transient `nvidia-smi` failure no longer
  kills GPU sampling for the rest of the run — it logs once, keeps sampling, and
  backs off after repeated failures.
- **Native persistence guard:** persistence now fails loudly instead of silently
  writing empty artifacts when the native backend is active (the Python event
  buffer is empty by design under native); native owns its own finalization.

## [0.2.8] - 2026-06-11
- `frx explain` training-telemetry path: auto-detected run directories,
  bottleneck-specific LLM questions, training-aware brief rendering.

## [0.2.7] - 2026-06-07
- `frx explain` brief enrichment: top kernels, roofline, and occupancy summary
  surfaced in the LLM prompt with gating.

## [0.2.6] - 2026-06-02
- Explain-brief enrichment and training-telemetry groundwork.

## [0.2.5] - 2026-05-29
- Framework Abstraction Tax meta-classifier (framework/runtime overhead vs.
  hardware/data-pipeline) on the telemetry path.

## [0.2.4] - 2026-05-25
- Roofline / MFU analysis and per-kernel attribution with opportunity scoring.

## [0.2.3] - 2026-05-23
- Release maintenance and analysis refinements.

## [0.2.2] - 2026-05-20
- Release maintenance.

## [0.2.1] - 2026-05-18
- Evidence Reconciliation Engine: merges source / PTX / NCU / profiler signals into
  unified diagnoses with confidence labels; `/reconcile` endpoint.

## [0.2.0] - 2026-05-16
- CLI fixes and consolidation.

## [0.1.9] - 2026-05-15
- CLI changes and NCU updates.

## [0.1.4] - 2026-05-11
- Packaging fix: bundle recommendation/rule YAML files in the wheel.

## [0.1.3] - 2026-05-10
- PTX analysis (register pressure, spill detection, instruction mix, control flow),
  Nsight Compute ingestion with bottleneck classification, and side-by-side
  implementation comparison.

## [0.1.0] - 2026-05-07
- Initial public release: profiler + analyzer CLI, bottleneck detection,
  rule-based recommendations, schemas, and docs.

[Unreleased]: https://github.com/jorgevee/fournex/compare/v0.2.9...HEAD
[0.2.9]: https://github.com/jorgevee/fournex/releases/tag/v0.2.9
[0.2.8]: https://github.com/jorgevee/fournex/releases/tag/v0.2.8
[0.2.7]: https://github.com/jorgevee/fournex/releases/tag/v0.2.7
[0.2.6]: https://github.com/jorgevee/fournex/releases/tag/v0.2.6
[0.2.5]: https://github.com/jorgevee/fournex/releases/tag/v0.2.5
[0.2.4]: https://github.com/jorgevee/fournex/releases/tag/v0.2.4
[0.2.3]: https://github.com/jorgevee/fournex/releases/tag/v0.2.3
[0.2.2]: https://github.com/jorgevee/fournex/releases/tag/v0.2.2
[0.2.1]: https://github.com/jorgevee/fournex/releases/tag/v0.2.1
[0.2.0]: https://github.com/jorgevee/fournex/releases/tag/v0.2.0
[0.1.9]: https://github.com/jorgevee/fournex/releases/tag/v0.1.9
[0.1.4]: https://github.com/jorgevee/fournex/releases/tag/v0.1.4
[0.1.3]: https://github.com/jorgevee/fournex/releases/tag/v0.1.3
[0.1.0]: https://github.com/jorgevee/fournex/releases/tag/v0.1.0
