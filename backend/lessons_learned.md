# Lessons Learned

## Environment and setup

* A Python virtual environment is recommended for local development even though the repo does not strictly require one yet.
* The reason is upcoming Phase 3 work will likely add Python-side dependencies such as `pybind11`, test tooling, and editable-package installs.
* Recommended local flow:
  * `python -m venv .venv`
  * `.\\.venv\\Scripts\\Activate.ps1`
  * `python -m pip install --upgrade pip`
  * `pip install -e .\\python`

we used git bash in windows 11
## Tooling observations

* `cmake` was not available in the current shell on April 7, 2026.
* `g++` was available at `C:\msys64\mingw64\bin\g++.exe`, which was enough for a compile check of the native sources.
* Because of that, native verification used a direct `g++` compile pass instead of a CMake configure/build step.
* The default `python` and `py` Windows app aliases were unusable in this shell.
* The working interpreter was `C:\Users\jorge\AppData\Local\Python\bin\python.exe` with Python 3.14.3 and `pip 25.3`.
* Any local setup instructions for this repo should prefer the real interpreter path or a venv created from it, not the Windows app alias.
* Verification that writes files should stay inside the workspace writable roots; using a system temp directory caused a permission failure in this environment.

## Repo observations

* The backend folder started nearly empty, so creating the target repo skeleton early was the right move.
* The parent repository already has unrelated deletions and additions outside `backend/`; those should not be touched casually while working on telemetry.

## Implementation notes

* Locking the event contract before deeper implementation reduced drift between docs, schema, Python stubs, and native stubs.
* The current native writer serializes payload values as strings for simplicity; this is acceptable for the skeleton but should be upgraded before richer payloads land.
* For local dev, the cleanest Python packaging flow is to keep editable install working without native compilation by default and make native extension builds opt-in.
* The schema already provides value during implementation: adding ad hoc payload keys like `loader_name` to `dataloader_span` immediately conflicted with the Phase 1 contract and had to be backed out.
* A good pattern for the SDK is to infer metadata from runtime objects when the schema already has a place for it, such as deriving `model_name` from `type(model).__name__` or `is_training` from `model.training`.
* For timing, the right early pattern is dual-mode collection: use CUDA events when available, but keep a `perf_counter_ns` fallback so the SDK stays testable and usable in local non-GPU development.
* The sampled profiler path should degrade gracefully: when `torch` is unavailable, emit `profiler_window` metadata and export a compact summary artifact instead of failing or silently doing nothing.
* The derived metrics reducer needs to tolerate mixed payload types because the current native writer serializes payload values as strings while the Python local path keeps native numeric types.
* Rule evidence should be normalized and bounded. In synthetic or very small local runs, sub-span durations can exceed enclosing step durations, so bottleneck fractions should be clamped to avoid impossible values.
* CUPTI should be treated as an explicitly requested debug capability, not a baseline dependency. When it is requested but not compiled into the current build, the engine should emit a warning event rather than fail initialization.
* The storage layout should separate raw and derived artifacts by directory, not just by filename. Using `.../raw/<run>.jsonl` and `.../derived/<run>_summary.json` makes replay and compare flows much cleaner.
* When a plan starts as a long design memo, the highest-leverage first step is usually to convert it into an execution document plus one validating schema/example artifact set.
* For the common IR layer, plain dataclasses plus explicit `validate()` and round-trip helpers are a good Phase 2 choice. They are fast to iterate on and keep Phase 3 mappers simple.
* For source mappers, preserve the raw record inside canonical `attrs` rather than trying to make the canonical schema carry every source-specific field directly.
* Sampled infrastructure sources like NVML fit naturally into canonical metrics first, with annotations added only when the sample clearly indicates a condition such as memory pressure or thermal pressure.
* Distributed source normalization should store both the normalized collective name and the raw source collective string. Normalized names make analytics clean; raw names preserve debug value.
* Data pipeline normalization should reuse the runtime stage vocabulary when it is already useful, then map those stages into canonical event types like `dataloader_wait` and `batch_ready` instead of inventing a second parallel taxonomy.
* Golden fixtures are the fastest way to stabilize mapper semantics while the IR is still evolving. They catch accidental normalization drift much earlier than larger end-to-end tests.
* The common IR needs its own derived-summary layer over canonical records. Reusing telemetry-specific reducers directly would couple the IR back to source-specific event shapes, which defeats part of the point of the normalization layer.
* For the bottleneck classifier, replayable event fixtures were the fastest way to turn the current rules into something testable. They exposed ranking and evidence behavior immediately without needing live telemetry collection.
* For run summaries, `dominant_stall_type` should prefer explicit non-compute stall buckets such as input, copy, and sync, then fall back to `compute_bound` only when no stall bucket is materially present. Treating compute as just another competing bucket made the summary disagree with the classifier in useful input-bound cases.
* In this workspace, `pytest` was not installed in the project virtual environment during bottleneck validation. The fallback was to verify the new assertions by running the fixture checks directly with `venv\\Scripts\\python.exe`.
* A thin diagnosis object on top of the ranked rule outputs is a good compatibility layer. It lets the backend keep the raw ranked labels for tuning while exposing a more stable user-facing contract with primary bottleneck, secondary bottlenecks, confidence, explanation bullets, and recommendations.
* Persistence tests should write under a workspace-local directory such as `traces/test_outputs/` rather than a system temp directory. In this environment, writes under the OS temp path hit permission issues even though ordinary workspace writes succeeded.
* Confidence scoring worked better once it was treated as a combination of primary score strength, separation from the next-best label, and agreement with the run-summary stall type. Using only raw rule score plus gap made clean single-signal `copy_bound` and `sync_bound` cases look weaker than they should.
* `shape_instability` was easy to validate with a tiny three-step fixture that alternates shape signatures while keeping timing otherwise steady. That is a good pattern for future taxonomy additions: isolate one changing signal and hold the rest of the run simple.
* For Phase 8, the right first primitive is scoped step selection rather than a full steady-state detector. A helper that summarizes and classifies an explicit set of step IDs gives us a stable base for later warmup filtering and steady-state window selection without rewriting the core reducer.
* A simple steady-state selector should stay explicit at first: skip the first `N` completed steps and optionally keep the last `K`. That is much easier to validate than an automatic detector, and it still gives downstream consumers a clean separation between whole-run and steady-state summaries.
* A combined run-plus-steady-state artifact is more informative than forcing one diagnosis over the whole trace. Mixed runs can legitimately be ambiguous at full-run scope while the steady-state slice remains decisive, and the backend should preserve that distinction rather than collapsing it away.
* The clean storage pattern is to keep the old single-summary persistence helper stable and add a separate persistence helper for the combined run-plus-steady-state artifact. That avoids breaking current callers while still making the richer Phase 8 report first-class.
* Once a backend default steady-state policy exists, it should be reported explicitly in artifact metadata. Hiding the applied selector makes diagnostics harder to interpret and makes it impossible to tell whether a run/steady-state disagreement came from model behavior or from the selection policy itself.
