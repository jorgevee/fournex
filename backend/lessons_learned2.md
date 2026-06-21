# Kernel Inspector Notes

- Keep kernel inspection data as Common IR kernel attrs instead of creating a separate schema too early. `EventRecord.attrs` already lets us attach register count, shared memory, block/grid dimensions, occupancy estimates, and raw profiler metadata without breaking existing consumers.
- Occupancy from registers/shared memory is an estimate, not a measured value. The first implementation uses conservative default SM limits and reports limiting factors; callers should override device limits when GPU architecture is known.
- Nsight Compute CLI CSV exports vary by command flags and version. The parser needs to support long metric rows (`Kernel Name`, `Metric Name`, `Metric Value`) and direct summary columns. Static and dynamic shared memory metrics should be summed for per-block shared-memory pressure.
- PTX/SASS extraction should remain best-effort and tool-gated. `cuobjdump`/`nvdisasm` may not be installed in CI or user environments, so code should raise clear missing-tool errors and tests should not require NVIDIA tooling.
- PyTorch profiler traces can carry launch metadata in `args`, but field names are inconsistent. Use alias mapping for register count, shared memory, block dimensions, and grid dimensions instead of assuming one canonical spelling.
- Run summaries should expose both per-kernel rich metadata and compact aggregate fields. Frontend consumers need `kernel_count`, `kernels_with_launch_metadata`, average estimated occupancy, and a launch summary list.

# Static CUDA Intelligence Notes

- Treat `.cu` / `.cuh` static parsing as heuristic source intelligence, not as a compiler. Regex plus brace matching is enough for first-pass kernel detection, launch extraction, thread-indexing patterns, memory style tags, atomics, reductions, and obvious anti-patterns.
- Keep launch advice language conservative. The backend should say "safe recommended starting configurations" and require benchmarking before calling anything optimal.
- Shared-memory bank-conflict detection should start with simple red flags such as tile dimensions that are exact multiples of 32. This catches common unpadded `tile[32][32]` patterns without pretending to understand every access permutation.
- `__syncthreads()` checks need nuance. A barrier without visible shared memory is suspicious, and a barrier inside a thread-dependent branch is high risk, but static detection cannot prove correctness.
- Occupancy estimates are most useful when static source signals and profiler/imported signals converge. Static CUDA can infer block sizes from literal launches, while Nsight Compute supplies actual registers/thread and shared memory/block.
- JSON source-upload endpoints are easier to support than multipart initially. The frontend can send `{filename, content}` for `.cu` / `.cuh`; backend analysis remains pure and testable.

# Eval Harness Notes

- Fournex golden evals should stay event-stream based. The useful unit of coverage is a named list of telemetry events that runs through `summarize_run`, not a mocked diagnosis object or a synthetic PyTorch workload.
- Golden fixtures need more than one completed step. Single-step traces can hide averaging bugs because a raw value and an average collapse to the same number.
- Existing golden fixtures use local step IDs such as `1` and `2`. Any test that concatenates fixtures must rewrite the later fixture's step IDs first; otherwise the reducer merges unrelated work into the same per-step bucket. A stricter future cleanup could give every shared fixture a globally unique namespace, but that requires updating tests that currently assert exact step IDs.
- Profiler-derived launch evidence is only as strong as the event stream feeding it. `kernel_count`, `median_cuda_kernel_duration_us`, and `small_kernel_fraction` are aggregated by the reducer from `profiler_window` payloads; these golden tests validate reducer propagation and classification evidence, not raw trace parsing from a real profiler export.
- Recommendation eval negatives are most valuable near rule boundaries. Prefer exclusions that guard adjacent or unsafe recommendations, such as no CUDA Graphs for unstable shapes, no batch increase under memory pressure, and no dataloader tuning for copy/sync-only cases.
- ExperimentRunner speedup evals should use deterministic summary writers. The writer should explicitly vary `throughput_steps_per_sec` from the env var changed by the candidate, because wall-clock timing or a no-op candidate makes these tests flaky or meaningless.
- `frx analyze` must tolerate both legacy single-scope summaries and newer run/steady-state summaries. Zip analysis tests should assert successful loading and reporting without assuming one exact summary wrapper.
- Zip bundle extraction needs explicit path validation before `extractall`. Reject absolute paths and `..` segments, then verify resolved targets stay inside the temp extraction root.
## 5/10/26

# NCU Ingestion Notes

- NCU CSV exports two column layouts: long-format (`Kernel Name, Metric Name, Metric Unit, Metric Value` rows — one metric per row) and direct-column format (one kernel per row, metrics as columns). Both exist in the wild; the parser handles both via `_rows_to_kernel_summaries`. Metric aliases must be normalized with lowercasing + space/dash/dot replacement before alias lookup.
- Warp stall reasons use the pattern `smsp__pcsamplingdata_pct_of_utilization_issue_stalled_<reason>`. The reason suffix is arbitrary and GPU-version-dependent. A regex-free prefix-strip fallback (`_NCU_STALL_PREFIX`) handles all variants without an exhaustive alias table.
- Key stall groupings for bottleneck classification: memory stalls = `{memory_throttle, long_scoreboard, mio, lg, texture}`; sync stalls = `{barrier, wait}`; compute stalls = `{short_scoreboard, dispatch, not_selected}`. These groupings drive the `memory_stall_fraction` and `compute_stall_fraction` aggregates in `derive_ncu_run_summary`.
- Bottleneck score ordering matters: `tensor_core_underutilized` score = `1.0 - tc/100`, so tc=5% scores 0.95; `memory_bandwidth_bound` score = `dram/100`, so dram=82% scores 0.82. Tests that assert primary bottleneck ordering must use scenarios where the target metric is clearly the worst.
- The recommendation engine's `generate_recommendations` now accepts an optional `signals=` keyword arg so NCU analysis can pass pre-computed `extract_ncu_signals()` output. This avoids the need to fake an SDK-style run_summary when calling the engine from outside the SDK event pipeline.
- `parse_nsight_compute_csv_text(text)` was added alongside the path-based version to enable clean API endpoints and unit tests without writing temp files. Both share `_rows_to_kernel_summaries(rows)`. Prefer the text version in tests on Windows where `tmp_path` may fail due to `C:\Users\jorge\AppData\Local\Temp\pytest-of-jorge` permission issues.
- `KernelLaunchSummary` uses `slots=True`. New fields must be added in the correct position (all with defaults). The `_compute_derived_ncu_fields(summary)` helper reads from `summary.metrics` dict and populates the new typed fields; it runs after `launch_summary_from_attrs` which only knows about launch config fields.
- NCU metric unit inference: metrics ending in `_pct` → `"percent"`, metrics starting with `warp_stall_` → `"percent"` (stall fractions are percentages even though they don't end in `_pct`), `_bytes` → `"bytes"`, otherwise `"count"`.
- The `/ncu/analyze` API endpoint accepts `{content, filename, environment}` JSON — no file upload, no tempfiles. `analyze_ncu_csv_text` is the pure function that the endpoint delegates to.

# PTX Analysis Notes

- PTX `.entry` detection regex must handle both `visible` and `extern` qualifiers: `\.(?:visible\s+|extern\s+)?\.entry\s+(?P<name>\w+)`. A kernel body is delimited by matching braces from the first `{` after `.entry`; `_find_matching_brace` tracks depth and handles nested blocks (e.g., when a kernel body has brace-delimited inline data).
- Register pressure counting has two non-obvious rules: (1) predicate registers (`pred`) don't draw from the hardware register file — exclude them from `register_count` but keep them in parsing for completeness; (2) 64-bit types (`f64`, `b64`, `u64`, `s64`) consume 2 hardware registers each. The `_DOUBLE_WIDE` frozenset encodes this weight. Virtual register count N from `.reg .<type> %<prefix><N>` is the array size, not bytes.
- Local memory in PTX (`.local .align 4 .b8 __local_depot0[SIZE]`) always indicates register spills — the compiler uses local memory as a scratch pad when it runs out of hardware registers. `spill_load_count` and `spill_store_count` are the counts of `ld.local` and `st.local` instructions; they're derived from `instruction_mix["local_loads"]` and `instruction_mix["local_stores"]` rather than parsed separately.
- Instruction classification uses an ordered pattern list: first-match wins. This avoids double-counting — a `fma.rn.f64` line hits `fp64_ops` before it can also match `fp32_ops`. Global/shared/local memory ops must appear before arithmetic categories in the list so memory ops aren't mis-classified as compute.
- Loop detection is a heuristic via back-edge analysis: record each label's line position during parsing, then for each `bra $TARGET` check if `label_positions[target] < current_line`. This correctly identifies `@%p0 bra $L__loop_start` as a loop back-edge even without a full CFG. Unconditional `bra.uni` back-edges also count. The estimate can overcount if a kernel has multiple conditional branches to the same label — use `{target for ...}` (set) to deduplicate.
- The `instruction_line_no` counter increments only for non-label, non-directive lines. Labels and `.reg`/`.local`/`.param` directives are consumed before `instruction_count` is incremented. This means the `< line_no` back-edge check compares instruction ordinal positions, not raw line numbers — correct for the purpose of detecting backward jumps.
- `analyze_ptx_text` extracts `.version` and `.target` from the module header before any kernel body, so these fields are present even when no kernels were parsed (empty string input → `kernel_count=0`, version/target = `None`, findings = `[]`).
- Severity rules: `register_count > 128` → HIGH (virtually certain to spill at runtime); `register_count > 64` → MEDIUM (may reduce occupancy); `has_register_spills` → HIGH (already spilling). The 128 threshold comes from the hardware register file limit on older SM targets; modern GPUs (sm_80+) allow up to 255 but 128 remains a practical upper bound for good occupancy.
- `high_global_memory_ratio` fires when `(global_loads + global_stores) / instruction_count > 0.40`. `no_shared_memory_usage` fires only when `global_loads > 20` — short kernels with a few global ops are not worth flagging for missing shared memory.
- The `to_dict()` method on `PtxKernelAnalysis` uses `dataclasses.asdict()` which recursively converts nested dicts and lists. No custom serializer needed.
- Tests use inline PTX fixture strings (no files, no `tmp_path`) to avoid the Windows temp-directory permission issue. Each fixture exercises a specific feature combination: PTX_SIMPLE (clean registers, global memory), PTX_SPILL (local depot, ld/st.local), PTX_FP64_SFU (f64 ops, approx funcs), PTX_BRANCH (conditional bra, back-edge), PTX_SHARED (smem ops), PTX_TENSOR (wmma), PTX_HIGH_GLOBAL (>40% global ratio).

# Comparison Module Notes

- The comparison module (`comparison.py`) is intentionally a pure diff layer: it calls existing analyzers (`inspect_cuda_source`, `analyze_ptx_text`, `analyze_ncu_csv_text`) and diffs their outputs, but adds zero parsing logic. Any improvement to an underlying analyzer automatically improves comparison results.
- Each input layer (CUDA source, PTX, NCU CSV) is fully optional. When both sides lack a layer, the corresponding `*_diff` dict carries `"available": False` and the scorecard dimension returns `None`. The verdict normalizes weights across only the available dimensions so a partial comparison is still meaningful.
- NCU data always overrides PTX-derived estimates for `memory_efficiency` and `compute_efficiency`. This is intentional: hardware counters (cache hit rates, issue slot utilization) are measured facts; PTX instruction ratios are structural proxies. When both are present, trust the measured signal.
- The `TIE_THRESHOLD = 0.02` applies at two levels: per-dimension winner (whether to assign dim to `dimensions_won_by_a/b`) and overall verdict (whether `overall_winner` is "tie"). Using the same threshold at both levels avoids the paradox of an "overall winner" whose winning dimensions are all "tied".
- `register_efficiency` score formula: `SPILL_PENALTY * REG_SCORE` where `SPILL_PENALTY = 0.0` if any kernel spills. A spilling kernel gets score 0.0 regardless of how few virtual registers it declared — spilling is a cliff, not a slope. `REG_SCORE` is linear between 32 (→ 1.0) and 128 (→ 0.0), matching the thresholds where ptx_analysis emits `high_register_count` (64+) and `very_high_register_count` (128+).
- `_diff_numeric` uses `delta = val_b - val_a` (positive delta means B is higher). `higher_is_better` controls whether positive delta → `better=="b"`. For DRAM throughput, `higher_is_better=False` because high DRAM utilization means the kernel is memory-bandwidth-saturated. For cache hit rates, `higher_is_better=True`.
- `_diff_findings` works on finding codes (strings), not full finding dicts. Using codes as the key means that if both kernels have `register_spills_detected` with slightly different messages (e.g. different byte counts), they're still treated as the same finding. This is the right semantic — the finding type matters for the comparison, not the per-instance message.
- The FastAPI endpoint test is guarded with `pytest.importorskip("fastapi")` rather than a `@pytest.mark.skipif` decorator, consistent with how the rest of this suite handles optional dependencies (the pattern avoids the import-time decorator overhead and keeps the skip reason visible in the test output).
- `_weighted_avg_present` only uses weights for pairs where the value is not None. It renormalizes so the weights of absent sub-signals don't dilute the score. Example: if l2 cache hit rate is unavailable, l1 and dram split the weight between themselves (0.40/0.70 and 0.30/0.70 respectively). This avoids penalizing an implementation for missing NCU metrics.

## 5/17/26

# Scoring Calibration Notes

- Normalizing bottleneck scores to the theoretical maximum (e.g., 32 sectors/request for fully scattered access) produces scores that are misleadingly low for real-world bad values. In practice, `sectors/request > 11` is essentially worst-case — real kernels don't hit 32. Compress the denominator to a practical ceiling rather than the hardware maximum.
- The `uncoalesced_access` score was recalibrated: denominator changed from `31.0` to `10.0`. This maps sectors=8 → 0.70 and sectors=10.4 → 0.94, making the bottleneck rank correctly alongside other hardware signals. With the old formula, a kernel with 10.4 sectors/request (10x wasted bandwidth) scored only 0.30 and ranked 6th.
- Calibration bugs are silent: the bottleneck is still detected, tests still pass, but the score is too low to rise above lower-confidence signals in the sorted ranking. The only way to catch them is explicit score-band regression tests, not just presence tests.
- Add scoring regression tests at natural severity boundaries, not just presence/absence tests. For `uncoalesced_access`: low (just above threshold), medium-high (8 sectors), high (10+ sectors), capped (20+ sectors). This prevents future rule changes from quietly weakening real signals.
- When writing score regression thresholds, derive the expected values analytically from the formula first, then write the test. Don't write tests that just pass against whatever the current formula happens to produce — the formula might be wrong, as it was here.
- `_uncoalesced_summary` and `_uncoalesced_score` test helpers were factored out as module-level functions in `test_ncu_analysis.py` rather than inlining the full summary dict into each calibration test. This avoids copy-paste drift if the summary schema gains new required fields later.
# Before/After Comparison Demo Notes

- `compare_implementations()` is source-only by default; it produces a meaningful diff (findings resolved, structural changes, launch_efficiency scorecard) without PTX or NCU. The three higher-value scorecard dimensions (memory, compute, register efficiency) require PTX at minimum. Be explicit about this in demos — the "unavailable" rows are honest, not a bug.
- The static analyzer's `strided_or_pitched` style fires on variable names: the index must literally contain `stride`, `pitch`, `ld`, or `width` as a substring. A naive matmul using `B[k * N + col]` where `N` is just called `N` will NOT trigger it. Name the parameter `width` (or `stride`) in the bad kernel to get the detection.
- `unnecessary_syncthreads` and `conditional_syncthreads` were originally penalizing `launch_efficiency`, which is semantically wrong — they are synchronization issues, not launch config issues. Both were moved to a new `sync_efficiency` scorecard dimension (weight 0.20). The remaining weights were rebalanced: register 0.20->0.15, memory 0.30->0.25, compute 0.30->0.25, launch 0.20->0.15. Verdict normalization is dynamic, so adding a new dimension doesn't break existing comparisons that lack static source data.
- When adding a new scorecard dimension, the only test that needs updating is the one that enumerates dimension names by string (`test_response_schema_and_required_fields`). The verdict, tie, and scorecard score tests are dimension-agnostic and pass automatically.
- Demo output on Windows CP1252 terminals: avoid all non-ASCII characters in print strings (arrows, em-dashes, smart quotes). Use `->`, `-`, `--` equivalents. Unicode display issues show as `?` and cause `UnicodeEncodeError` on encode, not on print — so the error appears mid-output, not at startup.
- The `demos/` directory is for runnable demo scripts; the existing test kernels live in `tests/cuda_kernels/`. Keep them separate: test kernels are designed for specific PTX analysis assertions, demo kernels are designed to illustrate before/after narratives.

# Rule min_score thresholds must be re-checked whenever a bottleneck's scoring formula changes. The `uncoalesced_access` rule had `min_score: 0.10` because the old formula scored the detection floor (sectors ≈ 4.1) at ≈0.10 — the threshold was tuned to the formula. After changing the denominator, the minimum possible score jumped to 0.30, making `min_score: 0.10` a no-op. Updated to 0.30. General principle: `min_score` in a rule should equal or slightly exceed the formula's floor at the classifier's detection threshold, so the threshold is meaningful rather than trivially satisfied.

## 5/18/26

# Evidence Reconciliation Engine Notes

- `reconcile_evidence()` takes four optional kwargs (`static`, `ptx`, `ncu`, `profiler`) as already-parsed result dicts — not raw text. Callers call the appropriate analyzer first, then pass the result in. This keeps reconciliation a pure diff layer with no I/O.
- Signal extraction reuses `extract_ptx_signals()` and `extract_ncu_signals()` from `signals.py`. These functions expect `(summary_dict, bottlenecks_list)` — for PTX that's `ptx["run_summary"]` and `ptx["bottlenecks"]`; for NCU it's `ncu["ncu_run_summary"]` and `ncu["bottlenecks"]`. Don't pass the full result dict directly.
- Confidence is computed against `n_available` = the number of layers that (a) were provided AND (b) have a check function defined for the diagnosis. A layer that was provided but has no check for a given diagnosis does not inflate `n_available`. This prevents a profiler-only diagnosis from appearing less confident because NCU wasn't passed.
- For the catalog's `ptx_claims`, use PTX *bottleneck labels* (e.g., `ptx_global_memory_heavy`, `ptx_register_spills`) — not signal keys. These are what appear in `ptx["bottlenecks"]` and what feeds the `unreconciled["ptx"]` section. The check function uses signal keys (`ptx_global_memory_heavy` from `extract_ptx_signals`) which happen to match some bottleneck labels but are semantically different objects.
- `unreconciled` only includes layers that were actually provided. If `ncu=None`, the `ncu` key is absent from `unreconciled`, not present with an empty list.
- The catalog lambdas capture signal keys that must exist in the signals dicts. If `extract_ptx_signals` or `extract_ncu_signals` add or rename a signal key, the catalog lambda will silently return False (dict `.get()` returns None → bool False). Add a unit test for each diagnosis at the signal boundary if the catalog grows.
- 33 tests pass, covering all 6 diagnoses (single-layer + two-layer), confidence levels (medium/low-medium/high/medium-high/confirmed), unreconciled tracking, false-positive guards, and API endpoint stub.

# frx compare CLI Notes

- `frx compare A.cu B.cu` is a new top-level subcommand — positional args for the two files, not flags. Different from `frx analyze --before A --after B` which uses layer-specific flags.
- `--with-ptx` compiles both files to PTX first; `--with-ncu` compiles to executables and runs ncu. If the .cu files need a `main()`, pass `--build-flags "-DBUILD_EXECUTABLE"`.
- `--ncu-a`/`--ncu-b` accept pre-existing NCU CSVs and take priority over `--with-ncu`. This is the reliable path for users who already have profiling data.
- The report has 7 sections: header, Winner, Resolved, Regressions, Improved, Root causes (from reconciliation), Still unknown. Empty sections are omitted.
- "Still unknown" is derived from scorecard dimension availability. If NCU is absent: DRAM bandwidth, cache rates, tensor core, occupancy, warp stalls are always unknown. If PTX is absent: register usage. Correctly scales with evidence level.
- `--json` wraps output as `{"mode": "compare", "result": {"comparison": ..., "reconciliation": ...}}`.
- 15 CLI tests in `test_compare_cli.py` pass: source-only, JSON schema, custom labels, error cases, identical-file tie, pre-existing NCU CSVs unlocking memory efficiency dimension.

## 5/18/26 — Recommendation cards with validation commands

- Added `validation_steps` field to all 12 NCU recommendation entries in `catalog.yaml`. Each step carries: `metric` (exact NCU counter name), `label`, `direction` (decrease/increase/stable), `expected` (prose), and optionally `threshold_good`. Kept `validation_templates` (plain-text sentences) in place — NCU-backed recs render from `validation_steps` while non-NCU recs fall back to `validation_templates`.
- `engine.py` passes `"validation_steps": entry.get("validation_steps", [])` through in every generated rec dict. Zero-cost for recs that have no steps.
- CLI renders a `Validate:` section in both `_print_ncu_report_full()` (frx profile path) and `_print_recommendation_list()` (frx analyze path). Format: `ncu --metrics <comma-joined metric names> \\\n    --csv ./report.csv ./your_app` then one line per step with direction arrow (`<--` decrease, `-->` increase) + label + expected outcome + target threshold when available.
- The `stable` direction renders without an arrow — used for regression-guard metrics (e.g., DRAM throughput when reducing shared-memory footprint to improve occupancy should not worsen DRAM pressure).
- Renamed `_print_recommendation_list()` header from "TOP RECOMMENDATIONS" to "RECOMMENDATIONS" and added full card rendering (score, tier, triggered_by, Why, Actions up to 3, Validate). Updated one assertion in `test_cli_collector.py` that checked for the old header string.
- 3 new tests in `test_profile_cli.py`: Validate section has NCU command, has direction arrows, JSON recs include `validation_steps` list. Total test count: 359 passing.

## 5/18/26 — CUDA antipattern library expansion

- Added 14 new finding codes to `cuda_static.py`, expanding the library from 7 to 21 total rules across 5 categories.
- **Memory (3 new):** `uncoalesced_access` (from existing `strided_or_pitched` style tag), `no_shared_memory_tiling` (for_count >= 1 + global_access_count >= 3 + no shared), `missing_vectorized_loads` (1D coalesced + scalar + no shared + 2+ global accesses).
- **Synchronization (2 new):** `sync_inside_tight_loop` (3+ __syncthreads + loop pattern), `warp_level_sync_misuse` (__syncwarp + shared memory + no __syncthreads).
- **Control flow (3 new):** `warp_divergence` (threadIdx.x % or & in if-condition), `excessive_branching` (> 6 ifs), `bounds_check_inside_hot_loop` (4+ bounds checks alongside for loops).
- **Occupancy (1 new in kernel body, 2 in launch-level):** `high_register_pressure` (> 20 local scalar decls), `poor_block_size` (from launch config — < 32 → high, not multiple of 32 → medium), `low_theoretical_occupancy` (estimated occupancy < 25% at 256 threads/block).
- **Tensor cores (3 new):** `fp32_only_matmul` (GEMM pattern + no FP16 + no TC intrinsics), `missing_wmma_mma_path` (GEMM pattern + FP16 present + no wmma/mma), `dimensions_not_tensor_core_friendly` (shared tile dim > 8 and not multiple of 16).
- Launch-level findings come from new `_launch_findings()` function called from `build_static_cuda_report()` after kernel findings. Appended to `all_findings` and sorted with other findings.
- `no_shared_memory_tiling` threshold: the initial `for_count >= 2` was too strict (single-loop GEMM has only 1 for loop). Fixed to `for_count >= 1 and global_access_count >= 3`.
- 22 new tests in `test_cuda_static.py` — one per new finding code plus negative cases. Total test count: 381 passing.

## 5/19/26 — Architecture-aware scoring + "What evidence is missing?" feature

### What evidence is missing?

- Added `evidence_needed` dict to all 6 `_CATALOG` entries in `reconciliation.py`. Each key is a layer name (`"ncu"`); each value is a list of metric dicts with `metric`, `label`, `why`. The field is empty for layers (e.g., profiler, PTX) where targeted collection commands don't yet apply.
- `_missing_evidence_for(entry, layers_confirming, layers_available)` computes the gap: for each layer in `evidence_needed` that is NOT in `layers_confirming`, collect the metrics into the output. Returns `None` when all confirming layers have been provided (nothing missing).
- `confidence_if_confirmed` predicts the confidence level if the missing evidence were provided, using `_compute_confidence(n_confirming + 1, n_available)`. This helps users see whether collecting more data will meaningfully change the diagnosis.
- `ncu_command` is the targeted command: `ncu --metrics <comma-joined metric names> --csv ./report.csv ./your_kernel`. Full-collection fallback `ncu --set full ...` is always included.
- `what_evidence_is_missing()` is a convenience public function: reconciles and filters to only diagnoses with non-None `missing_evidence`. Useful for tools that only want actionable items.
- CLI renders the "Missing evidence" block in `_print_compare_report()` after upgrade hints. Shows confidence arrow (`low-medium → medium-high if confirmed`), metric list with why-text, and targeted + full ncu commands.
- `textwrap` must be a top-level import in `cli.py`, not a local import inside a function, because `_print_missing_evidence()` is called from `_print_compare_report()` which has its own control flow.

### Architecture-aware scoring

- `arch_profiles.py` separates *scoring calibration* (when to flag) from *hardware limits* (what the GPU can do). `kernel_inspector.py` owns hardware facts (thread limits, shared mem per SM); `arch_profiles.py` owns threshold tuning (flag at 48 KB on Turing, 96 KB on Hopper).
- Product names are normalized to sm versions (`h100` → `sm_90`) via `_PRODUCT_TO_SM`. The `resolve_sm_version()` function also accepts sm strings directly (`"sm_90"` → `"sm_90"`) and is case-insensitive. Unknown GPUs return `None` (not an error) so callers can fall back to the generic default.
- Architecture overrides in YAML use **full replacement**, not merge. When `architecture_overrides.sm_90.conditions` is present, it completely replaces the base `conditions` dict. This enables *signal substitution* (replacing `tc_unfriendly_dims: true` with `tc_unfriendly_dims_hopper: true` for Hopper's 64-alignment requirement), which is impossible with pure merge (which ANDs both conditions together).
- The merge approach would have worked fine for numeric threshold changes (same key, different value) but fails for signal substitution (different key entirely). Since all existing rules in this codebase have single-condition overrides that happen to work either way, the full-replace switch is backward-compatible.
- Two separate signals are always computed: `tc_unfriendly_dims` (checks `% 16 != 0`) for generic WMMA and `tc_unfriendly_dims_hopper` (checks `% 64 != 0`) for Hopper wgmma. Separating them at signal extraction time rather than at the engine level keeps the engine simple: it just does condition matching, not architecture-dependent signal selection.
- `device_limits_for_gpu()` does exact lookup first, then substring fallback. This prevents `"sm_80"` from accidentally matching `"sm_800"` if that were ever a valid product string. Sm-version strings (starting with `sm_`) are directly keyed in `GPU_DEVICE_LIMITS`.
- Smoke test to validate architecture-aware scoring: a 64 KB static shared memory kernel should fire `large_static_shared_memory` on T4 (Turing, 48 KB limit) but NOT on H100 (Hopper, 96 KB limit). Run with `inspect_cuda_source(src, gpu_model='h100')` and check `findings`.
- 32 tests in `test_arch_profiles.py` — covers `resolve_sm_version`, `get_arch_profile`, YAML override behavior for each tuned rule, `device_limits_for_gpu` with sm strings, and public API exports. Total test count: 427 passing (2 pre-existing Windows temp permission errors, not regressions).

## 5/19/26 — Recommendation validation plans (before/after current values)

- `validation_steps` in `catalog.yaml` already had `metric`, `direction`, `expected`, and `threshold_good`. The missing piece was `current_value` — the actual measured value before the fix, so the checklist reads "was 10.4 → drops toward 1-4" instead of just "expect sectors/request to drop."
- `_VALIDATION_METRIC_TO_SIGNAL` in `engine.py` maps NCU metric names (used in catalog validation steps) to signal dict keys (computed by `extract_ncu_signals()`). Only 10 of the catalog's ~12 unique metrics have corresponding signals; the other 2 (`bank_conflicts` sum, `local_load_bytes` sum) are per-kernel totals that don't average meaningfully across a run.
- `_attach_current_values(validation_steps, signals)` copies each step and adds `"current_value": signals.get(signal_key)`. Returns `None` when the metric isn't available — the CLI then omits "was X" for that step, which is correct rather than showing a misleading fallback.
- Three signals were missing from `extract_ncu_signals()` and needed to be added: `avg_barrier_stall_pct` (from `warp_stall_breakdown["barrier"]`), `avg_math_throttle_stall_pct` (from `warp_stall_breakdown["math_pipe_throttle"]`), and `avg_registers_per_thread` (from `ncu_summary["avg_registers_per_thread"]`).
- Signal fallback values (`l1_cache_hit_rate_pct: 100.0`, `l2_cache_hit_rate_pct: 100.0`, `avg_global_load_sectors_per_request: 1.0`) are misleading when used as `current_value`. They make absent data look like perfect measurements. Changed all three to return `None` when the underlying NCU metric is absent. The boolean signals (`l1_cache_miss_heavy`, `uncoalesced_global_loads`) already check `is not None`, so this is a safe change.
- The NCU metric name in `validation_steps` (e.g., `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum_per_request`) and the alias used in the parser (`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld`) are different variants. The catalog uses the "sum" variant because it's the correct name for the targeted `ncu --metrics` command. The signal mapping uses the signal key rather than the metric name, so the mismatch is invisible to the engine.
- CLI format: `{direction_arrow} {label}: was {current_value}; {expected} (target: {threshold})`. The "was X;" prefix is prepended to `expected` when `current_value` is not None. Applied in both rendering paths (`_print_recommendation_list` and `_print_ncu_report_full`).
- 8 new tests: 6 in `test_ncu_recommendations.py` (coalescing/TC/occupancy current_value, None when absent, all steps have key, barrier stall) and 2 in `test_profile_cli.py` (CLI shows "was 7.0", JSON has current_value field). Total: 435 passing.

## 5/19/26 — Real hardware profiling: Windows setup, sm_120, NCU wide format

# Windows CUDA Toolchain Setup Notes

- Windows nvcc (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe`) requires MSVC `cl.exe` as the host compiler. MinGW gcc is not supported ("Host compiler targets unsupported OS"). Need Visual Studio Build Tools 2022 (workload: "Desktop development with C++") + Windows SDK (`winget install Microsoft.WindowsSDK.10.0.26100`).
- After VS Build Tools install, always compile from a shell initialized by `vcvars64.bat`, not a plain PowerShell. `VsDevCmd.bat` has a vswhere dependency that fails silently; `vcvars64.bat` is more reliable: `cmd /c "vcvars64.bat && nvcc ..."`.
- Nsight Compute on Windows (`ncu.bat`) wraps `target\windows-desktop-win7-x64\ncu.exe`. Call the `.exe` directly when piping output — the `.bat` wrapper can swallow or reorder stdout/stderr in PowerShell pipes.

# RTX 5060 / sm_120 Notes

- RTX 5060 (GB206, Blackwell consumer) reports `sm_120`, not `sm_100`. Do not assume Blackwell = sm_100; sm_100 is the data-center B100/B200. Consumer Blackwell RTX 50xx series is sm_120.
- Compiling without `-arch=sm_120` produces a PTX JIT-compiled binary. NCU cannot profile JIT-compiled kernels ("No kernels were profiled"). Always pass `-arch=sm_120` (or whatever `cudaGetDeviceProperties().major.minor` reports) when compiling for NCU profiling.
- PC sampling metrics (`smsp__pcsamplingdata_pct_of_utilization_issue_stalled_*`) are not available on sm_120. Any NCU command requesting these metrics fails with "Failed to find metric regex". Warp stall analysis requires different metrics on Blackwell; remove these from the default metric set or gate behind an architecture check.
- NVIDIA WDDM hardware counter access requires either: (a) run NCU as Administrator, or (b) enable "Allow access to GPU performance counters to all users" in NVIDIA Control Panel. The Control Panel change may require a reboot. Even as admin, profiling WSL2 CUDA processes from Windows NCU fails ("No kernels were profiled") — only native Windows binaries can be profiled this way.

# NCU Output Format Notes

- NCU 2026.1.1 outputs the wide format (one row per kernel, all metrics as columns) regardless of `--page raw`. The `--page raw` flag no longer produces the tall format (`Kernel Name, Metric Name, Metric Unit, Metric Value`). Fournex's parser must handle the wide format natively — requiring users to manually convert is not a viable path.
- The wide format NCU CSV includes a unit row (row index 1, after the header) where all values are unit strings or empty. Data rows start at index 2. The DictReader unit row filter `r.get("Kernel Name", "").strip()` correctly drops the unit row since "Kernel Name" is empty there.
- The profiled app's stdout (`printf("done\n")`) leaks into the NCU stdout stream alongside the CSV. Filtering only `==PROF==` / `==WARNING==` lines is insufficient — non-`==` non-CSV lines (any line not starting with `"`) must also be stripped before passing to the CSV parser. The safe filter: `keep only lines starting with "`.
- PowerShell `Out-File` pipe from native exe can wrap long lines at terminal buffer width. For NCU wide-format rows (thousands of characters), use `Out-File -Width 9999` or redirect at the cmd level to avoid truncation.

# Kernel Access Pattern Analysis Notes

- The `strided_kernel` test case (STRIDE=32, `in[tid + i*count]`) is NOT uncoalesced at the warp level. For each loop iteration `i`, adjacent threads access adjacent memory addresses (`in[tid+i*count]` vs `in[tid+1+i*count]`), which is stride-1 — a coalesced pattern. NCU confirmed: sectors/request = 4.0 (at threshold, not above it). The bottleneck is `l1_cache_thrashing` + `memory_bandwidth_bound`, not `uncoalesced_access`. Comments in the original kernel were misleading ("adjacent threads read 32 elements apart" refers to per-thread sequential access, not warp-level coalescing).
- Fournex correctly diagnosed the strided kernel: primary bottleneck `l1_cache_thrashing` (L1 hit rate 0.0%, DRAM 94.7%), with `low_issue_efficiency` (issue slots 3.7%) as secondary. The `uncoalesced_access` rule did not fire (sectors = 4.0, threshold > 4) — the correct call. This validates that the bottleneck classifier distinguishes cache pressure from access pattern inefficiency even at the boundary.
- When designing eval kernels, verify the access pattern at the warp level, not just per-thread. A kernel where each thread touches widely-separated addresses can still be warp-coalesced if the thread index determines the base address (all threads in a warp access a contiguous 128-byte cache line together).

## 5/19/26 — NCU wide format + done-line leakage (discovered on real GPU)

- NCU 2026.x changed the default CSV output from tall format (one metric per row: `Kernel Name, Metric Name, Metric Unit, Metric Value`) to wide format (one kernel per row, metrics as column headers). The parser only handled tall format, which silently produced empty analysis on any user running a 2026.x NCU version.
- Wide format detection: if the CSV header has a kernel-name column but no `Metric Name`/`Metric Value` column, it's wide format. Route to `_wide_ncu_to_tall_rows()` which converts to synthetic tall rows that the existing `_rows_to_kernel_summaries` pipeline consumes unchanged — zero changes to downstream code.
- `_wide_ncu_to_tall_rows` skips the units row (row 1, whose first cell is literally `"Kernel Name"`), averages numeric metric values across multiple launches of the same kernel name, then emits one synthetic tall row per (kernel, metric) pair.
- Averaging across launches is correct: NCU re-profiles the same kernel multiple times for statistical stability and emits one row per launch. Treating each launch as a separate kernel would inflate `kernel_count` and skew all aggregate metrics.
- `done` leakage: NCU writes `done` as a plain text line to stdout after the CSV body. When the output is redirected to a file (`> ncu_output.csv`), this line is present at the end and trips the CSV parser. Fix: in `_text_to_csv_rows`, if any remaining line starts with `"` (all valid CSV rows start with a quoted kernel name), drop all non-quoted lines. This silently discards `done` and any other binary stdout leakage without affecting unquoted tall-format CSVs.
- The wide-format threshold was initially `> 4 columns` (guessing), which broke on minimal wide CSVs with exactly 2 columns. Changed to `>= 2 columns` (kernel name + at least one metric).
- Discovered by running the actual `frx profile --ncu` workflow on a real GPU for the first time. No test fixture had ever produced a wide-format CSV or a `done` line. Both issues were invisible in CI. Lesson: always run the end-to-end CLI workflow on real hardware before shipping a profiling feature.
- 3 new tests added: single-launch wide format, multi-launch averaging, done-line leakage. All 43 NCU tests pass.

## 5/19/26 — CUDA antipattern zoo

- Created `demos/cuda_zoo/` with 4 bad/good kernel pairs: uncoalesced access, naive GEMM, excess sync, register pressure. Each pair is a minimal runnable `.cu` file plus a 9-test static-analysis suite (`test_zoo_static.py`) that verifies `frx analyze` catches the right rule IDs on bad.cu and clears them on good.cu.
- The `strided_or_pitched` signal is detected by regex `\w+\s*\[[^\]]*\*\s*(stride|pitch|ld|width)[^\]]*\]` — the variable named `stride` (or `pitch` etc.) must appear **inside** the array subscript brackets. Writing `strided_idx = tid * STRIDE; src[strided_idx]` is invisible to the rule; the fix is `src[tid * stride]` with the multiplication inside. Zoo bad.cu passes `stride` as a function parameter (lowercase) so it appears in the subscript.
- `sync_inside_tight_loop` fires when `sync_count_gte: 3` AND `has_loop: true`. The naively written tree reduction loop has only 2 `__syncthreads()` in the source (one load barrier + one per iteration). Added a redundant post-loop drain sync (a common beginner mistake) to reach the threshold of 3 while keeping `has_loop: true`.
- Zoo tests revealed that static-analysis rule conditions match source-level string patterns, not runtime behavior. A kernel with one `__syncthreads()` inside a 10-iteration loop has `sync_count == 1`, not 10. This is by design (static analysis can't know iteration count) but is important to remember when writing demo kernels intended to trigger specific rules.
- `good.cu` counterparts passed all negative assertions without modification (no false positives), confirming the rules are directional: they fire on bad patterns but not on the clean equivalents.

## 5/19/26 — Semantic signal improvements: alias tracking + sync-in-loop detection

**Problem 1: strided alias false negatives.**
`src[tid * stride]` triggered `uncoalesced_access` but `int idx = tid * stride; src[idx]` did not. The detection regex only inspected array subscripts, not the expressions assigned to index variables. Real-world code nearly always computes the index into a local variable first.

Fix: `_strided_aliases()` in `cuda_static.py` scans typed variable declarations (`int/size_t/auto/…`) whose RHS contains a multiplication with a stride-keyword (`stride`, `pitch`, `ld`, `width`). Those variable names are collected as "strided aliases" and checked against subscripts in a second pass. Both the direct pattern and alias path contribute to the `strided_or_pitched` style — additive, no existing tests broken.

**Problem 2: sync_inside_tight_loop false positives and inaccurate semantics.**
The rule condition was `sync_count_gte: 3` AND `has_loop: true`. This misfired on any kernel with 3+ setup syncs and an unrelated loop, and also failed to catch the real pattern (sync appearing inside a loop body) for kernels with < 3 total syncs.

Fix: New `_count_syncs_in_loops(body)` helper in `engine.py` — for each `for`/`while` loop, extracts the braced body by matching parentheses and braces, then counts `__syncthreads()` inside. New signal `sync_in_loop_count` added to `extract_source_signals()`. Rule condition changed to `sync_in_loop_count_gte: 3`, removing both old conditions.

Threshold rationale: tiled GEMM (the canonical correct pattern) uses 2 syncs per loop iteration (load fence + compute fence). Three syncs per iteration is always excessive — no correct algorithm needs it. Zoo bad.cu redesigned to the "paranoid barrier" pattern: 3 syncs per reduction step (pre-read, post-write, drain).

Both improvements are additive to the signal layer and do not touch any downstream code (reconciliation, NCU, recommendations).

## 5/20/26 — frx compare rename + validation delta table

**frx compare now handles two modes:**
1. Source comparison: `frx compare baseline.cu optimized.cu` (unchanged behavior)
2. Evidence comparison / validation: `frx compare --before before.csv --after after.csv`

Mode 2 was previously `frx analyze --before/--after`. Routing is gated by `_has_comparison_args()` which already existed; adding `--before/--after/--before-ncu/--after-ncu/--before-ptx/--after-ptx/--before-source/--after-source/--before-label/--after-label` flags to `compare_parser` and calling `_analyze_comparison()` early cost zero new logic.

**Validation delta table (`_print_validation_delta_table`):**
`_METRIC_CONFIGS` in `ncu_comparison.py` extended from 2-tuple `(key, higher_is_better)` to 4-tuple `(key, higher_is_better, display_label, unit)`. The `_diff_metrics()` function now embeds `label` and `unit` into each delta dict. The CLI's `_print_ncu_comparison_report` calls the new helper which formats a BEFORE/AFTER table with human-readable column headers, direction tags `[+]`/`[-]`/`[=]`, and a `Result:` verdict line.

**Unit handling:** Metrics with unit `%` display values as `87.4%`; unitless metrics (stall fraction, load sectors) display as raw numbers. Column width calculation accounts for the `%` suffix when present. Deltas render as `(+30.0%)` or `(+0.12)` depending on unit.

**Lessons:**
- 4-tuple extension of `_METRIC_CONFIGS` is backward-compatible: all existing tests continued to pass because tests accessed `d["direction"]`, `d["delta"]`, etc., not the config tuple directly.
- Adding `label` and `unit` to the API output is forward-compatible — `diff_ncu_runs()` callers that only read `direction`/`delta` see no change; callers wanting human-readable output now have what they need without re-parsing metric key names.
- The `frx compare --before/--after` routing design avoids a separate `frx validate` subcommand (which would require a new argparse subparser). Routing is transparent to callers — same binary, same subcommand name, different argument pattern.

## 5/23/26 — GPU testing bug-fixes (frx explain real-hardware validation)

**Context:** 477 tests passed after running frx explain on RTX 5060 (sm_120). 4 bugs surfaced.

**Bug: register_pressure false positive at high occupancy (reconciliation.py)**
`estimate_occupancy()` reports "registers" as the limiting factor even when actual occupancy
is 92% (the hardware ceiling is just below 100%). The `ncu_check` in the reconciliation
catalog was firing on the raw `occupancy_limit_causes` field regardless of whether occupancy
was actually constrained. Fix: gate the confirmation on `not occupancy_good` (occupancy <= 60%).
A kernel running at 92% occupancy is not register-limited, regardless of what the NCU
occupancy estimator says about theoretical ceilings.

**Bug: tensor_core_underutilized fires when TC metric is absent (recommendations/signals.py)**
`tc = ncu_summary.get("avg_tensor_core_utilization_pct") or 0.0` — `None or 0.0 = 0.0 < 30 = True`.
Any fixture without a TC metric (e.g., excessive_sync) triggered a false TC diagnosis.
Fix: use explicit `tc_raw = ncu_summary.get(...); tc_raw is not None and tc_raw < 30.0`.
Pattern: never use `or default` for numeric signals — use explicit `is not None` guards.

**Bug: CSV quote-stripping in column name validation (ncu_analysis.py)**
`line.split(",")` on a wide-format NCU header left `"Kernel Name"` with embedded quotes.
`"kernel name" != "kernel name"` (after lower+strip, the quotes remain). Fix: `.strip('"')`
after each column name strip. Wide-format NCU CSVs always quote all column names.

**Bug: tensor_core_underutilization severity was "medium" (reconciliation.py)**
Changed to "high". When TC is measurably idle on a matrix operation, the potential speedup
is 5-10x on modern hardware with FP16/BF16 — that is clearly high severity.

## 5/21/26 -- Fixture-based accuracy eval

**Goal:** Prove the NCU diagnostic pipeline is correct without requiring a GPU or CUDA toolchain. CI-compatible.

**What was built:**
- 5 NCU CSV fixtures in `backend/tests/evals/fixtures/` (uncoalesced_dram_bound, tensor_core_idle, excessive_sync, register_pressure, well_optimized)
- `test_eval_accuracy.py` -- 15 pytest assertions covering true positives, false positive checks, and two cross-layer reconciliation tests
- `run_accuracy_report.py` -- human-readable table showing TP/FP per scenario

**Result: 6/6 expected bottlenecks detected, 0 false positives.** All 15 eval tests pass.

**Key design decisions:**
- Fixtures use real NCU metric names (e.g., `dram__throughput.avg.pct_of_peak_sustained_elapsed`) that go through the same `_canonical_ncu_metric_name()` normalization path as real GPU data. Using shorthand aliases like `dram_throughput_pct` also works (falls through the alias lookup to direct storage) but real metric names make fixtures more recognizable.
- Metric values were designed from bottleneck thresholds working backwards, not from real GPU measurements. The well_optimized true-negative required checking all 10 bottleneck conditions are unsatisfied simultaneously.
- The `warp_stall_sync` bottleneck requires dominant stall to be `"barrier"` or `"wait"`. Including both at 42%+18% ensures barrier is dominant and triggers detection.
- `tensor_core_underutilized` condition: `tc < 30 AND (occ is None or occ > 40)`. Including `sm__warps_active` at 62% for the TC-idle fixture keeps occupancy above 40% so the condition fires.
- Cross-layer field name: reconciliation output uses `"layers_confirming"` (list), not `"n_confirming"` (int). Use `len(diag["layers_confirming"]) >= 2` for multi-layer confidence checks.

**What this unlocks:**
- Can now claim "detects 4/4 known pathology patterns with 0 false positives on eval suite" -- a quantifiable accuracy claim backed by code, not assertion.
- Fixture library is the starting point for future real-world case studies: profile a real kernel, use observed metric values as a new fixture, verify detection.

## 5/21/26 -- Eval fixture bug-fixes (5-issue user feedback round)

**Context:** After the fixture-based accuracy eval was shipped, manual CLI testing revealed 5 issues. All were addressed in the same session.

**Bug: ISU hint threshold mismatch (Fix 1)**
The CLI's metric hint table said "low < 40%" but the bottleneck classifier threshold is 60%. Always derive the hint string from the same constant that drives the classifier — never hardcode the threshold twice.

**Bug: Wrong top recommendation for warp_stall_sync (Fix 2)**
`rule_ncu_warp_stall_sync` bundled `rec_ncu_shared_mem_layout` alongside `rec_ncu_reduce_syncthreads`. Since `rec_ncu_shared_mem_layout` had a higher composite score (better speedup estimate), it appeared first — giving "fix bank conflicts" as the primary advice for a synchronization bottleneck. Fix: remove `rec_ncu_shared_mem_layout` from this rule entirely. `suppressed_if` suppresses the whole rule, not individual recs — splitting was not needed here since removal was correct.

**Bug: register_pressure fixture didn't test what the name implied (Fix 3)**
Initial fixture lacked `launch__registers_per_thread` and `launch__block_size`, so `estimate_occupancy()` couldn't identify registers as the limiting factor and fell back to `unknown_threads_per_block`. This caused `occupancy_limited_by_block_size` (wrong) instead of `occupancy_limited_by_registers`. Fix: add both launch metrics to the fixture. Always include the driver metrics that cause the specific sub-variant you're testing.

**Bug: DATA NOTES warnings fire regardless of relevance (Fix 4)**
`_print_ncu_report_full()` printed all CSV parse warnings unconditionally. A well-optimized kernel with no bottlenecks still saw "Memory diagnosis: missing metric X" and "Tensor core diagnosis: missing metric Y". Fix: gate each warning prefix on whether its associated bottleneck category was actually detected. Irrelevant diagnostics erode trust faster than missing features.

**Bug: "Occupancy limited by" shown even for clean kernels (Fix 5)**
The occupancy limit causes block ran whenever `occupancy_limit_causes` was non-empty. The default cause (`unknown_threads_per_block`) fires when block size is unknown — which is always true for CSVs that don't include `launch__block_size`. Fix: filter `unknown_threads_per_block` from the display list AND gate the whole block on `occ < 50`. Showing a constraint that isn't constraining anything reads as a false alarm.

**Pattern:**
When adding a new sub-variant detection (e.g., `occupancy_limited_by_registers` vs `occupancy_limited_by_block_size`), write a fixture that includes the exact metric that drives the sub-variant — not just the output metric (occupancy %). The output metric alone can't distinguish causes.

## 5/23/26 -- frx bench implementation (wall-clock subprocess comparator)

**Goal:** `frx bench bad.cu good.cu` -- compile two .cu files, time wall-clock with warmup, optionally profile with NCU, report speedup + bottleneck diff. No Roofline/MFU in v0.

**Timing design:** Wrap `subprocess.run()` with `time.perf_counter()` -- correct when the binary calls `cudaDeviceSynchronize()` before exit. Warmup runs are discarded to eliminate driver initialization noise. This avoids CUDA event injection (would require modifying the source files) and avoids shell timing (shell startup cost is negligible relative to GPU kernel time, but less precise).

**NCU stdout filtering:** NCU 2026.x output on Windows includes "done" lines and app stdout mixed into stdout. Filter: keep only lines starting with `"` (quoted CSV). This reliably isolates the CSV data from any other output, including leakage from the profiled application itself.

**arch requirement:** Without `-arch=sm_xxx`, nvcc produces a JIT-compiled binary. NCU cannot profile JIT-compiled binaries. Added a warning when `--with-ncu` is used without `--arch` so users get a clear message rather than a silent None result.

**Mocking pattern for CLI tests:** `bench_compare` is patched at `fournex.bench.bench_compare` (module-level import path). The CLI checks file existence before calling bench_compare, so tests create real temporary `.cu` files with `tempfile.NamedTemporaryFile(suffix=".cu", delete=False)` -- empty files pass the existence check.

**31 tests, 0 failures.** All timing tests use `side_effect` lists on `time.perf_counter` mock rather than sleeping -- deterministic and fast.

## 5/23/26 -- frx explain enrichment (EXPECTED IMPROVEMENT + validation targets)

**Goal:** The LLM prompt should tell the user's LLM not just "what's wrong" but "what success looks like" -- estimated speedup, current measured values, exact NCU metrics to re-run after the fix.

**Gap discovered:** `estimated_speedup_pct_min/max` existed in `catalog.yaml` and were consumed internally by `_speedup_score()` but never forwarded to the output dict. Same for `why`, `actions`, `validation_steps`. The engine had all the data; `build_explain_result()` was only extracting 5 of 21 fields.

**Fix:** Thread `estimated_speedup_pct_min/max` through `engine.py` rec dict, then expand `top_recommendations` in `explain.py` from 5 to 10 fields. `validation_steps` already carry `current_value` (computed from signals at engine time) -- no additional computation needed.

**EXPECTED IMPROVEMENT section design:**
- Direction arrows: `<--` for decrease-expected, `-->` for increase-expected, ` ~~` for stable. ASCII-only -- no Unicode em-dashes or arrows (Windows cp1252 console breakage).
- "was X" prefix on validation steps only when `current_value is not None`. Not all validation steps have a measured baseline.
- Re-profile command built from `validation_steps[*]["metric"]` joined with commas. Gives the user a copy-paste command that measures exactly the metrics that matter.
- Only render the section when `top_recommendations` is non-empty -- avoids an empty heading for well-optimized kernels.

**Summary enrichment:** WHAT TO FIX FIRST lines append `(est. X-Y% speedup)` only when both min and max are non-None. If catalog entry lacks speedup estimates (some recs don't have them), the line renders cleanly without the parenthetical.

**6 new tests added** (41 total for test_explain.py). Tests use `_EXCESSIVE_SYNC_CSV` for speedup/validation coverage because `rec_ncu_reduce_syncthreads` has catalog speedup estimates; `_UNCOALESCED_CSV` for structural field presence checks (value can be None).

## 5/27/26 — Roofline/MFU, kernel attribution, TC efficiency, occupancy analysis

### Roofline / MFU (`roofline.py`)
- All roofline values are estimated from utilization percentages (`avg_dram_throughput_pct`, `avg_tensor_core_utilization_pct`, `avg_issue_slot_utilization_pct`) because raw instruction counts and byte transfer totals are not available in the standard NCU export that Fournex currently processes. Always set `"estimated": True` in the returned dict.
- When TC is active (tc_pct > 5.0), use `achieved_tflops = tc_pct/100 × peak_fp16_tflops`; otherwise use `isu_pct/100 × peak_fp32_tflops`. The TC path intentionally uses the FP16/BF16 ceiling — tensor core throughput is measured relative to that ceiling, not FP32.
- `mfu_pct` is always computed against `peak_fp32_tflops` so it's consistent across kernels regardless of which path they use. TC kernels can legitimately exceed 100% MFU — clamp to 100 only when computing `mfu_gap` for opportunity scoring.
- Returns `None` (not raises) when arch_profile lacks peak specs. Downstream code must guard with `if roofline is not None:`.

### Arch profiles — peak hardware specs
- Added `peak_fp32_tflops`, `peak_fp16_tflops`, `peak_memory_bw_gbps` to all SM profiles. Roofline depends on these. Without them, `compute_roofline()` returns `None` (no crash, just no analysis).
- Values are GPU-family representative (not per-SKU). Users can override via YAML arch profile overrides.

### Kernel attribution (`kernel_attribution.py`)
- `compute_kernel_attribution(summaries, arch_profile, environment=None)` — `environment` param needed so per-kernel `analyze_tc_efficiency()` can check `mixed_precision` env flag.
- Per-kernel entries embed `tc_analysis` and `occupancy_analysis` sub-dicts. This co-locates all per-kernel diagnostics in one place, making `result["kernel_attribution"]["kernels"][i]` the single source of truth for any individual kernel's full diagnostic picture.
- `_build_ncu_result()` derives `tc_summary` and `occupancy_summary` from the per-kernel data already computed in `kernel_attribution["kernels"]` — no re-iteration over `summaries` needed. This avoids redundant computation and keeps the assembly point clean.

### TC analysis (`tc_analysis.py`)
- `tc_eligible` uses an OR condition: `isu_pct > 20` OR `tc_pct > 1.0`. The tc_pct proxy handles the case where a kernel has active (but low) TC use yet low ISU — it's still TC-eligible by definition of already using it.
- `fallback_to_cuda_cores` = `tc_eligible AND not tc_active AND isu > 20`. The ISU > 20 check re-confirms that compute work is happening — without it, a fully memory-bound kernel with low ISU would falsely appear to be "falling back."
- `mixed_precision_opportunity` gates on `bf16_supported OR fp8_supported` — if the arch only supports FP16 (neither bf16 nor fp8 in profile), don't advertise the opportunity. Check `arch_profile.get("bf16_supported")` and `arch_profile.get("fp8_supported")` explicitly; both default False.
- `efficiency_label` falls through: `>50% → efficient`, `active ≤ 50% → underutilized`, `eligible not active → inactive`, else `no_data`. Not eligible + not active always falls to `no_data` (no TC → nothing to label).

### Occupancy analysis (`occupancy_analysis.py`)
- `analyze_occupancy()` reads from `summary.occupancy_estimate` which is computed at parse time by `estimate_occupancy()`. No re-computation at analysis time. The estimate dict has `occupancy_pct`, `limiting_factors`, `blocks_per_sm_limits`.
- Diagnosis includes quantitative detail only when the relevant field is present: register count for register-limiter, shared memory KB for shared-memory-limiter. Don't print "Current: None registers/thread."
- `_LOW_OCCUPANCY_PCT = 40.0` and `_EFFICIENCY_GOOD_PCT = 80.0` are module-level thresholds — same values as used in the legacy `classify_ncu_bottlenecks()` `occupancy_limited` threshold. Keep them consistent.

### Testing pattern for analysis sub-modules
- Each module gets its own test file with: (1) schema checks (all required keys present), (2) math unit tests (gap = theoretical - achieved, efficiency = achieved/theoretical × 100), (3) edge cases (None inputs, zero denominators), (4) diagnosis text spot checks (does register limiter mention "register"?), (5) summarize() rollup checks, (6) integration tests that verify the sub-dict appears inside `kernel_attribution["kernels"]` and the summary appears in the NCU result dict. This 6-layer test structure covers both units and integration in one file.
- 81 new tests total (41 TC + 40 occupancy). 659 total passing after wiring.


## 5/29/26 — Framework Abstraction Tax (meta-classifier over existing telemetry)

# Framework Abstraction Tax Notes
- `compute_framework_abstraction_tax(run_summary, bottlenecks)` in `framework_abstraction_tax.py` — a 0-100 score for how much GPU inefficiency is framework/runtime overhead vs hardware/data-pipeline. It is an *aggregation + naming* layer; it adds NO new collectors. Every input already existed in the telemetry-path `run_summary` (the unnamed `launch_bound` classifier at `analysis.py:270-299` was the partial precursor).
- Core formula: `overhead_idle = max(0, (1 - gpu_active) - input_frac - h2d_frac - sync_frac)`, scaled by `fragmentation_weight = 0.5 + 0.5*frag_signal`. Subtracting the data-pipeline idle (input/copy/sync, read from bottleneck evidence exactly as `signals.py:27-29` does) is what prevents input-bound workloads from being mislabeled as framework tax — verified by a test where gpu_util=40 + dataloader=0.5 scores <20.
- **Placement matters**: this is a *telemetry-path* score, NOT NCU-path. The signals (`kernel_count_per_step`, `small_kernel_fraction`, `average_gpu_utilization_pct`, `shape_volatility_ratio`) live in `analysis.py` `run_summary`; NCU CSV summaries don't carry per-step launch counts or GPU-active %. Wired into `summarize_step_scope` (the single assembler both `run` and `steady_state` scopes flow through) so one insertion point covers both scopes + JSON.
- Returns `None` when `profiler_windows_exported == 0` (NCU-only path) so the key is omitted, not a meaningless zero — same None-guard discipline as `compute_roofline`.
- **Contributors are gated on `score >= moderate (20)`, not on raw `overhead_idle`.** First attempt gated on `overhead_idle < 0.05`, which left fragmentation contributors attached to a low (9/100) input-bound score — misleading. Gating on the score itself ties the explanation to the headline number.
- Two deliberate scope cuts (decided with user, both the conservative option): (1) **infer-only** for CUDA-graph/torch.compile — we do NOT detect their state (no collector exists), only infer "would help" from stable-shapes + heavy-launch, flagged `inferred=True`; (2) **no speedup multiplier** — the memo's "+1.4-2.1x" is uncalibrated, so V1 ships score + contributors only. Respects the score-calibration rule.
- Severity bands (<20 low / 20-45 moderate / >=45 high) have no `min_score` consumers yet, so no dead-threshold risk — but kept above the noise floor per the calibration habit.
- Integration sanity check: the `launch_bound_tiny_kernel` golden fixture scores 74/100 (high), fragmentation leading, graph-capture + fusion inferred. 7 new tests, full suite still 713 passing (13 pre-existing `test_real_cuda_kernels.py` failures are nvcc-toolkit-absent, unrelated).


## 5/31/26 — frx explain brief enrichment (close analysis/brief gap)

# explain.py Enrichment Notes
- `build_explain_result()` now extracts three new fields from `ncu_result` and includes them in the return dict: `top_kernels` (from `kernel_attribution["top_opportunities"][:5]`), `roofline` (from `ncu_run_summary["roofline"]`), `occupancy_summary` (from `ncu_result["occupancy_summary"]`). These were already computed by the NCU analysis pipeline but were never passed to the LLM.
- `render_llm_prompt_txt()` gains four new sections, all gated:
  1. **SECONDARY ISSUES** — only when `len(diagnoses) > 1`
  2. **TOP KERNELS TO OPTIMIZE** — only when `len(top_kernels) > 1`; single-kernel workloads omit it (nothing to rank)
  3. **Roofline region + MFU line** — appended to the NCU evidence block when `roofline` is present
  4. **Occupancy limiter line** — appended only when occupancy is in primary/secondary bottleneck AND `occupancy_summary` has a `dominant_limiter`
- `render_summary_txt()` already had SECONDARY ISSUES and MISSING DATA — those were fine. The gap was entirely in the LLM prompt, not the summary.
- Gating on `score >= _MODERATE_THRESHOLD` pattern from framework_abstraction_tax is analogous here: surface richer context only when it's actionable. TOP KERNELS at len > 1 follows the same logic — there's nothing to rank when there's one kernel.
- 12 new tests in `test_explain.py` (50 total). Full suite 724 passing.
- Remaining gap (explicitly out of scope): training telemetry → explain path. There's no `frx explain runs/<id>` today; the explain command only accepts NCU CSVs. That's the next meaningful feature if the user base skews toward PyTorch training engineers.


## 6/1/26 — frx explain training telemetry path

# frx explain Run-Directory Path
- `frx explain <dir>` auto-detects directories (and .zip) and routes to the training path; `.csv` continues to use the existing NCU path. Detection is in `explain_cmd()` (cli.py) — purely `input_path.is_dir()` check, no fragile magic.
- `_explain_training()` helper loads summary via `_load_or_generate_summary()` (already handles trace.jsonl / derived/summary.json / profiler bundle), selects scope with `_select_scope_data()` (prefers steady_state over run), then calls the three new explain functions.
- `build_telemetry_explain_result(scope_data)` produces schema `frx_telemetry_explain_v0`. It extracts from `diagnosis`, `run_summary`, `bottlenecks`, and `framework_abstraction_tax`. Phase fractions (input/h2d/sync idle) are pulled from bottleneck evidence, not recomputed. Secondary bottlenecks are filtered against both `user_facing_bottleneck` AND `primary_bottleneck` — golden fixture revealed they can differ, causing duplicate in secondary list.
- `render_training_llm_prompt_txt()` uses bottleneck-specific question templates that interpolate real metric values (gpu %, kernel count, median us, etc.) via `_format_training_question()`. Templates use `.format()` with a try/except so a missing metric value silently falls back to the default question.
- ASCII-only in all render functions: replaced `μs` with `us`, `—` with `-`, `•` with `-`. Windows console (cp1252) cannot encode these and would break piped usage.
- `_select_scope_data()` is a public function (exported via `from .explain import`) because `explain_cmd` needs it and tests need to exercise it directly.
- 29 new tests in `test_explain_training.py`; 753 total passing. Backward compat confirmed: all existing NCU explain + CLI tests still pass (76 tests).
- `--scope` arg added to the `explain` subparser (for run-dir path; ignored for CSV path).
- Remaining for Phase 2: `--prompt-only` for training path should interpolate run_id into the re-run command shown at the bottom.


## 6/2/26 — Developer simplicity: GPU auto-detect + profile --explain

# detect_gpu_model
- `detect_gpu_model(gpu_name: str | None) -> str | None` in `arch_profiles.py`. Walks `_PRODUCT_TO_SM` checking whether any key is a **substring** of the normalized device name (lowercase, strip "nvidia ", strip spaces/dashes). Longest match wins — this is what makes "rtx3090ti" win over "rtx3090" for an RTX 3090 Ti.
- Wired into `_detect_environment()` in cli.py: if `gpu_name` is available and `gpu_model` is not already set, auto-populates `env["gpu_model"]` + `env["gpu_type"]`. Explicit `--gpu-model` flag always takes precedence (set after `_detect_environment()` in `_environment_from_args`).
- RTX 5060/5060Ti/5070/5070Ti were missing from `_PRODUCT_TO_SM` — added as sm_120 alongside 5080/5090.
- 10 new tests in `test_arch_profiles.py`. The "longest match wins" invariant is explicitly tested.

# frx profile --explain
- Added `--explain` flag (+ optional `--explain-out DIR`) to `frx profile`. After `_print_ncu_report_full()`, if `--explain` is set: calls `build_explain_result(ncu_result=result)` (no re-analysis, same result already in memory) and writes 3 files. Prints footer path + paste instruction.
- Same flag added to `frx collect` — calls `_explain_training(args, run_dir)` after `_print_collection_summary()`.
- The key insight: `build_explain_result()` takes `ncu_result` directly, so `--explain` costs only rendering time (no second NCU pass).
- Before this: CUDA workflow was 2 commands; training was 3. After: both are 1.
- 763 total tests passing after all changes.

# 2026-06-10: Technical robustness fixes (WS1/WS2/WS3)

## Audit corrections (non-obvious mismatches between the audit report and the code)

- `classifier_version` was ALREADY emitted by `analysis.py:build_diagnosis_result` (both branches) and pinned by `test_analysis_bottlenecks.py`. The NCU result (`_build_ncu_result`) had no version field — that was the actual gap. Do not bump `CLASSIFIER_VERSION` in behavior-preserving refactors or the golden test assertion breaks.
- `storage.py` is write-only. `RunRecord.from_dict` is reached only via `common_ir.validate_run_dict` / `common_ir_validators.validate_run_payload`. The CLI never reads artifacts back through `RunRecord.from_dict`; it reads raw SDK event JSONL and summary JSON directly. Schema gate in `from_dict` still protects the validator paths.
- cli.py:2496/2507 are corrupt-summary-cache fallbacks (`_load_or_generate_summary`), NOT NCU subprocess failures. The real subprocess excepts are: cli.py `_compile_ptx_cu` and `_run_ncu_compare`. The audit misattributed them. `profile()` already hard-errors on ncu failure — leave it.
- `sdk.SCHEMA_VERSION="0.1.0"` versions the telemetry JSONL stream (distinct artifact family from common_ir). Don't unify; they evolve independently.

## WS3 (schema versioning): key decisions

- Missing `schema_version` is treated as current, preserving back-compat with every existing artifact (they all lack it).
- Future-version artifacts raise `SchemaVersionError` naming both the artifact version and the supported max.
- The migration registry is empty today; any older-versioned artifact therefore raises "no migration path" rather than silently accepting stale data. This is the correct conservative behavior.
- `SchemaVersionError` extends `ValueError` so existing `except ValueError` guards still catch it.

## WS1 (threshold centralization): key decisions

- All 22 classification trigger cutoffs moved to `ClassifierThresholds` dataclass in `thresholds.py`. Score-formula constants (like `occ/40.0` inside the occupancy score) use the same field to avoid divergence.
- `resolve_thresholds()` mirrors `get_arch_profile()` from `arch_profiles.py` — reuses `_select_profile_override`/`_normalize_gpu_key`. Threshold overrides go under `classifier_thresholds:` key inside per-SM-arch profile blocks in the existing YAML override mechanism.
- Unknown override key raises `ValueError` immediately (typos must not silently no-op — this is the key inspectability requirement).
- Provenance stamped as `thresholds_source` + `thresholds_hash` in both the diagnosis dict (analysis path) and the `classifier` block (NCU path). `classifier_version` remains untouched.

## WS2 (evidence surfacing): key decisions

- `StepResult(ok, output, error, step)` dataclass is the seam. Both `_compile_ptx_cu` and `_run_ncu_compare` now return `StepResult`(s) instead of `str | None`.
- `evidence_failures` is a new additive list in the JSON output alongside `comparison` and `reconciliation`. It is NOT injected into reconciliation's `missing_evidence` (that schema has ~10 tests pinning it).
- The `evidence layer unavailable: LAYER (target) — reason` line printed right after the header (before "Winner") is user-visible immediately.
- Exit code stays 0 for degraded analysis — the comparison still runs on whatever evidence is available.
- `_load_or_generate_summary` broad `except Exception: pass` narrowed to `(json.JSONDecodeError, OSError)` with a `[warn]` stderr line before falling through.
- 35 new tests added (14 WS3, 12 WS1, 9 WS2). Total suite: 825 passing.

# 2026-06-15: Telemetry classifier environment propagation

- Architecture-aware threshold provenance is only trustworthy if every public summary entry point carries `environment` through to `summarize_step_scope()`. `summarize_run_with_steady_state()` must pass the same environment to both `run` and `steady_state`; otherwise the two scopes silently fall back to default thresholds even when lower-level helpers are correct.
- CLI-generated `derived/summary.json` must use the environment already written to `run_config.yaml`. The live collect path can pass the in-memory config, but regeneration from existing bundles needs to reread the generated run config or `thresholds_source` / `thresholds_hash` become misleading.
- Do not bump `CLASSIFIER_VERSION` for plumbing-only fixes. The classifier ruleset did not change; the fix makes existing threshold resolution reachable from the telemetry paths that previously skipped it.

# 2026-06-14: Production-readiness audit + version/logging hardening

- Local test "failures" were entirely environmental, not defects: 812 passed. The 13 `test_real_cuda_kernels` fails are `nvcc` present-on-PATH but no MSVC host compiler (Windows); the `@skipif(not NVCC)` guards only check `shutil.which("nvcc")`, so a broken-but-present nvcc still attempts compilation and hard-fails. The 29 errors are a locked `%TEMP%\pytest-of-jorge` dir plus running under Python 3.14 (outside the declared 3.10–3.12 matrix). All skip cleanly on Linux CI.
- IP/release finding (documented, deferred by user): the wheel ships proprietary modules. `pyproject.toml` uses `packages.find where=["."]` with no exclude, so every build bundles `autopilot/`, `recommendations/engine.py`, and `signals.py` — confirmed inside `dist/fournex-0.1.2-py3-none-any.whl`. V0_RELEASE_CHECKLIST marks these private. The 0.2.x PyPI wheels almost certainly leak them. Decision pending.
- Version single-sourcing: replaced the hardcoded `__version__` literal (drifted to 0.2.7 vs pyproject 0.2.8) with `importlib.metadata.version("fournex")` + `PackageNotFoundError` fallback. Caveat: a stale non-editable copy in site-packages was shadowing the editable source, so `import fournex` from the repo root resolved the wrong package — verify from `backend/python/` when sanity-checking.
- The "6 non-CLI prints" were a false signal: 1 is inside a docstring usage example (`kernel_attribution.py`), the other 5 are `verbose`-gated relays of child-process stdout/stderr in the proprietary autopilot module, where `print` is correct (forwarding subprocess output, not logging). Left them; the real gap was zero logging infrastructure.
- Added a `fournex` logger configured by top-level `-v/--verbose`/`--debug` in `cli.py:_configure_logging`, logging to stderr (never stdout, which carries pipeable JSON/briefs). `--debug` outranks `-v`; handler added once (idempotent). Wired `logging.getLogger(__name__)` + debug calls into the two central library entry points (`analysis.summarize_run`, `ncu_analysis.analyze_ncu_csv`).

# 2026-06-14: Case-study harness (credibility loop)

- The harness deliberately runs on **static source analysis**, not synthetic NCU CSVs. The cuda_zoo bad/good pairs produce genuine, reproducible finding diffs with zero GPU (e.g. 01 resolves `uncoalesced_access`, 02 resolves `fp32_only_matmul`+`no_shared_memory_tiling`). This avoids the credibility trap of hand-tuning fixture data to pass our own validator — the proof is real analysis of real source. NCU is layered on only when `before_ncu`/`after_ncu` are supplied.
- Reused the whole existing stack rather than re-implementing: `compare_implementations` (multi-layer diff with `static_diff.findings_diff.resolved_in_b/new_in_b`), `inspect_cuda_source`, and the `explain` renderers (`render_summary_txt`/`render_llm_prompt_txt`/`render_evidence_json`). The harness is orchestration + validation + artifacts only.
- Manifests were authored from *actual* analyzer output, not guesses — ran `inspect_cuda_source` on each pair first to capture real finding codes. Important nuance: persistent shared findings (e.g. `missing_vectorized_loads` in 01, `missing_obvious_bounds_guard` in 04) are NOT regressions because they're in `shared`, not `new_in_b`, so `no_new_findings` still holds.
- `build_explain_result` only fills `top_recommendations` from the NCU path; for static-only it's empty. The actionable fix text lives in each static finding's `message` field (the `recommendation`/`fix`/`title` fields are None). The transcript's "Recommended fix" falls back to resolved-finding messages.
- Validation = exit code: `frx case-study run` returns non-zero on validation failure, so it doubles as a CI regression gate for classifier drift.
- Test temp-dir caveat persists: on this box `tmp_path` fails with PermissionError on `%TEMP%\pytest-of-jorge`; run `pytest --basetemp=.cs_tmp` (repo-local) to exercise artifact-emission tests. Full suite then: 860 passed, 13 failed (all broken-nvcc), 3 skipped.

# 2026-06-16: Telemetry reliability hardening

- Architecture confirmed: telemetry is dual-backend behind one API (`sdk.emit_event`). Python path (default, `HAS_NATIVE=False`) = dict events in `_local_events` + nvidia-smi thread + torch.cuda.Event. Native path (opt-in `BUILD_NATIVE=1`) = C++ CUPTI/NVML engine in `native/`. The "serious" engine ships disabled, so most users get the coarse Python path — frame marketing accordingly.
- Streaming durability fix: chose to stream raw events to disk during the run as a *durability layer only*, keeping `_local_events` for the end-of-run summary (the classifier needs all events; true memory-bounding needs a streaming summary = bigger change, deferred). Key trick to avoid a two-writer conflict: `_auto_persist_artifacts` calls `_close_trace_stream()` BEFORE `persist_run_artifacts()` rewrites the canonical file "w" from memory. On crash, persist never runs but the streamed file survives. Streaming gated on FRX_AUTO_PERSIST/FRX_STREAM_TRACE so bare library use doesn't create surprise files.
- Native-divergence guard: centralized event resolution in `storage._resolve_events`; raises RuntimeError under `HAS_NATIVE` when no explicit events passed, instead of silently writing empty artifacts. `_auto_persist_artifacts` short-circuits to `native.flush()/shutdown()` under native. This is a defensive guard, NOT the full TelemetryBackend interface refactor (that's the proper long-term fix, deferred).
- Sampler resilience: `_sample_gpu_metrics` `except` now continues instead of `return`; caps warnings (1st + Nth), backs off to >=5s after 5 consecutive failures. One transient nvidia-smi hiccup previously killed sampling for the whole run.
- `timestamp_ns` is `perf_counter_ns()` (monotonic, process-relative) — fine for single-process durations, not cross-process/CUPTI-correlatable. Left as-is; flagged clock-domain metadata as a P2 for when native/distributed correlation matters.
- 6 new tests (test_telemetry_reliability.py). Full suite: 866 passed, 13 failed (broken nvcc), 3 skipped. Run with --basetemp=.cs_tmp to dodge the locked %TEMP% on this box.

# 2026-06-17: `frx eval sakana` — validating the analyzer on the AI-CUDA-Engineer Archive

- The decisive design constraint came from *inspecting the live dataset*, not the dataset card. `NCU_Profile` cells are a **fixed section set** (Speed-of-Light + Occupancy + a few instruction/divergence stats) and are **null on ~35% of rows**; incorrect kernels split into compile/runtime failures (Error text, no profile) and **silent numerical mismatches** (Error/Clang/NCU all null, large Max_Diff). The profiles consistently LACK warp-stall breakdown, sectors-per-request, and tensor-pipe metrics — so `warp_stall_*`, `uncoalesced_access` (from NCU), and `tensor_core_underutilized` are simply not weak-labelable here. Treated this as a feature: absent sections flow into Fournex's existing missing-evidence/confidence-downgrade machinery.
- Cells are **Python-repr strings** (single quotes, `True/False/None`, apostrophes inside rule descriptions), NOT JSON. Parse with `ast.literal_eval`; naive `'`→`"` replacement corrupts contractions (`doesn't` → `doesn"t`) and fails. Reused the same parser for `Clang_Tidy` (same format).
- The trap that justified a dedicated adapter (`sakana_ncu_adapter.py`): the generic header map in `kernel_inspector` aliases `"Memory Throughput" → dram_throughput_pct`, but in this dataset that metric is **byte/second**, not %-of-peak. Routing it through makes every kernel look memory-bound. The adapter only fills a `*_pct` field from a metric whose unit is literally `%`, and uses `Max Bandwidth` (the real %-of-peak DRAM figure) for `dram_throughput_pct`. Pinned this with a test.
- Surprise (good): even without warp-stall data, the **reconciliation + roofline** layer still surfaces `memory_bandwidth_saturation` / `roofline_memory_bound` at **medium** confidence from the `Max Bandwidth`-derived DRAM%, so Fournex does explain the slow memory-bound matmuls — and never exceeds medium-high on this dataset (no stall sampling). The NCU *classifier* label `memory_bandwidth_bound` correctly stays silent because it gates on `memory_stall_fraction` which is structurally 0 here. Don't relax that gate to chase the label; the roofline path already covers it honestly.
- Correctness is genuinely partly out of reach: a profiler/static analyzer cannot detect silent numerical mismatches (a wrong kernel profiles fine). The harness makes this explicit — `correctness.status` is always `not_verified_by_fournex`; `warning` only fires on real signals (Error text, clang `: error:`, `conditional_syncthreads`). The leaderboard buckets recall into build/runtime-error (100%) vs silent-mismatch (0%, labeled a documented blind spot). Resisted the urge to treat `missing_obvious_bounds_guard` as a correctness flag — it fires on correct-by-construction conv2d indexing and tanked precision (warnings on correct kernels jumped from 6% to ~25%). Kept the risk set to just `conditional_syncthreads`.
- Gold-set honesty: the dataset has no bottleneck labels, so "accuracy" only exists vs a 52-row hand-labeled `gold.yaml`. The strongest, non-circular assertion is the **confidence ceiling** (`<= medium-high` on every profiled row — a structural truth from the missing stall data) and **correctness-warning** (derived from the ground-truth `Error`/`Correct` flags, not from Fournex output). `expected_primary` is the circular-risk one; set it only where the SoL reading is unambiguous (memory-bound matmuls, register-bound MLPs, memory-bound elementwise ReLU even when >1x torch) and left null elsewhere. Every leaderboard metric carries an explicit `basis: objective | heuristic-circular | vs-truth` tag so nothing reads as truth that isn't.
- Offline/CI-first: dependency-free downloader (`scripts/fetch_sakana.py`) pages the HF datasets-server `/rows` REST API (no `datasets`/pandas), skips truncated cells, and balances a correctness mix (early rows are all correct, so incorrect kernels must be scanned for deeper). Cached 102-row subset + gold live under `fournex/data/sakana/`, registered as package-data (needs `data/__init__.py` so setuptools treats it as a package). Reused `analyze_ncu_profile_dict` → `_build_ncu_result`, `inspect_cuda_source`, `build_explain_result`, `render_summary_txt` — the harness is adapter + orchestration + scoring only.
- 14 new tests (test_eval_sakana.py), all offline. Same env caveats as prior sessions: full suite shows 13 `test_real_cuda_kernels` fails (no nvcc host compiler) + tmp-path PermissionErrors under `%TEMP%`; everything else passes (844 in the clean subset).

# 2026-06-17: Live GPU dogfood on WSL2 + source-built PyTorch (RTX 5060, sm_120)

- **Drive method that works:** the whole live path is testable from the dev box by shelling into WSL — `wsl.exe bash -lc 'micromamba run -n torch-dev <frx ...>'`. The user's torch is a source build (`2.12.0a0+git…`) in micromamba env `torch-dev` (Py 3.11); `frx` installed editable from `/mnt/c/...` into that env. Caveat on nested quoting: `wsl.exe bash -lc '...'` mangles `$VAR` and `$(...)` and embedded quotes — write a `/tmp/*.py` (or `.sh`) via a quoted heredoc and run *that* instead of inlining complex one-liners.
- **WSL nvidia-smi sampling is solid:** `frx collect` sampled real GPU utilization in WSL2 (captured a sustained **99%** on the compute-bound run, ~1–5% on the starved run). nvidia-smi lives at `/usr/lib/wsl/lib/nvidia-smi`. `frx doctor` + `frx smoke-test` both pass cleanly in WSL.
- **Bundled examples are useless for a real GPU test** (`backend/examples/train_*.py` use FakeTensor mocks → GPU idle). Wrote two real instrumented workloads with `.cuda()` tensors: `train_gpu_bound.py` (3×4096 MLP fwd/bwd/opt, no loader) and `train_input_bound.py` (slow `Dataset` + `num_workers=0`, wrapped in `instrument_dataloader`). These are the right positive controls.
- **Classifier verdicts on REAL telemetry were correct:** input-bound run → `input_bound`, confidence **high (1.00)**, evidence "DataLoader wait fraction 0.991", recs = num_workers / move-transforms / prefetch (spot on). gpu-bound run → "No bottleneck above threshold" (correct: no false alarm). Throughput: gpu-bound 9.65 steps/s (heavy 4096³), input-bound 121.5 steps/s but GPU-starved. The trace-derived path (dataloader/phase spans → input/copy/sync + dominant stall) is genuinely working end-to-end.
- **BUG found (real, with proven fix): sampled GPU utilization is dropped on the instrumented `collect` path.** `frx collect` writes nvidia-smi samples to `gpu_metrics.csv`, but `_generate_derived_summary_from_trace` (cli.py:2862) builds its event list *only* from `raw/trace.jsonl` and never folds in the CSV. The converter `_gpu_metrics_csv_to_sdk_events` (cli.py:3125) exists but is wired ONLY into the profiler-bundle path (`_events_from_profiler_bundle`, cli.py:2941). Net effect: every SDK-instrumented `collect` run reports `average_gpu_utilization_pct == 0.0` (and `frx analyze`/`explain` print "GPU Utilization 0.0%") even when the GPU was pegged — and the util-keyed rules (`underutilized_gpu`, `launch_bound`) can never fire from collected data. Proven non-mutating repro: `summarize_run(trace)` → 0.0%; `summarize_run(trace + _gpu_metrics_csv_to_sdk_events(csv))` → **71.1%**. Fix is a one-liner: in `_generate_derived_summary_from_trace`, after `_read_jsonl_events`, also `events.extend(_gpu_metrics_csv_to_sdk_events(run_dir / "gpu_metrics.csv"))` (sort by timestamp if the classifier cares about ordering). Note the dead-threshold sibling risk (see [[feedback_score_calibration]]): once util is populated, re-check that the input_bound case isn't now also flagged underutilized_gpu as a noisy secondary.
- **NCU deferred this pass** (telemetry-first, per plan). Toolchain IS present in WSL for a later pass: CUDA 12.3/13.2, `nvcc`, `ncu` (Nsight Compute 2026.1.0). Expect sm_120 (Blackwell) to lack PC-sampling stall metrics and WSL to need the perf-counter permission dance — its own focused session.

## Update (same day): the GPU-util fix was deeper than one line

The "just extend events with the CSV" hypothesis was necessary but NOT sufficient. Adding the
fold-in to `_generate_derived_summary_from_trace` passed unit tests but the live `collect` path
STILL showed 0% util. Root cause #2: under `collect` the workload **subprocess's SDK auto-persist
writes `derived/summary.json` first** (from its own events — no GPU samples, since nvidia-smi
sampling runs in the *parent*), then the parent's `_generate_derived_summary_from_trace` hit
`if derived_path.exists(): return` and skipped regeneration. Only the parent has `gpu_metrics.csv`,
so the parent must own the derived summary. Fix: added `overwrite: bool=False` to the function and
`collect` now calls it with `overwrite=True` (still preserves an existing summary when the trace is
empty, so it never clobbers with nothing). Diagnosing this needed forcing regen on a real run dir
(`rm derived/summary.json` then re-run the function → util jumped 0.0 → 76%). Lesson: when a
collected metric is "silently disconnected", check *who writes the artifact first* — a subprocess
auto-persist racing the parent is easy to miss. 3 regression tests in test_cli_collector.py
(absent-csv → 0%, present-csv → >0%, **stale-subprocess-summary + overwrite** → >0%). Verified
end-to-end via `collect`: gpu-bound now 79.4% util (was 0.0), no false bottleneck; input-bound
genuinely 0% (GPU idle between 1s samples) so `underutilized_gpu`'s `util>0` guard keeps it from
firing as a noisy secondary — primary stays cleanly `input_bound`. Full suite unchanged otherwise
(only pre-existing nvcc + tmp_path env failures remain).

# 2026-06-17: Live NCU/bench/explain pass on Blackwell (RTX 5060 / sm_120) under WSL

- **NCU works on sm_120 under WSL** — no `ERR_NVGPUCTRPERM`. A trivial vecadd profiled cleanly with `ncu --set basic` and returned real %-of-peak (DRAM 90.6% vs Compute 23.3%). The perf-counter permission wall we feared isn't present on driver 591.86 + Nsight 2026.1.0. This de-risks the whole NCU half.
- **Blackwell metric-gating bug (found + FIXED):** every `frx` NCU preset — including `memory` — bundles `smsp__pcsamplingdata_*` PC-sampling stall metrics, which Blackwell removed. ncu fails the ENTIRE pass (`exited with code 9`) if any requested metric is invalid, so even the valid DRAM/cache metrics were lost. Fix in `ncu_presets.py`: `filter_metrics_for_sm(metrics, sm_version)` drops `smsp__pcsamplingdata_*` when sm >= 100 (Blackwell DC + consumer); `build_ncu_command(..., sm_version=)` applies it. Wired sm_version through the three call sites: `profile` (resolves from env gpu_model, prints a "dropped on Blackwell" note), `bench.profile_with_ncu` (uses the `--arch` string directly as the sm hint), and `compare`'s `_run_ncu_on` (from `args.gpu_model`). Gate at sm>=100 not the literal ">= sm_120" — PC sampling is gone for the whole Blackwell generation; Hopper sm_90 still has it. Unknown arch (None) keeps all metrics. 6 tests in test_ncu_presets.py.
- **End-to-end results after the fix (all 5 goals met):**
  - `frx bench bad.cu good.cu --with-ncu --arch sm_120` on cuda_zoo/01_uncoalesced: compiles both for sm_120, and the NCU diff correctly reports **RESOLVED: uncoalesced_access** (verdict "improved, 1 resolved, 0 new"). Caveat: wall-clock speedup was ~1.0x because the whole-binary timing is dominated by ~160ms CUDA context init — bench's micro-kernel timing is not meaningful at this scale; the NCU bottleneck diff is the trustworthy signal. (Possible future: time the kernel region via cudaEvent, not the process.)
  - `frx profile --preset memory -- ./vecadd`: live capture works, prints the Blackwell drop note, classifies from real metrics (DRAM 87.8%, L1 0%, L2 0.1%).
  - `frx explain <csv> --src vecadd.cu`: **merges source + NCU layers** ("Layers: source, ncu"), primary = memory_bandwidth_saturation at **medium** confidence (correctly not high — PTX absent + stall data dropped), evidence drawn from both layers, fixes with speedup estimates, honest MISSING DATA section. This is the "second half" proof: live PyTorch telemetry AND live CUDA kernel profiling both work on real Blackwell hardware.
- **Driving NCU from WSL via this session:** `ncu`/`nvcc` live in `/usr/local/cuda/bin` (not on micromamba-run PATH); `micromamba` is a shell function (not a binary). Reliable recipe: write the script with the Write tool to a `/mnt/c/...AppData/Local/Temp/*.sh`, then `MSYS_NO_PATHCONV=1 wsl.exe bash /mnt/c/.../script.sh`. Inside the script: `export PATH=/usr/local/cuda/bin:$PATH` and call `$HOME/micromamba/envs/torch-dev/bin/frx` directly. Avoids the triple-layer quoting hell that mangles `$PATH` (Windows paths have spaces + parens) and `/mnt/c` path translation.
- Full suite after changes: 462 passed in the ncu/bench/cli/analysis subset; only the pre-existing nvcc + tmp_path env failures remain.

# 2026-06-17: `frx bench` kernel-time verdict (P0a — trustworthy speedup)

- **Problem this fixes:** after the Blackwell pass, the analyzer was sound but `frx bench`'s headline "speedup" was the least trustworthy number in the pipeline — whole-process wall time is dominated by ~160ms CUDA context init, so a 3x-faster micro-kernel still reads ~1.0x. A misleading number is worse than no number.
- **The cheap path that already existed:** `kernel_inspector._canonical_ncu_metric_name` already maps `gpu__time_duration.sum` / `.avg` / `duration` → `kernel_duration_us` and stores it per-kernel (used by `kernel_attribution.runtime_share_pct`). The gap was only: (1) no preset *requested* the metric, (2) `derive_ncu_run_summary` never aggregated it, (3) `diff_ncu_runs` never surfaced it. So P0a was pure plumbing, no new parsing.
- **What changed:**
  - `ncu_presets.py`: added `gpu__time_duration.sum` to the `memory` preset (so `full` inherits it). It is NOT a PC-sampling metric, so it survives Blackwell gating — verified by test.
  - `ncu_analysis.derive_ncu_run_summary`: aggregates `total_kernel_duration_us` (sum across kernels) + `kernels_with_duration_data`. Sum, not avg — the before/after total is the like-for-like basis for a ratio.
  - `ncu_comparison.diff_ncu_runs`: new `kernel_time` block `{available, baseline_us, optimized_us, speedup_x}`. `_build_verdict(bottleneck_diff, kernel_time)` now labels the headline `outcome` from kernel time when available (noise band 0.95–1.05 → neutral), keeps `bottleneck_outcome` separately, and tags `basis` = `kernel_gpu_time` | `bottleneck_diff`. Back-compat: legacy CSVs with no duration fall back to bottleneck basis and the old outcomes are unchanged.
  - `bench.bench_compare`: times compile per side (`compile_ms`), and sets `primary_speedup_x` / `primary_speedup_basis` — kernel GPU time when NCU has it, else wall clock.
  - `cli._print_bench_report`: three timings (compile / process wall / kernel GPU), and **interprets the gap** — if kernel ≥1.10x but wall ∈[0.9,1.1], prints "host overhead/CUDA init dominates this binary, not the kernel" (the same where-did-the-time-go diagnosis the telemetry path makes). Verdict prints its basis and flags any kernel-vs-bottleneck disagreement.
- **Honesty caveat baked into the output:** NCU serializes/replays kernels, so the *absolute* `kernel GPU time` is profiler-inflated vs production — but the before/after *ratio* is valid because both runs pay the same tax. The report says this in-line. The fully profiler-free absolute number is P0b (a `cudaEvent` bench-harness convention + `FRX_KERNEL_US:` stdout sentinel), deferred.
- **Design note:** the kernel-time-vs-bottleneck disagreement is a *feature*, not noise — "bottleneck resolved but kernel time flat" is itself diagnostic, so the verdict surfaces both rather than letting one silently win.
- Tests: +5 `test_ncu_comparison`, +2 `test_ncu_presets`, +2 `test_ncu_analysis`, +5 `test_bench` (incl. a `_print_bench_report` gap-note assertion). Full suite: 903 passed / 3 skipped; only the pre-existing nvcc (Windows host) + tmp_path-perm env failures remain. Not committed (user commits).

# 2026-06-17: `frx bench` P0b — profiler-free cudaEvent timing (and what it exposed)

- **What shipped:** `backend/python/fournex/data/frx_bench_harness.cuh` — a header providing `frx_bench([&]{ kernel<<<g,b>>>(...); })` that times only the kernel region with cudaEvents (warmup + N launches, median) and prints `FRX_KERNEL_US: <µs>`. `bench.compile_kernel` auto-adds `-I <package data dir>` so a bare `#include "frx_bench_harness.cuh"` resolves (no -I for the user). `bench.time_binary` parses the sentinel → `kernel_event_us` (median across runs). `bench_compare` computes `event_speedup_x` + a `kernel_event` block; **basis priority = cuda_event > kernel_gpu_time (NCU) > wall_clock**. `cli`: `frx bench --emit-harness [PATH]` writes the header; report shows all three timings, marks the verdict basis, and keeps the gap-interpretation note. All 8 cuda_zoo demos retrofitted to use the harness. Shipped in wheel via `pyproject` `"*.cuh"`. +10 tests, full suite 913 passed/3 skipped.
- **Harness robustness (validated live):** it calls `cudaGetLastError()` after the loop and, on error, prints to **stderr** and emits **no** sentinel — so a failed kernel falls back rather than reporting garbage. This fired for real (see 03 below).
- **THE BIG FINDING — kernel timing exposed that 3 of the 4 cuda_zoo demos don't actually show a kernel speedup.** Wall clock (init-dominated, ~1.0x) had masked all of it; the demos "passed" only because the *static/NCU bottleneck classifier* flagged the right anti-pattern. Measured on real sm_120:
  - **02 matmul_notiled: ✅ 1.36x** (tiled beats naive) — the one honest, fair demo. Proof P0b works.
  - **01 uncoalesced: ❌ 0.58x (good SLOWER).** `bad.cu` launches `N/STRIDE` threads (copies 32K elements); `good.cu` launches `N` threads (copies 1M). The "good" kernel does **32x more work** — never an apples-to-apples comparison. NCU agreed (0.31x) while still (correctly) reporting `uncoalesced_access` RESOLVED.
  - **03 excess_sync: ⚠️ illegal memory access in `bad.cu`** — `reduction_oversync` reads `sdata[tid + stride]` with `tid+stride` up to ~1535 against `__shared__ float sdata[1024]` (OOB). Harness withheld the sentinel; bench fell back to a meaningless 9.98x wall number.
  - **04 register_pressure: ❌ 0.70x (good SLOWER).** The low-register "good" version splits into two kernels with a global round-trip; at this tiny per-thread workload the occupancy gain can't offset the extra DRAM traffic (the demo's own comment admits the tradeoff).
- **Deeper lesson (credibility loop):** the case-study/static harness validates "good *resolves* the bottleneck" but never verified "good is actually *faster*" — and for 3/4 pairs it isn't. **Static bottleneck resolution ≠ runtime speedup.** P0b closes that loop; the demo suite needs fixing so each `good.cu` is a fair workload that is genuinely faster AND still triggers the right finding. Relates to [[project_case_study_harness]] and [[feedback_lessons_learned]].
- **Demo fixes applied + verified on sm_120 (2026-06-17):**
  - **01 uncoalesced — redesigned to equal work.** Both kernels now copy all N elements; `bad.cu` reads column-major (`src[col * width + row]`, strided by `width`) vs `good.cu` row-major (`src[tid]`). Result: cudaEvent **3.07x faster** (41.8→13.6µs), `uncoalesced_access` RESOLVED. NOTE the static `strided_or_pitched` detector (`cuda_static._memory_access_styles`) only fires on a subscript multiply against a stride keyword `(stride|pitch|ld|width)` — so the strided index MUST use one of those names (`width` here) or the `uncoalesced_access` finding won't trigger. (Minor: the coalesced copy now trips a `l2_cache_thrashing` NEW bottleneck — a streaming copy legitimately has ~0 L2 reuse; classifier-tuning nit, doesn't change the "improved" verdict.)
  - **03 excess_sync — fixed illegal memory access.** `float val = (tid < stride) ? sdata[tid + stride] : 0.0f;` guards the OOB read while keeping all three `__syncthreads()` in the loop (so `sync_inside_tight_loop` still fires). Result: runs clean, cudaEvent **2.19x faster** (55.7→25.4µs). Good P0b showcase: NCU bottleneck diff is neutral (0 resolved/0 new) yet kernel time correctly reports the 2.19x win.
  - **04 register_pressure — initially documented, then redesigned (see below).**
  - `test_zoo_static.py` still green (all bad.cu still trigger their findings). Full suite 913 passed/3 skipped.

# 2026-06-18: 04 register_pressure — compute-bound redesign (the Volkov trap)

- **Measured the naive premise and it's FALSE on modern HW.** "Many independent live accumulators = high register pressure = slow" does NOT hold: on sm_120, 32 indep accumulators (39 regs) ran 0.97x (good *slower*), and 128 accumulators (134 regs, no spill) only 1.06x. The extra registers feed **ILP that hides latency even at low occupancy** (Volkov's "better performance at lower occupancy"). This is exactly why the original demo never showed a speedup.
- **Register pressure only costs runtime once it SPILLS.** Swept FEAT accumulators with `nvcc -Xptxas -v`: 192→197 regs/no spill/1.00x; **256→255 regs + 72B spill → 1.98x**; 384→255 regs + 1256B spill → 19x. Spilling to local memory (off-chip DRAM) is the real harm. Chose **FEAT=256** for the demo: clearly spills, believable ~2x (not a cartoonish 19x).
- **Final demo (single kernel each, equal work):** `bad.cu` holds `float acc[256]` live across a dependent ITERS loop (255 regs, 72B spill); `good.cu` computes the identical sum one chain at a time (10 regs, 0 spill). cudaEvent **1.98x faster**, verified live. No global round-trip (the old "good" used a 2-kernel split that added DRAM traffic — itself part of why it was slower).
- (placeholder — see register-pressure analyzer note below)

# 2026-06-18: eager-vs-torch.compile validation + the SDK profiler stub fix

- **Goal:** first bridge from synthetic CUDA-zoo demos toward real framework behavior — does Fournex correctly explain eager PyTorch launch overhead vs torch.compile fusion? Built `backend/examples/train_pointwise.py` (a long chain of tiny elementwise ops on small tensors — the launch-overhead regime).
- **Eager result (correct, the win):** Fournex diagnoses it perfectly — kernel_count_per_step ≈ 280–1116, small_kernel_fraction ≈ 1.0 (≈100% sub-10us kernels), GPU util 1–7%, framework_abstraction_tax 93–99, `launch_bound` 0.5, explain → "Primary issue: launch bound (high)." Real framework behavior, correctly characterized.
- **torch.compile is BLOCKED in this env (not a Fournex bug):** the source-built PyTorch (`/home/jorge/src/pytorch`) has **no Triton** → `torch._inductor.exc.TritonMissing` → torch.compile can't generate kernels; the compiled model errors during warmup and every Kineto trace came back EMPTY (0 events). So the eager→compiled "resolved" half can't be shown here until Triton is installed. Did NOT fake it.
- **THE GAP (found + fixed): the live path's kernel-count signals had no real data source.** `launch_bound` and `framework_abstraction_tax` both gate on `profiler_windows_exported > 0` with a `kernel_count` payload, read by the reducer (`analysis.py` ~712) from `profiler_window` events in the raw trace. But:
  1. `SampledProfilerController._export_summary()` was a **metadata-only stub** — wrote `recorded_ops: 0`, emitted NO `kernel_count`. So `frx collect` with plain SDK instrumentation could NEVER produce kernel counts.
  2. The only real source was an externally-captured torch.profiler Chrome trace ingested via `_profiler_trace_to_sdk_events`/`_events_from_profiler_bundle` — but `analyze` only takes that path when there is **no raw SDK trace** (`_load_or_generate_summary`: derived → raw → profiler-bundle). With an SDK raw trace present, the profiler trace is silently ignored.
  Net: the launch-overhead diagnosis was unreachable through the normal SDK collect UX; I had to run a profiler-trace-only bundle to get any numbers.
- **The fix (chose: wire the SDK profiler):** implemented `SampledProfilerController` to actually run `torch.profiler.profile()` over each sampled window, export a Chrome trace, and count kernels with a new shared `summarize_chrome_trace_kernels()` (cat=="kernel", per-step, median, small-fraction — same counting as the analyze-time ingestion, so the two agree). The exported `profiler_window` event now carries `kernel_count` / `median_cuda_kernel_duration_us` / `small_kernel_fraction` into the raw trace, where the existing reducer already consumes them — **no merge needed, and it keys off the SDK's own step ids (no double-counting).** `frx.init` auto-configures the controller when `FRX_PROFILER_ENABLED=1` (set by `frx collect`; opt out with `FRX_PROFILER=0`); torch-absent / capture-failure falls back to the old metadata-only behavior. Verified live: plain `frx collect -- python train_pointwise.py --mode eager` (no manual profiler trace) now yields kernel_count_per_step=280, framework_tax=99, launch_bound high. +6 tests (`test_profiler_controller.py`), 921 passed.
- **Remaining (noted, not fixed):** (a) `analyze` still ignores an *external* `profiler/*.json` when a raw trace exists — lower priority now that the SDK captures counts itself; (b) an empty/unparseable profiler trace degrades to `primary: none` while a captured nvidia-smi util (e.g. 2.4%) goes unmentioned — Fournex is correctly refusing to over-claim, but could surface the empty-trace + util more loudly. Relates to [[project_explain_training]], [[project_framework_abstraction_tax]], [[feedback_lessons_learned]].

# 2026-06-17 (cont.): register-pressure analyzer note
- **Analyzer gap this surfaced + fixed:** `high_register_pressure` keyed on *named scalar* count (`float NAME=|;`) and was BLIND to `float acc[256]` — i.e. blind to the very thing that spills. Taught `cuda_rules/engine.py` to add statement-leading **local** array element counts to `local_var_count` (regex `(?:^|[;{}])\s*<type>\s+\w+\s*\[(\d+)\]`). Statement-leading match excludes `__shared__` arrays (preceded by the qualifier) — shared mem isn't register pressure. Literal sizes only (no preprocessor, so `acc[FEAT]` macro wouldn't match — the demo uses a literal `256`). +2 tests in test_cuda_static (large local array fires; shared array does not). Full suite 915 passed/3 skipped. All 4 cuda_zoo demos now show genuine, GPU-verified speedups: 01=3.07x, 02=1.36x, 03=2.19x, 04=1.98x.
