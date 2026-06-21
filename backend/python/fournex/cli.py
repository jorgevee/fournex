from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any


@dataclasses.dataclass
class StepResult:
    ok: bool
    output: str | None = None
    error: str | None = None
    step: str = ""


BUNDLE_SCHEMA_VERSION = "0.1.0"
DEFAULT_CASE_STUDY_ROOT = "demos/cuda_zoo"
EXPECTED_ARTIFACTS = (
    "metadata.json",
    "run_config.yaml",
    "gpu_metrics.csv",
    "optional_logs.txt",
)


def _configure_logging(args: argparse.Namespace) -> None:
    """Set the ``fournex`` logger level/handler from --verbose/--debug.

    Defaults to WARNING so normal runs stay quiet; -v raises to INFO and
    --debug to DEBUG. Logs go to stderr so they never pollute stdout output
    (JSON, briefs) that callers may pipe.
    """
    if getattr(args, "debug", False):
        level = logging.DEBUG
    elif getattr(args, "verbose", False):
        level = logging.INFO
    else:
        level = logging.WARNING

    root = logging.getLogger("fournex")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args)

    try:
        return _dispatch(parser, args)
    except (FileNotFoundError, ValueError) as exc:
        # User-facing input errors (bad --arch-profile/--config path, malformed
        # override YAML, etc.) should print a clean message, not a traceback.
        print(f"frx: {exc}", file=sys.stderr)
        return 1


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.command == "collect":
        args.workload_command = _normalize_workload_command(args.workload_command)
        args.workload_command = _resolve_workload_command(args.workload_command)
        if not args.workload_command:
            parser.error("collect requires a workload command after --")
        return collect(args)
    elif args.command == "analyze":
        return analyze(args)
    elif args.command == "init":
        return init_cmd(args)
    elif args.command == "doctor":
        return doctor(args)
    elif args.command == "smoke-test":
        return smoke_test(args)
    elif args.command == "ncu-command":
        args.workload_command = _normalize_workload_command(args.workload_command)
        args.workload_command = _resolve_workload_command(args.workload_command)
        return ncu_command(args)
    elif args.command == "profile":
        args.workload_command = _normalize_workload_command(args.workload_command)
        args.workload_command = _resolve_workload_command(args.workload_command)
        return profile(args)
    elif args.command == "tune":
        args.workload_command = _normalize_workload_command(args.workload_command)
        args.workload_command = _resolve_workload_command(args.workload_command)
        if not args.workload_command:
            parser.error("tune requires a workload command after --")
        return tune(args)
    elif args.command == "compare":
        return compare(args)
    elif args.command == "compare-variants":
        return compare_variants_cmd(args)
    elif args.command == "explain":
        return explain_cmd(args)
    elif args.command == "bench":
        return bench_cmd(args)
    elif args.command == "case-study":
        return case_study_cmd(args)
    elif args.command == "eval":
        return eval_cmd(args)
    else:
        parser.print_help()
        return 1


def collect(args: argparse.Namespace) -> int:
    run_id = args.run_id or f"run-{uuid.uuid4().hex[:12]}"
    job_name = args.name or "frx-run"
    output_root = Path(args.out)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = run_dir / "raw"
    derived_dir = run_dir / "derived"
    profiler_dir = run_dir / "profiler"
    raw_dir.mkdir(exist_ok=True)
    derived_dir.mkdir(exist_ok=True)
    profiler_dir.mkdir(exist_ok=True)

    started_at = datetime.now(timezone.utc)
    monotonic_start = time.perf_counter()
    logs_path = run_dir / "optional_logs.txt"
    gpu_metrics_path = run_dir / "gpu_metrics.csv"
    warnings: list[str] = []

    env = os.environ.copy()
    env.update(
        {
            "FRX_RUN_ID": run_id,
            "FRX_JOB_NAME": job_name,
            "FRX_OUTPUT_DIR": str(run_dir),
            "FRX_RAW_TRACE_PATH": str(raw_dir / "trace.jsonl"),
            "FRX_DERIVED_SUMMARY_PATH": str(derived_dir / "summary.json"),
            "FRX_AUTO_PERSIST": "1",
            "FRX_SAMPLE_INTERVAL_MS": str(args.sample_interval_ms),
            # Enable the SDK sampled profiler so step_context captures real CUDA
            # kernel counts (feeds launch_bound / framework_abstraction_tax).
            # The workload can opt out with FRX_PROFILER=0.
            "FRX_PROFILER_ENABLED": "1",
        }
    )

    config = _build_run_config(args, run_id, job_name, run_dir)
    _write_yaml(run_dir / "run_config.yaml", config)

    stop_sampling = Event()
    sampler = Thread(
        target=_sample_gpu_metrics,
        args=(gpu_metrics_path, args.sample_interval_ms, stop_sampling, warnings),
        daemon=True,
    )
    sampler.start()

    exit_code = 1
    try:
        with logs_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                args.workload_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log_handle.write(line)
            exit_code = process.wait()
    finally:
        stop_sampling.set()
        sampler.join(timeout=2.0)

    ended_at = datetime.now(timezone.utc)
    duration_s = time.perf_counter() - monotonic_start

    artifact_dirs = [Path(d) for d in args.artifact_dirs] if args.artifact_dirs else [Path.cwd() / "frx-job-run"]
    imported = _import_workload_bundle_artifacts(
        run_dir,
        artifact_dirs,
        warnings,
        import_profiler=not args.no_profiler_import,
    )
    if imported:
        warnings.append(f"Imported workload artifacts: {', '.join(imported)}")
    environment = config.get("environment") if isinstance(config.get("environment"), dict) else None
    # overwrite=True: the parent owns the derived summary because only it has the
    # captured GPU metrics; the subprocess's GPU-less summary must not win.
    _generate_derived_summary_from_trace(run_dir, warnings, environment=environment, overwrite=True)
    _generate_derived_summary_from_profiler_bundle(run_dir, warnings, environment=environment)
    artifacts = _discover_artifacts(run_dir)
    _append_limited_bundle_warnings(artifacts, warnings)
    metadata = _build_metadata(
        args=args,
        run_id=run_id,
        job_name=job_name,
        run_dir=run_dir,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        exit_code=exit_code,
        artifacts=artifacts,
        warnings=warnings,
    )
    _write_json(run_dir / "metadata.json", metadata)
    artifacts["metadata"] = "metadata.json"

    manifest = _build_manifest(run_dir, artifacts, warnings)
    _write_json(run_dir / "manifest.json", manifest)

    zip_path: Path | None = None
    if not args.no_zip:
        zip_path = output_root / f"{run_id}.zip"
        _zip_run_dir(run_dir, zip_path)

    _print_collection_summary(run_dir, zip_path, manifest, exit_code, imported)

    if getattr(args, "explain", False):
        print("\n  Generating LLM brief from run data...")
        explain_out = Path(getattr(args, "explain_out", None) or ".")
        rc = _explain_training(args, run_dir)
        if rc == 0:
            print(f"  LLM brief written to {explain_out / 'frx_llm_prompt.txt'}")
            print("  Paste frx_llm_prompt.txt into your LLM for optimization suggestions.")

    return exit_code

def tune(args: argparse.Namespace) -> int:
    from .autopilot.actions import PromotionThresholds
    from .autopilot.benchmark import BenchmarkWindow
    from .autopilot.quality import QualityPolicy
    from .autopilot.report import format_report
    from .autopilot.runner import ExperimentRunner
    from .autopilot.safety import SafetyPolicy

    environment = _detect_environment()
    environment["require_quality_checks"] = args.require_quality_checks
    thresholds = PromotionThresholds(min_speedup=args.min_speedup)
    safety_policy = SafetyPolicy(
        allow_risky_actions=args.allow_risky_actions,
        require_quality_checks_for_precision=args.require_quality_checks,
    )
    benchmark_window = BenchmarkWindow(
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measure_steps,
        repeat_count=args.repeat_count,
        timeout_s=args.time_budget_s,
    )
    quality_policy = QualityPolicy(
        max_final_loss_regression=args.max_final_loss_regression,
        max_loss_divergence=args.max_loss_divergence,
        output_abs_tolerance=args.output_abs_tolerance,
        require_finite_loss=args.require_finite_loss,
    )

    runner = ExperimentRunner(
        workload_command=args.workload_command,
        job_name=args.name,
        out_dir=args.out,
        max_trials=args.max_trials,
        safe_only=args.safe,
        benchmark_window=benchmark_window,
        race_enabled=not args.no_race,
        race_promote_count=args.race_promote_count,
        race_warmup_steps=args.race_warmup_steps,
        race_measure_steps=args.race_measure_steps,
        sample_interval_ms=args.sample_interval_ms,
        thresholds=thresholds,
        environment=environment,
        bottleneck_diagnosis=args.bottleneck,
        safety_policy=safety_policy,
        quality_policy=quality_policy,
        resume_dir=args.resume,
        verbose=True,
    )
    report = runner.run()
    print(format_report(report))
    return 0 if report.improved else 1


def compare(args: argparse.Namespace) -> int:
    """Compare two CUDA source files or evidence files across available layers."""
    # Evidence-comparison mode: --before/--after or layer-specific flags
    if _has_comparison_args(args):
        return _analyze_comparison(args)

    # Source-comparison mode: two positional .cu files
    if not args.file_a or not args.file_b:
        print(
            "frx compare: provide two source files  OR  use --before/--after for evidence comparison.\n"
            "  frx compare baseline.cu optimized.cu\n"
            "  frx compare --before before.csv --after after.csv",
            file=sys.stderr,
        )
        return 1

    from .comparison import compare_implementations
    from .cuda_static import inspect_cuda_source
    from .reconciliation import reconcile_evidence

    file_a = Path(args.file_a)
    file_b = Path(args.file_b)

    for path in (file_a, file_b):
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1
        if path.suffix.lower() not in {".cu", ".cuh", ".cuda"}:
            print(f"Error: expected a .cu source file, got: {path}", file=sys.stderr)
            return 1

    label_a = args.label_a or file_a.name
    label_b = args.label_b or file_b.name

    src_a = _read_text_file(file_a)
    src_b = _read_text_file(file_b)

    # PTX layer
    ptx_a = ptx_b = None
    evidence_failures: list[dict[str, Any]] = []
    if args.with_ptx or args.with_ncu:
        print(f"  Compiling PTX for {file_a.name} ...", end="", flush=True)
        res_a = _compile_ptx_cu(file_a)
        print(" ok" if res_a.ok else " failed")
        if res_a.ok:
            ptx_a = res_a.output
        else:
            evidence_failures.append({"layer": "ptx", "target": label_a, "step": res_a.step, "reason": res_a.error})

        print(f"  Compiling PTX for {file_b.name} ...", end="", flush=True)
        res_b = _compile_ptx_cu(file_b)
        print(" ok" if res_b.ok else " failed")
        if res_b.ok:
            ptx_b = res_b.output
        else:
            evidence_failures.append({"layer": "ptx", "target": label_b, "step": res_b.step, "reason": res_b.error})

    # NCU layer — pre-existing CSVs take priority over --with-ncu compilation
    ncu_a_text = ncu_b_text = None
    if args.ncu_a:
        ncu_path = Path(args.ncu_a)
        if not ncu_path.exists():
            print(f"Error: NCU CSV not found: {ncu_path}", file=sys.stderr)
            return 1
        ncu_a_text = ncu_path.read_text(encoding="utf-8-sig", errors="replace")
    if args.ncu_b:
        ncu_path = Path(args.ncu_b)
        if not ncu_path.exists():
            print(f"Error: NCU CSV not found: {ncu_path}", file=sys.stderr)
            return 1
        ncu_b_text = ncu_path.read_text(encoding="utf-8-sig", errors="replace")

    if args.with_ncu and not (ncu_a_text and ncu_b_text):
        ncu_res_a, ncu_res_b = _run_ncu_compare(file_a, file_b, ptx_a, ptx_b, args)
        if ncu_res_a.ok:
            ncu_a_text = ncu_res_a.output
        else:
            evidence_failures.append({"layer": "ncu", "target": label_a, "step": ncu_res_a.step, "reason": ncu_res_a.error})
        if ncu_res_b.ok:
            ncu_b_text = ncu_res_b.output
        else:
            evidence_failures.append({"layer": "ncu", "target": label_b, "step": ncu_res_b.step, "reason": ncu_res_b.error})

    # Determine evidence description for header
    layers: list[str] = ["CUDA source"]
    if ptx_a and ptx_b:
        layers.append("PTX")
    if ncu_a_text and ncu_b_text:
        layers.append("NCU")
    evidence_desc = " + ".join(layers)

    # Resolve GPU/arch settings once (loads any --arch-profile YAML a single time)
    env = _environment_from_args(args)

    # Build comparison inputs
    input_a: dict[str, Any] = {
        "label": label_a,
        "cuda_source": src_a,
        "cuda_filename": str(file_a),
        "gpu_model": args.gpu_model,
    }
    input_b: dict[str, Any] = {
        "label": label_b,
        "cuda_source": src_b,
        "cuda_filename": str(file_b),
        "gpu_model": args.gpu_model,
    }
    arch_overrides = env.get("arch_profile_overrides")
    if arch_overrides:
        input_a["arch_profile_overrides"] = arch_overrides
        input_b["arch_profile_overrides"] = arch_overrides
    if ptx_a:
        input_a.update({"ptx": ptx_a, "ptx_filename": str(file_a.with_suffix(".ptx"))})
    if ptx_b:
        input_b.update({"ptx": ptx_b, "ptx_filename": str(file_b.with_suffix(".ptx"))})
    if ncu_a_text:
        input_a["ncu_csv"] = ncu_a_text
    if ncu_b_text:
        input_b["ncu_csv"] = ncu_b_text

    result = compare_implementations(input_a, input_b)

    # Reconciliation for the A (baseline) side
    static_a = inspect_cuda_source(src_a, filename=str(file_a), gpu_model=args.gpu_model)
    ptx_result_a = None
    if ptx_a:
        from .ptx_analysis import analyze_ptx_text
        ptx_result_a = analyze_ptx_text(ptx_a, filename=str(file_a.with_suffix(".ptx")))
    ncu_result_a = None
    if ncu_a_text:
        from .ncu_analysis import analyze_ncu_csv_text
        ncu_result_a = analyze_ncu_csv_text(ncu_a_text, environment=env)

    rec = reconcile_evidence(static=static_a, ptx=ptx_result_a, ncu=ncu_result_a)

    if args.output_json:
        _print_json_result("compare", {"comparison": result, "reconciliation": rec, "evidence_failures": evidence_failures})
        return 0

    _print_compare_report(result, rec, label_a, label_b, evidence_desc, evidence_failures)
    return 0


def _compile_ptx_cu(cu_path: Path) -> StepResult:
    """Compile a .cu file to PTX text. Returns a StepResult (ok=True on success)."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return StepResult(ok=False, step="nvcc -ptx", error="nvcc not found on PATH")
    with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [nvcc, "-ptx", "-o", str(tmp_path), str(cu_path)],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            return StepResult(
                ok=False, step="nvcc -ptx",
                error=f"nvcc exited {proc.returncode}: {proc.stderr.strip()[:200]}",
            )
        return StepResult(ok=True, output=tmp_path.read_text(encoding="utf-8", errors="replace"), step="nvcc -ptx")
    except subprocess.TimeoutExpired:
        return StepResult(ok=False, step="nvcc -ptx", error="nvcc -ptx timed out after 120s")
    except OSError as exc:
        return StepResult(ok=False, step="nvcc -ptx", error=f"nvcc could not run: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)


def _run_ncu_compare(
    file_a: Path, file_b: Path,
    ptx_a: str | None, ptx_b: str | None,
    args: argparse.Namespace,
) -> tuple[StepResult, StepResult]:
    """Compile both .cu files to executables and run NCU on each. Returns (StepResult_a, StepResult_b)."""
    nvcc = shutil.which("nvcc")
    ncu_bin = shutil.which("ncu")

    if not nvcc:
        msg = "nvcc not found on PATH"
        print(f"[warn] --with-ncu: {msg} — skipping NCU", file=sys.stderr)
        print("  Provide pre-existing CSVs with --ncu-a FILE --ncu-b FILE", file=sys.stderr)
        return StepResult(ok=False, step="nvcc", error=msg), StepResult(ok=False, step="nvcc", error=msg)
    if not ncu_bin:
        msg = "ncu not found on PATH"
        print(f"[warn] --with-ncu: {msg} — skipping NCU", file=sys.stderr)
        print("  Provide pre-existing CSVs with --ncu-a FILE --ncu-b FILE", file=sys.stderr)
        return StepResult(ok=False, step="ncu", error=msg), StepResult(ok=False, step="ncu", error=msg)

    build_flags = args.build_flags.split() if args.build_flags.strip() else []

    def _compile_exec(cu: Path) -> StepResult:
        with tempfile.NamedTemporaryFile(suffix=".exe" if sys.platform == "win32" else "", delete=False) as tmp:
            out_path = Path(tmp.name)
        try:
            cmd = [nvcc, str(cu), "-o", str(out_path)] + build_flags
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                reason = f"nvcc exited {proc.returncode}: {proc.stderr.strip()[:200]}"
                print(f"  [warn] nvcc failed for {cu.name}: {proc.stderr[:200]}", file=sys.stderr)
                out_path.unlink(missing_ok=True)
                return StepResult(ok=False, step="nvcc", error=reason)
            return StepResult(ok=True, output=str(out_path), step="nvcc")
        except subprocess.TimeoutExpired:
            reason = "nvcc timed out after 120s"
            print(f"  [warn] nvcc error for {cu.name}: {reason}", file=sys.stderr)
            return StepResult(ok=False, step="nvcc", error=reason)
        except OSError as exc:
            reason = f"nvcc could not run: {exc}"
            print(f"  [warn] nvcc error for {cu.name}: {exc}", file=sys.stderr)
            return StepResult(ok=False, step="nvcc", error=reason)

    def _run_ncu_on(exe_path: str) -> StepResult:
        try:
            from .ncu_presets import build_ncu_command
            from .arch_profiles import resolve_sm_version
            cmd = build_ncu_command("full", [exe_path], sm_version=resolve_sm_version(args.gpu_model))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                reason = f"ncu exited {proc.returncode}: {proc.stderr.strip()[:200]}"
                print(f"  [warn] ncu failed: {proc.stderr[:200]}", file=sys.stderr)
                return StepResult(ok=False, step="ncu", error=reason)
            return StepResult(ok=True, output=proc.stdout, step="ncu")
        except subprocess.TimeoutExpired:
            reason = "ncu timed out after 300s"
            print(f"  [warn] ncu error: {reason}", file=sys.stderr)
            return StepResult(ok=False, step="ncu", error=reason)
        except OSError as exc:
            reason = f"ncu could not run: {exc}"
            print(f"  [warn] ncu error: {exc}", file=sys.stderr)
            return StepResult(ok=False, step="ncu", error=reason)

    print(f"  Compiling {file_a.name} ...", end="", flush=True)
    compile_a = _compile_exec(file_a)
    print(" ok" if compile_a.ok else " failed")

    print(f"  Compiling {file_b.name} ...", end="", flush=True)
    compile_b = _compile_exec(file_b)
    print(" ok" if compile_b.ok else " failed")

    ncu_a = ncu_b = StepResult(ok=False, step="ncu", error="executable build failed")
    try:
        if compile_a.ok:
            print(f"  Profiling {file_a.name} with ncu ...", end="", flush=True)
            ncu_a = _run_ncu_on(compile_a.output)  # type: ignore[arg-type]
            print(" ok" if ncu_a.ok else " failed")
        else:
            ncu_a = StepResult(ok=False, step="ncu", error=f"executable build failed: {compile_a.error}")
        if compile_b.ok:
            print(f"  Profiling {file_b.name} with ncu ...", end="", flush=True)
            ncu_b = _run_ncu_on(compile_b.output)  # type: ignore[arg-type]
            print(" ok" if ncu_b.ok else " failed")
        else:
            ncu_b = StepResult(ok=False, step="ncu", error=f"executable build failed: {compile_b.error}")
    finally:
        if compile_a.ok and compile_a.output:
            Path(compile_a.output).unlink(missing_ok=True)
        if compile_b.ok and compile_b.output:
            Path(compile_b.output).unlink(missing_ok=True)

    if not ncu_a.ok or not ncu_b.ok:
        print(
            "  [hint] If your .cu files need a main(), compile with: --build-flags \"-DBUILD_EXECUTABLE\"",
            file=sys.stderr,
        )

    return ncu_a, ncu_b


# ── Finding / dimension name maps ──────────────────────────────────────────────

_FINDING_NAMES: dict[str, str] = {
    "unnecessary_syncthreads":      "spurious __syncthreads()",
    "conditional_syncthreads":      "conditional __syncthreads() in divergent branch",
    "missing_obvious_bounds_guard": "missing bounds guard",
    "strided_or_pitched":           "strided/pitched memory access",
    "bank_conflict_risk":           "shared memory bank conflict risk",
    "register_spills_detected":     "register spills",
    "high_register_count":          "high register count",
    "very_high_register_count":     "very high register count",
    "high_global_memory_ratio":     "high global-memory instruction ratio",
    "no_shared_memory_usage":       "no shared memory usage",
    "fp64_detected":                "FP64 operations detected",
    "ptx_register_spills":          "register spills",
    "ptx_register_pressure":        "high register pressure",
    "ptx_global_memory_heavy":      "high global-memory instruction ratio",
    "ptx_high_branch_density":      "high branch density",
}

_DIM_NAMES: dict[str, str] = {
    "launch_efficiency":   "launch configuration",
    "sync_efficiency":     "synchronization overhead",
    "memory_efficiency":   "memory efficiency (DRAM / cache)",
    "compute_efficiency":  "compute efficiency (IPC)",
    "register_efficiency": "register usage / spills",
}


def _finding_label(code: str) -> str:
    return _FINDING_NAMES.get(code, code.replace("_", " "))


def _dim_label(dim: str) -> str:
    return _DIM_NAMES.get(dim, dim.replace("_", " "))


def _print_missing_evidence(diagnoses: list[dict[str, Any]]) -> None:
    actionable = [d for d in diagnoses if d.get("missing_evidence")]
    if not actionable:
        return
    print()
    print("── Missing evidence " + "─" * 47)
    for diag in actionable:
        me = diag["missing_evidence"]
        conf_now = diag["confidence"]
        conf_after = me["confidence_if_confirmed"]
        print(f"\n  {diag['display_name']}  [{conf_now} → {conf_after} if confirmed]")
        for m in me["metrics"]:
            print(f"    · {m['label']}")
            print(f"        {m['metric']}")
            lines = textwrap.wrap(m["why"], width=56)
            print(f"        {lines[0]}")
            for continuation in lines[1:]:
                print(f"        {continuation}")
        if me.get("ncu_command"):
            print(f"\n    Run:")
            cmd_parts = me["ncu_command"].split(" --csv ")
            print(f"      ncu --metrics {cmd_parts[0].replace('ncu --metrics ', '')} \\")
            print(f"          --csv ./report.csv ./your_kernel")
        print(f"\n    Or collect everything:")
        print(f"      ncu --set full --csv ./report.csv ./your_kernel")


def _print_compare_report(
    result: dict[str, Any],
    rec: dict[str, Any],
    label_a: str,
    label_b: str,
    evidence_desc: str,
    evidence_failures: list[dict[str, Any]] | None = None,
) -> None:
    sep = "=" * 66
    verdict = result.get("verdict", {})
    winner = verdict.get("overall_winner", "tie")
    winner_label = label_b if winner == "b" else (label_a if winner == "a" else "tie")
    score_a = verdict.get("score_a")
    score_b = verdict.get("score_b")

    print()
    print(sep)
    print("  frx compare")
    print(f"  A: {label_a}")
    print(f"  B: {label_b}")
    print(f"  Evidence: {evidence_desc}")
    print(sep)

    if evidence_failures:
        print()
        print("Evidence layers unavailable:")
        for failure in evidence_failures:
            target = failure.get("target", "")
            reason = failure.get("reason", "unknown error")
            layer = failure.get("layer", "")
            print(f"  evidence layer unavailable: {layer.upper()} ({target}) — {reason}")

    # Winner
    print()
    if winner == "tie":
        print("Winner: tie")
    else:
        score_str = ""
        if score_a is not None and score_b is not None:
            score_str = f"  (score {score_b:.3f} vs {score_a:.3f})"
        print(f"Winner: {winner_label}{score_str}")

    # Resolved findings (A → B: gone)
    resolved: list[str] = []
    for key in ("static_diff", "ptx_diff"):
        diff = result.get(key, {})
        if diff.get("available"):
            for code in diff.get("findings_diff", {}).get("resolved_in_b", []):
                label = _finding_label(code)
                if label not in resolved:
                    resolved.append(label)
    if resolved:
        print()
        print("Resolved in B:")
        for item in resolved:
            print(f"  {item}")

    # Regressions (new findings in B)
    regressions: list[str] = []
    for key in ("static_diff", "ptx_diff"):
        diff = result.get(key, {})
        if diff.get("available"):
            for code in diff.get("findings_diff", {}).get("new_in_b", []):
                label = _finding_label(code)
                if label not in regressions:
                    regressions.append(label)
    if regressions:
        print()
        print("Regressions in B (new findings):")
        for item in regressions:
            print(f"  {item}")

    # Improved and regressed scorecard dimensions
    dims_b = verdict.get("dimensions_won_by_b", [])
    dims_a = verdict.get("dimensions_won_by_a", [])
    sc = result.get("scorecard", {})

    if dims_b:
        print()
        print("Improved in B:")
        for dim in dims_b:
            d = sc.get(dim, {})
            sa, sb = d.get("score_a"), d.get("score_b")
            delta = (sb - sa) if sa is not None and sb is not None else None
            delta_str = f"  (+{delta:.2f})" if delta is not None and delta >= 0 else ""
            print(f"  {_dim_label(dim)}{delta_str}")

    if dims_a:
        print()
        print("Regressed in B:")
        for dim in dims_a:
            d = sc.get(dim, {})
            sa, sb = d.get("score_a"), d.get("score_b")
            delta = (sb - sa) if sa is not None and sb is not None else None
            delta_str = f"  ({delta:.2f})" if delta is not None else ""
            print(f"  {_dim_label(dim)}{delta_str}")

    # Tradeoffs
    tradeoffs = result.get("tradeoffs", [])
    if tradeoffs:
        print()
        print("Tradeoffs:")
        for t in tradeoffs[:3]:
            print(f"  {t.get('label', 'tradeoff')}: {t.get('message', '')}")

    # Root-cause diagnoses for A (from reconciliation)
    if rec["diagnoses"]:
        print()
        print("Root causes in A:")
        for d in rec["diagnoses"]:
            sev_sym = {"high": "!!", "medium": "! "}.get(d["severity"], "  ")
            conf = d["confidence"]
            layers_str = " + ".join(d["layers_confirming"])
            print(f"  {sev_sym} {d['display_name']}  [{conf} - {layers_str}]")

    # Still unknown
    unknown = _compare_unknown_items(result)
    if unknown:
        print()
        print("Still unknown (need more evidence):")
        for item in unknown:
            print(f"  {item}")

    # Upgrade hints
    _print_compare_upgrade_hints(result)

    # Missing evidence
    if rec.get("diagnoses"):
        _print_missing_evidence(rec["diagnoses"])

    print()
    print(sep)
    print()


def _compare_unknown_items(result: dict[str, Any]) -> list[str]:
    """List specific metrics that cannot be determined from available evidence."""
    sc = result.get("scorecard", {})
    has_ncu = result.get("ncu_diff", {}).get("available", False)
    unknown: list[str] = []

    if not sc.get("register_efficiency", {}).get("available"):
        unknown.append("register usage and spills")

    if not has_ncu:
        unknown.append("DRAM bandwidth, L1/L2 cache hit rates")
        unknown.append("tensor core utilization")
        unknown.append("achieved occupancy (measured)")
        unknown.append("runtime warp stall reasons")
    else:
        if not sc.get("compute_efficiency", {}).get("available"):
            unknown.append("warp scheduler / issue slot utilization")

    return unknown


def _print_compare_upgrade_hints(result: dict[str, Any]) -> None:
    has_ptx = result.get("ptx_diff", {}).get("available", False)
    has_ncu = result.get("ncu_diff", {}).get("available", False)
    sc = result.get("scorecard", {})

    hints: list[str] = []
    if not has_ptx and not sc.get("register_efficiency", {}).get("available"):
        hints.append("--with-ptx to unlock register efficiency scores")
    if not has_ncu:
        hints.append("--with-ncu or --ncu-a/--ncu-b to measure runtime hardware behavior")

    if hints:
        print()
        print("Run with " + ", ".join(hints) + ".")


def explain_cmd(args: argparse.Namespace) -> int:
    from pathlib import Path
    from .explain import (
        build_explain_result, render_summary_txt, render_llm_prompt_txt, render_evidence_json,
        build_telemetry_explain_result, render_training_summary_txt, render_training_llm_prompt_txt,
        _select_scope_data,
    )

    input_path = Path(args.ncu_path) if args.ncu_path else None
    if input_path is None or not input_path.exists():
        print(f"error: file or directory not found: {args.ncu_path}", file=sys.stderr)
        return 1

    # Auto-detect: directory or zip → training telemetry path; CSV → NCU path.
    if input_path.is_dir() or input_path.suffix.lower() == ".zip":
        return _explain_training(args, input_path)

    # ── NCU CSV path (existing) ────────────────────────────────────────────────
    from .ncu_analysis import analyze_ncu_csv_text
    from .cuda_static import inspect_cuda_source

    csv_text = input_path.read_text(encoding="utf-8", errors="replace")
    env = _environment_from_args(args)
    ncu_result = analyze_ncu_csv_text(csv_text, environment=env)

    static_result = None
    kernel_source: str | None = None
    src_path = Path(args.src_path) if args.src_path else None
    if src_path is not None:
        if not src_path.exists():
            print(f"warning: source file not found: {src_path}", file=sys.stderr)
        else:
            kernel_source = src_path.read_text(encoding="utf-8", errors="replace")
            static_result = inspect_cuda_source(kernel_source, filename=src_path.name, gpu_model=args.gpu_model)

    result = build_explain_result(ncu_result=ncu_result, static_result=static_result, environment=env)

    if args.prompt_only:
        print(render_llm_prompt_txt(result, kernel_source=kernel_source,
                                    src_filename=src_path.name if src_path else None))
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path  = out_dir / "frx_summary.txt"
    prompt_path   = out_dir / "frx_llm_prompt.txt"
    evidence_path = out_dir / "frx_evidence.json"

    summary_path.write_text(render_summary_txt(result, ncu_filename=input_path.name,
                                               src_filename=src_path.name if src_path else None),
                            encoding="utf-8")
    prompt_path.write_text(render_llm_prompt_txt(result, kernel_source=kernel_source,
                                                 src_filename=src_path.name if src_path else None),
                           encoding="utf-8")
    evidence_path.write_text(render_evidence_json(result), encoding="utf-8")

    primary = result.get("primary_diagnosis") or "(none)"
    diagnoses = result.get("diagnoses", [])
    conf = next((d["confidence"] for d in diagnoses if d["label"] == primary), "")
    _print_explain_footer(summary_path, prompt_path, evidence_path, primary, conf)
    return 0


def _explain_training(args: Any, run_path: Any) -> int:
    """Training-telemetry explain path (run dir or zip)."""
    from pathlib import Path
    from .explain import (
        build_telemetry_explain_result, render_training_summary_txt,
        render_training_llm_prompt_txt, render_evidence_json, _select_scope_data,
    )

    run_path = Path(run_path)

    # Zip extraction (reuse existing helper)
    import tempfile, shutil
    tmp_dir: str | None = None
    if run_path.suffix.lower() == ".zip":
        tmp_dir = tempfile.mkdtemp(prefix="frx_explain_")
        shutil.unpack_archive(str(run_path), tmp_dir)
        run_path = Path(tmp_dir)

    try:
        summary = _load_or_generate_summary(run_path)
        if summary is None:
            print(f"error: no analysis data found in {run_path}. "
                  "Run 'frx collect' first or check the directory.", file=sys.stderr)
            return 1

        scope_arg = getattr(args, "scope", "auto")
        scope_data = _select_scope_data(summary, scope_arg)
        run_id = run_path.name

        result = build_telemetry_explain_result(scope_data=scope_data)

        if getattr(args, "prompt_only", False):
            print(render_training_llm_prompt_txt(result))
            return 0

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path  = out_dir / "frx_summary.txt"
        prompt_path   = out_dir / "frx_llm_prompt.txt"
        evidence_path = out_dir / "frx_evidence.json"

        summary_path.write_text(render_training_summary_txt(result, run_id=run_id), encoding="utf-8")
        prompt_path.write_text(render_training_llm_prompt_txt(result), encoding="utf-8")
        evidence_path.write_text(render_evidence_json(result), encoding="utf-8")

        primary = result.get("primary_bottleneck") or "(none)"
        conf_level = result.get("confidence", {}).get("level", "")
        _print_explain_footer(summary_path, prompt_path, evidence_path, primary, conf_level)
        return 0
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _print_explain_footer(
    summary_path: Any, prompt_path: Any, evidence_path: Any,
    primary: str, conf: str,
) -> None:
    print()
    print("  frx explain complete")
    print()
    print(f"  frx_summary.txt     ->  {summary_path}")
    print(f"  frx_llm_prompt.txt  ->  {prompt_path}")
    print(f"  frx_evidence.json   ->  {evidence_path}")
    print()
    label = primary.replace("_", " ") if primary else "(none)"
    if conf:
        print(f"  Primary issue  : {label}  ({conf})")
    else:
        print(f"  Primary issue  : {label}")
    print("  Paste frx_llm_prompt.txt into your preferred LLM for optimization suggestions.")
    print()


def compare_variants_cmd(args: argparse.Namespace) -> int:
    """Rank multiple kernel variants by measured throughput + explain NCU deltas."""
    from pathlib import Path
    from .variant_comparison import load_variants_csv, analyze_variants

    results_path = Path(args.results_csv)
    try:
        variants = load_variants_csv(results_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"frx: {exc}", file=sys.stderr)
        return 1

    env = _environment_from_args(args)
    try:
        report = analyze_variants(
            variants,
            baseline_variant=getattr(args, "baseline", None) or None,
            environment=env,
        )
    except ValueError as exc:
        print(f"frx: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        _print_json_result("compare_variants", report)
        return 0

    baseline = report["baseline"]
    ranked = report["variants_ranked"]
    transitions = report["transitions"]

    print(f"\nfrx compare-variants -- {report['variant_count']} variants\n")

    # Header — account for " (baseline)" suffix in column width
    col_v = max(len(v["variant"]) + (len(" (baseline)") if v["is_baseline"] else 0) for v in ranked) + 2
    print(f"  {'Variant':<{col_v}} {'GFLOP/s':>9}  {'vs baseline':>12}  {'Primary bottleneck'}")
    print(f"  {'-'*col_v} {'-'*9}  {'-'*12}  {'-'*22}")
    for v in ranked:
        tag = " (baseline)" if v["is_baseline"] else ""
        label = f"{v['variant']}{tag}"
        delta = f"+{v['delta_vs_baseline_x']:.2f}x" if v["delta_vs_baseline_x"] is not None and not v["is_baseline"] else ("--" if v["is_baseline"] else "")
        bn = v["primary_bottleneck"] or "—"
        print(f"  {label:<{col_v}} {v['throughput']:>9.1f}  {delta:>12}  {bn}")

    if transitions:
        print(f"\nNotable transitions:")
        for t in transitions:
            delta_str = f"+{t['throughput_delta_x']:.2f}x" if t["throughput_delta_x"] is not None and t["throughput_delta_x"] >= 1.0 else (f"{t['throughput_delta_x']:.2f}x" if t["throughput_delta_x"] is not None else "?")
            resolved_str = f"  [{', '.join(t['bottleneck_resolved'])} resolved]" if t["bottleneck_resolved"] else ""
            new_str = f"  [{', '.join(t['bottleneck_new'])} introduced]" if t["bottleneck_new"] else ""
            headline = ""
            if t.get("headline_metric"):
                m = t["headline_metric"]
                unit = m.get("unit", "")
                a = f"{m['baseline']:.1f}{unit}" if m.get("baseline") is not None else "?"
                b = f"{m['optimized']:.1f}{unit}" if m.get("optimized") is not None else "?"
                headline = f"  {m['label']}: {a} -> {b}"
            print(f"  {t['from_variant']} -> {t['to_variant']}  {delta_str}{headline}{resolved_str}{new_str}")

    top_rec = report.get("top_recommendation")
    if top_rec:
        print(f"\nTop recommendation across all variants: {top_rec}")

    if getattr(args, "explain", False):
        # Generate LLM brief from the best-performing variant
        from .explain import build_explain_result, render_summary_txt, render_llm_prompt_txt, render_evidence_json
        best = ranked[0]
        explain_result = build_explain_result(ncu_result=best["ncu_result"], environment=env)
        out_dir = Path(getattr(args, "explain_out", None) or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "frx_summary.txt").write_text(render_summary_txt(explain_result, ncu_filename=best["variant"]), encoding="utf-8")
        (out_dir / "frx_llm_prompt.txt").write_text(render_llm_prompt_txt(explain_result), encoding="utf-8")
        (out_dir / "frx_evidence.json").write_text(render_evidence_json(explain_result), encoding="utf-8")
        print(f"\n  LLM brief written to {out_dir / 'frx_llm_prompt.txt'} (best variant: {best['variant']})")

    print()
    return 0


def bench_cmd(args: argparse.Namespace) -> int:
    from pathlib import Path
    from .bench import bench_compare, harness_header_path

    if args.emit_harness is not None:
        dest = Path(args.emit_harness)
        dest.write_text(harness_header_path().read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Wrote cudaEvent bench harness to {dest}")
        print('Include it in your kernel: #include "frx_bench_harness.cuh"')
        print("Then wrap your launch: frx_bench([&]{ my_kernel<<<g,b>>>(args); });")
        return 0

    if not args.before or not args.after:
        print("error: bench requires two .cu files: frx bench BEFORE AFTER", file=sys.stderr)
        return 1

    before_src = Path(args.before)
    after_src  = Path(args.after)
    for p in (before_src, after_src):
        if not p.exists():
            print(f"error: file not found: {p}", file=sys.stderr)
            return 1

    if args.with_ncu and not args.arch:
        print(
            "warning: --with-ncu without --arch may profile JIT-compiled code "
            "(NCU cannot capture JIT kernels; pass --arch sm_120 or similar)",
            file=sys.stderr,
        )

    out_dir = Path(args.out) if args.out else None

    try:
        result = bench_compare(
            before_src,
            after_src,
            warmup=args.warmup,
            runs=args.runs,
            with_ncu=args.with_ncu,
            arch=args.arch,
            build_flags=args.build_flags,
            out_dir=out_dir,
            preset="full",
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        print(json.dumps(result, indent=2))
        return 0 if not result["compile_errors"] else 1

    _print_bench_report(result)
    return 0 if not result["compile_errors"] else 1


def _print_bench_report(result: dict[str, Any]) -> None:
    before_name = Path(result["before"]["src"]).name
    after_name  = Path(result["after"]["src"]).name
    header = f"BENCH: {before_name} -> {after_name}"
    sep    = "=" * len(header)
    print()
    print(header)
    print(sep)

    if result["compile_errors"]:
        for err in result["compile_errors"]:
            print(f"\nCompile error ({err['side']}): {err['src']}")
            print(err["error"].strip())
        return

    arch_str = f" ({result['arch']})" if result.get("arch") else ""
    cb = result["before"].get("compile_ms")
    ca = result["after"].get("compile_ms")
    if cb is not None and ca is not None:
        print(f"\nCompile:   OK{arch_str}   (before {cb:.0f} ms / after {ca:.0f} ms)")
    else:
        print(f"\nCompile:   OK{arch_str}")

    bt = result["before"]["timing"]
    at = result["after"]["timing"]
    runs   = bt["runs"]
    warmup = bt["warmup"]
    wall_sx = result.get("speedup_x")
    basis = result.get("primary_speedup_basis")

    ncu_diff = result.get("ncu_diff")
    ncu_kt = (ncu_diff or {}).get("kernel_time") if ncu_diff else None
    ncu_sx = result.get("kernel_speedup_x")
    event = result.get("kernel_event") or {}
    event_sx = result.get("event_speedup_x")

    print("\nTiming:")
    wall_sx_str = f"   ({wall_sx:.2f}x)" if wall_sx is not None else ""
    wall_marker = "   <- verdict basis" if basis == "wall_clock" else ""
    print(
        f"  Process wall ({runs} runs, {warmup} warmup):  "
        f"{bt['median_ms']:.1f} ms -> {at['median_ms']:.1f} ms{wall_sx_str}{wall_marker}"
    )
    if event.get("available"):
        eb = event["baseline_us"]
        ea = event["optimized_us"]
        esx_str = f"   ({event_sx:.2f}x)" if event_sx is not None else ""
        marker = "   <- verdict basis" if basis == "cuda_event" else ""
        print(
            f"  Kernel GPU time (cudaEvent):        "
            f"{eb:,.1f} us -> {ea:,.1f} us{esx_str}{marker}"
        )
        print("    (in-binary harness: profiler-free, init-free — production-representative)")
    if ncu_kt and ncu_kt.get("available"):
        kb = ncu_kt["baseline_us"]
        ka = ncu_kt["optimized_us"]
        ksx_str = f"   ({ncu_sx:.2f}x)" if ncu_sx is not None else ""
        marker = "   <- verdict basis" if basis == "kernel_gpu_time" else ""
        print(
            f"  Kernel GPU time (NCU):              "
            f"{kb:,.1f} us -> {ka:,.1f} us{ksx_str}{marker}"
        )
        print("    (NCU serializes kernels: the absolute is profiler-inflated, the ratio is valid)")

    # The gap between kernel time and wall time is itself diagnostic: a fast kernel
    # inside a slow binary means host overhead, not the kernel, is the limiter.
    kernel_region_sx = event_sx if event_sx is not None else ncu_sx
    if kernel_region_sx is not None and wall_sx is not None and kernel_region_sx >= 1.10 and 0.9 <= wall_sx <= 1.1:
        print(
            f"\n  Note: kernel GPU time improved {kernel_region_sx:.2f}x but process wall stayed "
            f"{wall_sx:.2f}x —\n        host overhead/CUDA init dominates this binary, not the kernel."
        )

    psx = result.get("primary_speedup_x")
    if psx is not None and basis:
        basis_label = {
            "cuda_event":      "kernel GPU time, cudaEvent",
            "kernel_gpu_time": "kernel GPU time, NCU",
            "wall_clock":      "wall-clock",
        }.get(basis, basis)
        print(f"\nSpeedup: {psx:.2f}x  (basis: {basis_label})")
    elif wall_sx is not None:
        print(f"\nSpeedup: {wall_sx:.2f}x  (basis: wall-clock)")

    if ncu_diff:
        bd = ncu_diff["bottleneck_diff"]
        baseline_bottlenecks  = ncu_diff["baseline"].get("bottlenecks", [])
        optimized_bottlenecks = ncu_diff["optimized"].get("bottlenecks", [])
        print("\nBottleneck changes (NCU):")

        for label in bd.get("resolved", []):
            old_score = next((b["score"] for b in baseline_bottlenecks if b["label"] == label), None)
            score_str = f"  (was {old_score:.2f})" if old_score is not None else ""
            print(f"  RESOLVED:  {label}{score_str}")

        new_labels = bd.get("new", [])
        for label in new_labels:
            new_score = next((b["score"] for b in optimized_bottlenecks if b["label"] == label), None)
            score_str = f"  (score {new_score:.2f})" if new_score is not None else ""
            print(f"! NEW BOTTLENECK:  {label}{score_str}")

        for label in bd.get("persistent", []):
            bs = next((b["score"] for b in baseline_bottlenecks  if b["label"] == label), None)
            as_ = next((b["score"] for b in optimized_bottlenecks if b["label"] == label), None)
            if bs is not None and as_ is not None:
                print(f"  UNCHANGED: {label}  ({bs:.2f} -> {as_:.2f})")

        if new_labels:
            print()
        else:
            print("\nNo new bottlenecks introduced.")

        verdict  = ncu_diff.get("verdict", {})
        outcome  = verdict.get("outcome", "")
        basis    = verdict.get("basis", "")
        n_res    = verdict.get("bottlenecks_resolved", 0)
        n_new    = verdict.get("bottlenecks_new", 0)
        basis_note = f", basis: {basis.replace('_', ' ')}" if basis else ""
        print(f"\nVerdict: {outcome}  ({n_res} resolved, {n_new} new{basis_note})")
        bottleneck_outcome = verdict.get("bottleneck_outcome")
        if bottleneck_outcome and basis == "kernel_gpu_time" and bottleneck_outcome != outcome:
            print(f"  (bottleneck diff alone would read: {bottleneck_outcome})")

    print()


def _resolve_case_study(target: str, root: str):
    """Resolve a case-study target to a loaded CaseStudy.

    *target* may be a path to a directory containing case_study.yaml, or a bare
    case-study name to look up by name under *root*.
    """
    from .case_study import discover_case_studies, load_case_study

    candidate = Path(target)
    if (candidate / "case_study.yaml").exists():
        return load_case_study(candidate)

    for case in discover_case_studies(root):
        if case.name == target or case.case_dir.name == target:
            return case

    raise FileNotFoundError(
        f"case study '{target}' not found (looked for a dir with case_study.yaml, "
        f"then by name under {root})"
    )


def case_study_cmd(args: argparse.Namespace) -> int:
    from .case_study import discover_case_studies, emit_case_study_artifacts, run_case_study, render_case_study_txt

    if args.cs_action == "list":
        cases = discover_case_studies(args.root)
        if not cases:
            print(f"No case studies found under {args.root}")
            return 1
        width = max(len(c.name) for c in cases)
        for c in cases:
            print(f"  {c.name:<{width}}  {c.title}  [{c.category}]")
        return 0

    if args.cs_action != "run":
        print("usage: frx case-study {run,list} ...", file=sys.stderr)
        return 1

    case = _resolve_case_study(args.target, args.root)
    result = run_case_study(case)

    if args.output_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(render_case_study_txt(result))

    if not args.no_artifacts:
        out_dir = Path(args.out) if args.out else Path("artifacts/case_studies") / case.name
        written = emit_case_study_artifacts(result, out_dir, emit_readme=args.emit_readme)
        print(f"\nArtifacts written to {out_dir}/")
        for name in sorted(written):
            print(f"  - {name}")

    return 0 if result["validation"]["passed"] else 1


def _fmt_pct(metric: dict | None) -> str:
    """Render a {value, n} leaderboard metric as 'NN% (n=K)' or 'n/a'."""
    if not metric or metric.get("value") is None:
        return "n/a"
    n = metric.get("n")
    suffix = f" (n={n})" if n is not None else ""
    return f"{metric['value'] * 100:.1f}%{suffix}"


def eval_cmd(args: argparse.Namespace) -> int:
    if args.eval_action != "sakana":
        print("usage: frx eval sakana [source] [--sample N] [--level {1,2,3}] [--out PATH]", file=sys.stderr)
        return 1

    from . import eval_sakana as ev

    environment = {"gpu_model": args.gpu_model} if args.gpu_model else None
    try:
        result = ev.run_eval(
            source=args.source,
            sample=args.sample,
            level=args.level,
            seed=args.seed,
            environment=environment,
            use_gold=not args.no_gold,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    lb = result["leaderboard"]
    cwr = lb["correctness_warning_recall"]
    sp = lb["speedup_alignment"]

    print(f"frx eval sakana - {lb['rows_evaluated']} kernels from {result['dataset']}")
    print("(we test Fournex, not Sakana; no GPU needed)\n")
    ceiling = lb["confidence_ceiling_respected"]
    max_emitted = lb["max_confidence_emitted"]["value"] or "none"
    print("OBJECTIVE")
    print(f"  concrete diagnosis coverage   : {_fmt_pct(lb['coverage_concrete_diagnosis'])}")
    print(f"  confidence ceiling respected  : {_fmt_pct(ceiling)}  (cap = {ceiling['ceiling']}; no warp-stall data to justify more)")
    print(f"  max confidence emitted        : {max_emitted}")
    print(f"  slow kernels explained        : {_fmt_pct(sp['slow_kernels_explained'])}")
    print(f"  fast kernels not over-claimed : {_fmt_pct(sp['fast_kernels_not_overclaimed'])}")
    print("CORRECTNESS (never inferred from the profile)")
    print(f"  warned on build/runtime errors: {_fmt_pct(cwr['build_or_runtime_error'])}")
    print(f"  silent numerical mismatches   : {_fmt_pct(cwr['silent_numerical_mismatch'])}  (documented blind spot)")
    print(f"  warnings on correct kernels   : {_fmt_pct(cwr['warning_rate_on_correct_kernels'])}")
    print(f"SELF-CONSISTENCY (heuristic, not truth)")
    print(f"  agrees w/ NCU weak label      : {_fmt_pct(lb['self_consistency_vs_weak_label'])}")
    if "gold" in lb:
        g = lb["gold"]
        print(f"GOLD ({g['gold_rows_in_subset']}/{g['gold_rows_total']} hand-labeled rows present)")
        print(f"  primary bottleneck accuracy   : {_fmt_pct(g['primary_bottleneck_accuracy'])}")
        print(f"  confidence ceiling respected  : {_fmt_pct(g['confidence_ceiling_respected'])}")
        print(f"  correctness-warning accuracy  : {_fmt_pct(g['correctness_warning_accuracy'])}")
        if g["failures"]:
            print(f"  {len(g['failures'])} gold check failure(s):")
            for f in g["failures"][:10]:
                print(f"    - {f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"\nFull report written to {out_path}")

    gold_failed = bool(lb.get("gold", {}).get("failures"))
    return 1 if gold_failed else 0


def _build_parser() -> argparse.ArgumentParser:
    import sys
    _stem = Path(sys.argv[0]).stem.lower().replace(".exe", "")
    prog = "fournex" if "fournex" in _stem else "frx"
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="enable verbose (INFO) logging to stderr",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="enable debug logging to stderr",
    )
    subparsers = parser.add_subparsers(dest="command")

    collect_parser = subparsers.add_parser("collect", help="run a workload and package a run bundle")
    collect_parser.add_argument("--name", default=None, help="human-readable job name")
    collect_parser.add_argument("--out", default="runs", help="output directory for run folders")
    collect_parser.add_argument("--config", default=None, help="optional user run_config.yaml to merge")
    collect_parser.add_argument("--sample-interval-ms", type=int, default=1000)
    collect_parser.add_argument("--run-id", default=None)
    collect_parser.add_argument("--no-zip", action="store_true")
    collect_parser.add_argument(
        "--explain",
        action="store_true",
        default=False,
        help="after collection, automatically generate the LLM optimization brief",
    )
    collect_parser.add_argument(
        "--explain-out",
        dest="explain_out",
        default=None,
        metavar="DIR",
        help="directory for LLM brief files when --explain is set (default: current directory)",
    )
    collect_parser.add_argument(
        "--artifact-dir",
        action="append",
        dest="artifact_dirs",
        metavar="DIR",
        default=None,
        help="directory to import workload artifacts from (default: frx-job-run); may be repeated",
    )
    collect_parser.add_argument(
        "--no-profiler-import",
        action="store_true",
        help="skip importing profiler_trace.json from the artifact directory",
    )
    collect_parser.add_argument("workload_command", nargs=argparse.REMAINDER)

    analyze_parser = subparsers.add_parser("analyze", help="analyze a run bundle, CUDA file, PTX file, NCU CSV, or before/after pair")
    analyze_parser.add_argument(
        "run_path",
        nargs="?",
        default=None,
        help="path to a run directory, .zip bundle, .ptx, .cu/.cuh, or Nsight Compute .csv",
    )
    analyze_parser.add_argument(
        "--baseline",
        default=None,
        metavar="CSV",
        help="Nsight Compute CSV for baseline — enables before/after comparison mode",
    )
    analyze_parser.add_argument(
        "--optimized",
        default=None,
        metavar="CSV",
        help="Nsight Compute CSV for optimized run — pair with --baseline",
    )
    analyze_parser.add_argument("--before", default=None, metavar="FILE", help="baseline file for before/after comparison")
    analyze_parser.add_argument("--after", default=None, metavar="FILE", help="optimized file for before/after comparison")
    analyze_parser.add_argument("--before-label", default=None, help="label for the baseline side")
    analyze_parser.add_argument("--after-label", default=None, help="label for the optimized side")
    analyze_parser.add_argument("--before-source", default=None, metavar="CU", help="baseline CUDA source file")
    analyze_parser.add_argument("--after-source", default=None, metavar="CU", help="optimized CUDA source file")
    analyze_parser.add_argument("--before-ptx", default=None, metavar="PTX", help="baseline PTX file")
    analyze_parser.add_argument("--after-ptx", default=None, metavar="PTX", help="optimized PTX file")
    analyze_parser.add_argument("--before-ncu", default=None, metavar="CSV", help="baseline Nsight Compute CSV file")
    analyze_parser.add_argument("--after-ncu", default=None, metavar="CSV", help="optimized Nsight Compute CSV file")
    analyze_parser.add_argument("--gpu-model", default=None, help="optional GPU model used by CUDA source launch advisor")
    analyze_parser.add_argument("--arch-profile", default=None, metavar="YAML", help="YAML hardware-spec overrides for roofline analysis")
    analyze_parser.add_argument(
        "--scope",
        choices=["run", "steady_state", "auto"],
        default="auto",
        help="which analysis scope to report (default: steady_state when available)",
    )
    analyze_parser.add_argument("--json", dest="output_json", action="store_true", help="output raw JSON")

    init_parser = subparsers.add_parser(
        "init",
        help="guided setup: check environment and show how to instrument your training loop",
    )
    init_parser.add_argument(
        "--patch",
        metavar="FILE",
        default=None,
        help="training script to patch with SDK instrumentation (asks for confirmation)",
    )

    subparsers.add_parser("doctor", help="check environment for frx requirements")

    subparsers.add_parser("smoke-test", help="run a synthetic workload and verify end-to-end bundle generation")

    ncu_parser = subparsers.add_parser("ncu-command", help="print an Nsight Compute command for Fournex metric presets")
    ncu_parser.add_argument(
        "preset",
        nargs="?",
        default="full",
        choices=["memory", "tensor", "occupancy", "stalls", "full"],
        help="metric preset to collect (default: full)",
    )
    ncu_parser.add_argument("--list", dest="list_presets", action="store_true", help="list available presets")
    ncu_parser.add_argument("--output", "-o", default=None, help="CSV output path to append as shell redirection")
    ncu_parser.add_argument("--kernel-name", default=None, help="optional Nsight Compute --kernel-name filter")
    ncu_parser.add_argument("--target-processes", default="all", help="Nsight Compute target process mode (default: all)")
    ncu_parser.add_argument("--launch-skip", type=int, default=None, help="skip the first N kernel launches")
    ncu_parser.add_argument("--launch-count", type=int, default=None, help="profile at most N kernel launches")
    ncu_parser.add_argument("--json", dest="output_json", action="store_true", help="output command and metrics as JSON")
    ncu_parser.add_argument("workload_command", nargs=argparse.REMAINDER)

    profile_parser = subparsers.add_parser(
        "profile",
        help="run NCU and print a full detailed bottleneck + recommendation report",
    )
    profile_parser.add_argument(
        "--ncu",
        default=None,
        metavar="CSV",
        help="analyze an existing Nsight Compute CSV instead of running ncu",
    )
    profile_parser.add_argument(
        "--ptx",
        default=None,
        metavar="PTX",
        help="analyze a PTX file instead of running ncu",
    )
    profile_parser.add_argument(
        "--preset",
        default="full",
        choices=["memory", "tensor", "occupancy", "stalls", "full"],
        help="metric preset to collect when running ncu (default: full)",
    )
    profile_parser.add_argument("--out", default=None, metavar="FILE", help="save captured NCU CSV to this path")
    profile_parser.add_argument("--kernel-name", default=None, help="Nsight Compute --kernel-name filter")
    profile_parser.add_argument("--launch-skip", type=int, default=None, help="skip the first N kernel launches")
    profile_parser.add_argument("--launch-count", type=int, default=None, help="profile at most N kernel launches")
    profile_parser.add_argument("--gpu-model", default=None, help="GPU model hint for launch advisor")
    profile_parser.add_argument("--arch-profile", default=None, metavar="YAML", help="YAML hardware-spec overrides for roofline analysis")
    profile_parser.add_argument("--json", dest="output_json", action="store_true", help="output raw JSON")
    profile_parser.add_argument(
        "--explain",
        action="store_true",
        default=False,
        help="after profiling, also write frx_summary.txt / frx_llm_prompt.txt / frx_evidence.json",
    )
    profile_parser.add_argument(
        "--explain-out",
        dest="explain_out",
        default=None,
        metavar="DIR",
        help="directory for LLM brief files when --explain is set (default: current directory)",
    )
    profile_parser.add_argument("workload_command", nargs=argparse.REMAINDER)

    tune_parser = subparsers.add_parser(
        "tune",
        help="run safe autopilot: sweep configs and recommend the fastest one",
    )
    tune_parser.add_argument("--name", default="frx-tune", help="job name")
    tune_parser.add_argument("--out", default="runs", help="output directory")
    tune_parser.add_argument(
        "--resume",
        default=None,
        metavar="TUNE_DIR",
        help="resume a tune directory and reuse matching trial artifacts",
    )
    tune_parser.add_argument("--max-trials", type=int, default=12, help="maximum candidate configs to try (default: 12)")
    tune_parser.add_argument("--safe", action="store_true", default=True, help="only Tier-0 safe actions (default)")
    tune_parser.add_argument("--no-safe", dest="safe", action="store_false", help="also run Tier-1 actions: batch size and AMP")
    tune_parser.add_argument("--time-budget-s", type=int, default=60, help="max seconds per trial (default: 60)")
    tune_parser.add_argument("--warmup-steps", type=int, default=5, help="steps to skip before measuring (default: 5)")
    tune_parser.add_argument("--measure-steps", type=int, default=20, help="steps to measure per trial (default: 20)")
    tune_parser.add_argument("--repeat-count", type=int, default=1, help="benchmark repeats per trial (default: 1)")
    tune_parser.add_argument("--no-race", action="store_true", help="disable quick candidate screening")
    tune_parser.add_argument("--race-promote-count", type=int, default=3, help="candidates to promote from quick screen (default: 3)")
    tune_parser.add_argument("--race-warmup-steps", type=int, default=1, help="warmup steps for quick screen (default: 1)")
    tune_parser.add_argument("--race-measure-steps", type=int, default=5, help="measure steps for quick screen (default: 5)")
    tune_parser.add_argument("--min-speedup", type=float, default=0.08, help="min throughput improvement to recommend (default: 0.08)")
    tune_parser.add_argument("--bottleneck", default=None, help="focus candidates on a specific bottleneck, e.g. input_bound")
    tune_parser.add_argument("--allow-risky-actions", action="store_true", help="allow high-risk candidates")
    tune_parser.add_argument("--no-quality-checks", dest="require_quality_checks", action="store_false", default=True)
    tune_parser.add_argument("--max-final-loss-regression", type=float, default=0.05)
    tune_parser.add_argument("--max-loss-divergence", type=float, default=0.50)
    tune_parser.add_argument("--output-abs-tolerance", type=float, default=0.005)
    tune_parser.add_argument("--allow-nonfinite-loss", dest="require_finite_loss", action="store_false", default=True)
    tune_parser.add_argument("--sample-interval-ms", type=int, default=1000)
    tune_parser.add_argument("workload_command", nargs=argparse.REMAINDER)

    compare_parser = subparsers.add_parser(
        "compare",
        help="compare CUDA source files or evidence files and report what improved, regressed, and is still unknown",
    )
    # Source-comparison mode: two positional .cu files (optional so --before/--after can be used alone)
    compare_parser.add_argument("file_a", nargs="?", default=None, help="baseline CUDA source file (.cu)")
    compare_parser.add_argument("file_b", nargs="?", default=None, help="optimized CUDA source file (.cu)")
    compare_parser.add_argument(
        "--with-ptx",
        action="store_true",
        help="compile both files with nvcc and include PTX analysis",
    )
    compare_parser.add_argument(
        "--with-ncu",
        action="store_true",
        help="compile both files to executables and run Nsight Compute (requires nvcc + ncu)",
    )
    compare_parser.add_argument(
        "--ncu-a",
        default=None,
        metavar="CSV",
        help="pre-existing NCU CSV for file A (alternative to --with-ncu)",
    )
    compare_parser.add_argument(
        "--ncu-b",
        default=None,
        metavar="CSV",
        help="pre-existing NCU CSV for file B (alternative to --with-ncu)",
    )
    compare_parser.add_argument("--label-a", default=None, help="display label for file A (default: filename)")
    compare_parser.add_argument("--label-b", default=None, help="display label for file B (default: filename)")
    compare_parser.add_argument("--gpu-model", default=None, help="GPU model hint for launch advisor")
    compare_parser.add_argument("--arch-profile", default=None, metavar="YAML", help="YAML hardware-spec overrides for roofline analysis")
    compare_parser.add_argument(
        "--build-flags",
        default="",
        metavar="FLAGS",
        help="extra nvcc flags for --with-ncu compilation (e.g. \"-DBUILD_EXECUTABLE\")",
    )
    compare_parser.add_argument("--json", dest="output_json", action="store_true", help="output raw JSON")

    # Evidence-comparison mode: --before/--after (type auto-detected from extension)
    compare_parser.add_argument("--before", default=None, metavar="FILE",
        help="baseline evidence file for before/after comparison (.cu, .ptx, or .csv — type auto-detected)")
    compare_parser.add_argument("--after", default=None, metavar="FILE",
        help="optimized evidence file for before/after comparison (.cu, .ptx, or .csv)")
    compare_parser.add_argument("--before-label", default=None, help="display label for the baseline side")
    compare_parser.add_argument("--after-label", default=None, help="display label for the optimized side")
    compare_parser.add_argument("--before-source", default=None, metavar="CU", help="baseline CUDA source file")
    compare_parser.add_argument("--after-source", default=None, metavar="CU", help="optimized CUDA source file")
    compare_parser.add_argument("--before-ptx", default=None, metavar="PTX", help="baseline PTX file")
    compare_parser.add_argument("--after-ptx", default=None, metavar="PTX", help="optimized PTX file")
    compare_parser.add_argument("--before-ncu", default=None, metavar="CSV", help="baseline Nsight Compute CSV")
    compare_parser.add_argument("--after-ncu", default=None, metavar="CSV", help="optimized Nsight Compute CSV")
    # Hidden aliases for _analyze_comparison() compatibility
    compare_parser.add_argument("--baseline", default=None, help=argparse.SUPPRESS)
    compare_parser.add_argument("--optimized", default=None, help=argparse.SUPPRESS)

    explain_parser = subparsers.add_parser(
        "explain",
        help="generate LLM-ready optimization brief from NCU CSV or a training run directory",
    )
    explain_parser.add_argument(
        "ncu_path",
        metavar="PATH",
        nargs="?",
        default=None,
        help="NCU CSV file  OR  frx run directory (auto-detected by type)",
    )
    explain_parser.add_argument(
        "--scope",
        choices=["run", "steady_state", "auto"],
        default="auto",
        help="which analysis scope to use for run directories (default: steady_state when available)",
    )
    explain_parser.add_argument(
        "--src",
        metavar="CU",
        dest="src_path",
        default=None,
        help="CUDA source file for static analysis layer",
    )
    explain_parser.add_argument(
        "--out",
        metavar="DIR",
        default=".",
        help="output directory for generated files (default: current directory)",
    )
    explain_parser.add_argument(
        "--gpu-model",
        default=None,
        metavar="MODEL",
        help="GPU model for arch-aware scoring (e.g. h100, a100)",
    )
    explain_parser.add_argument(
        "--arch-profile",
        default=None,
        metavar="YAML",
        help="YAML hardware-spec overrides for roofline analysis",
    )
    explain_parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="print frx_llm_prompt.txt to stdout only (for piping/clipboard)",
    )

    cv_parser = subparsers.add_parser(
        "compare-variants",
        help="rank multiple kernel variants by measured throughput + explain NCU deltas",
    )
    cv_parser.add_argument(
        "results_csv",
        metavar="RESULTS_CSV",
        help="CSV manifest with columns: variant, ncu_csv, throughput_gflops, notes",
    )
    cv_parser.add_argument(
        "--baseline",
        default=None,
        metavar="VARIANT",
        help="name of the reference variant (default: lowest throughput)",
    )
    cv_parser.add_argument("--gpu-model", default=None, help="GPU model hint for arch-aware scoring")
    cv_parser.add_argument("--arch-profile", default=None, metavar="YAML", help="YAML hardware-spec overrides for roofline analysis")
    cv_parser.add_argument("--json", dest="output_json", action="store_true", help="output raw JSON")
    cv_parser.add_argument("--explain", action="store_true", default=False, help="write frx_llm_prompt.txt for the best variant")
    cv_parser.add_argument("--explain-out", dest="explain_out", default=None, metavar="DIR", help="directory for LLM brief when --explain is set")

    bench_parser = subparsers.add_parser(
        "bench",
        help="compile and benchmark two .cu kernels side-by-side",
    )
    bench_parser.add_argument("before", nargs="?", default=None, help="baseline .cu source file")
    bench_parser.add_argument("after", nargs="?", default=None, help="optimized .cu source file")
    bench_parser.add_argument(
        "--emit-harness",
        nargs="?",
        const="frx_bench_harness.cuh",
        default=None,
        metavar="PATH",
        help="write the cudaEvent timing harness header to PATH (default: ./frx_bench_harness.cuh) and exit",
    )
    bench_parser.add_argument("--warmup", type=int, default=2, help="warmup runs to discard (default: 2)")
    bench_parser.add_argument("--runs", type=int, default=5, help="timed runs (default: 5)")
    bench_parser.add_argument("--with-ncu", action="store_true", help="also profile with NCU and report bottleneck changes")
    bench_parser.add_argument("--arch", default=None, metavar="SM", help="nvcc -arch flag, e.g. sm_120 (required for --with-ncu)")
    bench_parser.add_argument("--build-flags", default="-DBUILD_EXECUTABLE", metavar="FLAGS", help="extra nvcc flags (default: -DBUILD_EXECUTABLE)")
    bench_parser.add_argument("--out", default=None, metavar="DIR", help="directory for compiled exes and NCU CSVs")
    bench_parser.add_argument("--json", dest="output_json", action="store_true", help="output raw JSON")

    cs_parser = subparsers.add_parser(
        "case-study",
        help="run a bad/good kernel case study and emit a validated proof bundle",
    )
    cs_sub = cs_parser.add_subparsers(dest="cs_action")
    cs_run = cs_sub.add_parser("run", help="run one case study (name under --root, or a path to a case dir)")
    cs_run.add_argument("target", help="case-study name or path to a directory containing case_study.yaml")
    cs_run.add_argument("--root", default=DEFAULT_CASE_STUDY_ROOT, metavar="DIR", help="search root for case-study names")
    cs_run.add_argument("--out", default=None, metavar="DIR", help="artifact output directory (default: artifacts/case_studies/<name>)")
    cs_run.add_argument("--emit-readme", action="store_true", help="also write README.md to the artifact bundle")
    cs_run.add_argument("--no-artifacts", action="store_true", help="print the transcript only; do not write files")
    cs_run.add_argument("--json", dest="output_json", action="store_true", help="output raw JSON result")
    cs_list = cs_sub.add_parser("list", help="list discoverable case studies")
    cs_list.add_argument("--root", default=DEFAULT_CASE_STUDY_ROOT, metavar="DIR", help="search root for case studies")

    eval_parser = subparsers.add_parser(
        "eval",
        help="evaluate Fournex's analyzer against a labeled dataset",
    )
    eval_sub = eval_parser.add_subparsers(dest="eval_action")
    sakana = eval_sub.add_parser(
        "sakana",
        help="evaluate against the SakanaAI/AI-CUDA-Engineer-Archive (offline cached subset by default)",
    )
    sakana.add_argument("source", nargs="?", default=None,
                        help="path to a JSONL subset (default: packaged cached fixture)")
    sakana.add_argument("--sample", type=int, default=None, metavar="N", help="evaluate a deterministic sample of N rows")
    sakana.add_argument("--level", type=int, choices=(1, 2, 3), default=None, help="restrict to a KernelBench level")
    sakana.add_argument("--seed", type=int, default=0, help="sampling seed (for reproducibility)")
    sakana.add_argument("--gpu-model", default=None, metavar="NAME", help="GPU model for arch-aware analysis (e.g. h100)")
    sakana.add_argument("--out", default=None, metavar="PATH", help="write the full sakana_eval_v1 JSON report here")
    sakana.add_argument("--no-gold", action="store_true", help="skip scoring against the hand-labeled gold set")

    return parser


def _normalize_workload_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def _resolve_workload_command(command: list[str]) -> list[str]:
    if command and command[0].lower() in {"python", "python.exe", "python3"}:
        return [sys.executable, *command[1:]]
    return command


def _build_run_config(
    args: argparse.Namespace,
    run_id: str,
    job_name: str,
    run_dir: Path,
) -> dict[str, Any]:
    user_config = _read_simple_yaml(Path(args.config)) if args.config else {}
    config = {
        "collector": {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "command": "frx collect",
            "sample_interval_ms": args.sample_interval_ms,
            "output_dir": str(run_dir),
        },
        "run": {
            "run_id": run_id,
            "job_name": job_name,
            "workload_command": list(args.workload_command),
        },
        "environment": _detect_environment(),
    }
    return _deep_merge(config, user_config)


def _detect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
    }
    try:
        import torch  # type: ignore

        env["framework"] = "pytorch"
        env["pytorch_version"] = getattr(torch, "__version__", "unknown")
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["num_gpus"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env["gpu_name"] = gpu_name
        if gpu_name and "gpu_model" not in env:
            from .arch_profiles import detect_gpu_model
            detected = detect_gpu_model(gpu_name)
            if detected:
                env["gpu_model"] = detected
                env["gpu_type"] = detected
    except Exception as exc:
        env["framework"] = "unknown"
        env["framework_detection_error"] = str(exc)
    return env


def _environment_from_args(args: argparse.Namespace) -> dict[str, Any]:
    env = _detect_environment()
    gpu_model = getattr(args, "gpu_model", None)
    if gpu_model:
        env["gpu_model"] = gpu_model
        env["gpu_type"] = gpu_model
    arch_profile = getattr(args, "arch_profile", None)
    if arch_profile:
        from .arch_profiles import load_arch_profile_overrides

        env["arch_profile_overrides"] = load_arch_profile_overrides(arch_profile)
    return env


def ncu_command(args: argparse.Namespace) -> int:
    from .ncu_presets import build_ncu_command, describe_ncu_presets, format_shell_command, get_ncu_preset

    _extract_ncu_remainder_options(args)
    args.workload_command = _resolve_workload_command(_normalize_workload_command(args.workload_command))

    if args.list_presets:
        presets = describe_ncu_presets()
        if args.output_json:
            _print_json_result("ncu_presets", {"presets": presets})
        else:
            print("\nFournex NCU metric presets\n")
            for preset in presets:
                print(f"  {preset['name']:<10} {preset['description']}")
                print(f"             metrics: {len(preset['metrics'])}")
            print()
        return 0

    workload = args.workload_command or ["./your_app"]
    try:
        command = build_ncu_command(
            args.preset,
            workload,
            output=args.output,
            kernel_name=args.kernel_name,
            target_processes=args.target_processes,
            launch_skip=args.launch_skip,
            launch_count=args.launch_count,
        )
        preset = get_ncu_preset(args.preset)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        _print_json_result(
            "ncu_command",
            {
                "preset": preset.name,
                "description": preset.description,
                "metrics": list(preset.metrics),
                "command": command,
                "shell": format_shell_command(command),
            },
        )
        return 0

    print("\nFournex NCU command\n")
    print(f"Preset : {preset.name}")
    print(f"Purpose: {preset.description}")
    print("\nCommand:")
    print(f"  {format_shell_command(command)}")
    print("\nMetrics:")
    for metric in preset.metrics:
        print(f"  - {metric}")
    print()
    return 0


def profile(args: argparse.Namespace) -> int:
    """Run NCU on a workload (or analyze existing data) and print a full detailed report."""
    import fournex as at
    from .ncu_presets import build_ncu_command, format_shell_command

    environment = _environment_from_args(args)

    ncu_csv_text: str | None = None
    source_label: str = "live profiling"

    if args.ncu:
        ncu_path = Path(args.ncu)
        if not ncu_path.exists():
            print(f"Error: NCU CSV not found: {ncu_path}", file=sys.stderr)
            return 1
        ncu_csv_text = ncu_path.read_text()
        source_label = str(ncu_path)

    elif args.ptx:
        ptx_path = Path(args.ptx)
        if not ptx_path.exists():
            print(f"Error: PTX file not found: {ptx_path}", file=sys.stderr)
            return 1
        result = at.analyze_ptx_text(ptx_path.read_text())
        if args.output_json:
            _print_json_result("profile", result)
        else:
            _print_ptx_report(result)
        return 0

    else:
        workload = args.workload_command
        if not workload:
            print(
                "Error: provide a workload command (-- python train.py), --ncu FILE, or --ptx FILE",
                file=sys.stderr,
            )
            return 1

        ncu_bin = shutil.which("ncu")
        if not ncu_bin:
            print("Error: 'ncu' not found on PATH.", file=sys.stderr)
            print("  Install Nsight Compute, or use the manual two-step workflow:", file=sys.stderr)
            print(f"    frx ncu-command --preset {args.preset} -- {' '.join(workload)}", file=sys.stderr)
            print("    frx profile --ncu <output.csv>", file=sys.stderr)
            return 1

        from .arch_profiles import resolve_sm_version
        from .ncu_presets import get_ncu_preset, pc_sampling_supported
        sm_version = resolve_sm_version(environment.get("gpu_model") if environment else None)

        try:
            command = build_ncu_command(
                args.preset,
                workload,
                output=None,
                kernel_name=args.kernel_name,
                launch_skip=args.launch_skip,
                launch_count=args.launch_count,
                sm_version=sm_version,
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        if not pc_sampling_supported(sm_version) and any(
            m.startswith("smsp__pcsamplingdata_") for m in get_ncu_preset(args.preset).metrics
        ):
            print(
                f"Note: PC-sampling stall metrics are unavailable on {sm_version} (Blackwell); "
                "they were dropped so the rest of the preset can be collected.",
                file=sys.stderr,
            )

        print(f"\nProfiling: {format_shell_command(command)}\n")
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Error: ncu exited with code {proc.returncode}", file=sys.stderr)
            if proc.stderr:
                print(proc.stderr[:600], file=sys.stderr)
            return 1

        ncu_csv_text = proc.stdout
        source_label = f"ncu {format_shell_command(workload)}"

        if args.out:
            out_path = Path(args.out)
            out_path.write_text(ncu_csv_text)
            print(f"NCU CSV saved → {out_path}")

    result = at.analyze_ncu_csv_text(ncu_csv_text, environment=environment)

    if args.output_json:
        _print_json_result("profile", result)
        return 0

    _print_ncu_report_full(result, source=source_label, args=args)

    if getattr(args, "explain", False):
        from .explain import build_explain_result, render_summary_txt, render_llm_prompt_txt, render_evidence_json
        explain_result = build_explain_result(ncu_result=result, environment=environment)
        out_dir = Path(getattr(args, "explain_out", None) or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "frx_summary.txt").write_text(
            render_summary_txt(explain_result, ncu_filename=source_label), encoding="utf-8"
        )
        (out_dir / "frx_llm_prompt.txt").write_text(
            render_llm_prompt_txt(explain_result), encoding="utf-8"
        )
        (out_dir / "frx_evidence.json").write_text(render_evidence_json(explain_result), encoding="utf-8")
        print(f"\n  LLM brief written to {out_dir / 'frx_llm_prompt.txt'}")
        print("  Paste frx_llm_prompt.txt into your LLM for optimization suggestions.")

    return 0


def _extract_ncu_remainder_options(args: argparse.Namespace) -> None:
    command = list(args.workload_command or [])
    if "--" not in command:
        return

    separator = command.index("--")
    option_tokens = command[:separator]
    workload_tokens = command[separator + 1:]
    index = 0
    while index < len(option_tokens):
        token = option_tokens[index]
        value = option_tokens[index + 1] if index + 1 < len(option_tokens) else None
        if token in {"--output", "-o"} and value is not None:
            args.output = value
            index += 2
        elif token == "--kernel-name" and value is not None:
            args.kernel_name = value
            index += 2
        elif token == "--target-processes" and value is not None:
            args.target_processes = value
            index += 2
        elif token == "--launch-skip" and value is not None:
            args.launch_skip = int(value)
            index += 2
        elif token == "--launch-count" and value is not None:
            args.launch_count = int(value)
            index += 2
        else:
            workload_tokens = option_tokens[index:] + ["--"] + workload_tokens
            break
    args.workload_command = workload_tokens


def _sample_gpu_metrics(
    output_path: Path,
    interval_ms: int,
    stop: Event,
    warnings: list[str],
) -> None:
    fields = [
        "timestamp",
        "index",
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "pcie.link.gen.current",
        "pcie.link.width.current",
    ]
    query = ",".join(fields[1:])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nvidia_smi = shutil.which("nvidia-smi")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        if nvidia_smi is None:
            warnings.append("nvidia-smi was not found; gpu_metrics.csv contains headers only.")
            return

        base_wait_s = max(interval_ms, 100) / 1000.0
        consecutive_failures = 0
        max_consecutive = 5
        while not stop.is_set():
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                result = subprocess.run(
                    [
                        nvidia_smi,
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in result.stdout.splitlines():
                    if line.strip():
                        writer.writerow([timestamp, *[part.strip() for part in line.split(",")]])
                handle.flush()
                consecutive_failures = 0
            except Exception as exc:
                # A transient nvidia-smi failure (timeout, momentary busy/lock)
                # must not kill sampling for the rest of the run. Keep going,
                # cap the warning noise, and back off once failures persist.
                consecutive_failures += 1
                if consecutive_failures in (1, max_consecutive):
                    warnings.append(
                        f"nvidia-smi sampling failed ({consecutive_failures}x), continuing: {exc}"
                    )
            wait_s = base_wait_s
            if consecutive_failures >= max_consecutive:
                wait_s = max(base_wait_s, 5.0)  # degrade frequency while failing
            stop.wait(wait_s)


def _discover_artifacts(run_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    candidates = {
        "metadata": run_dir / "metadata.json",
        "run_config": run_dir / "run_config.yaml",
        "gpu_metrics": run_dir / "gpu_metrics.csv",
        "logs": run_dir / "optional_logs.txt",
        "raw_trace": run_dir / "raw" / "trace.jsonl",
        "derived_summary": run_dir / "derived" / "summary.json",
    }
    for key, path in candidates.items():
        if path.exists():
            artifacts[key] = path.relative_to(run_dir).as_posix()

    profiler_files = sorted((run_dir / "profiler").glob("*.json"))
    if profiler_files:
        artifacts["profiler_trace"] = profiler_files[0].relative_to(run_dir).as_posix()

    return artifacts


def _import_workload_bundle_artifacts(
    run_dir: Path,
    source_dirs: list[Path],
    warnings: list[str],
    import_profiler: bool = True,
) -> list[str]:
    imported: list[str] = []

    for source_dir in source_dirs:
        if not source_dir.exists() or not source_dir.is_dir():
            continue
        mappings: dict[str, Path] = {
            "trace.jsonl": run_dir / "raw" / "trace.jsonl",
            "summary.json": run_dir / "derived" / "summary.json",
            "metadata.json": run_dir / "raw" / "workload_metadata.json",
            "run_config.yaml": run_dir / "workload_run_config.yaml",
            "gpu_metrics.csv": run_dir / "gpu_metrics.csv",
        }
        if import_profiler:
            mappings["profiler_trace.json"] = run_dir / "profiler" / "profiler_trace.json"

        for source_name, destination in mappings.items():
            source = source_dir / source_name
            if not source.exists() or not source.is_file():
                continue
            if destination.exists() and source_name != "gpu_metrics.csv":
                continue
            if source_name == "gpu_metrics.csv" and destination.exists() and source.stat().st_size <= destination.stat().st_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            imported.append(destination.relative_to(run_dir).as_posix())

    return imported


def _append_limited_bundle_warnings(artifacts: dict[str, str], warnings: list[str]) -> None:
    has_raw_trace = "raw_trace" in artifacts
    has_profiler_trace = "profiler_trace" in artifacts

    if not has_raw_trace and not has_profiler_trace:
        warnings.append(
            "No SDK trace or profiler trace was captured; this bundle supports limited diagnosis only. "
            "Instrument the workload with fournex or export a PyTorch profiler trace for richer results."
        )
    elif has_profiler_trace and not has_raw_trace:
        warnings.append(
            "Profiler trace captured; SDK raw trace was not captured. Profiler-based diagnosis is available."
        )

    if "derived_summary" not in artifacts and not has_profiler_trace:
        warnings.append(
            "No derived summary was generated; recommendations may be less precise until trace events are captured."
        )


def _build_metadata(
    *,
    args: argparse.Namespace,
    run_id: str,
    job_name: str,
    run_dir: Path,
    started_at: datetime,
    ended_at: datetime,
    duration_s: float,
    exit_code: int,
    artifacts: dict[str, str],
    warnings: list[str],
) -> dict[str, Any]:
    env = _detect_environment()
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "run_id": run_id,
        "job_name": job_name,
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": duration_s,
        "workload_command": list(args.workload_command),
        "output_dir": str(run_dir),
        "host": env.get("host"),
        "platform": env.get("platform"),
        "python_version": env.get("python_version"),
        "framework": env.get("framework"),
        "pytorch_version": env.get("pytorch_version"),
        "cuda_available": env.get("cuda_available"),
        "num_gpus": env.get("num_gpus"),
        "gpu_name": env.get("gpu_name"),
        "artifacts": artifacts,
        "collection_warnings": list(dict.fromkeys(warnings)),
    }


def _build_manifest(run_dir: Path, artifacts: dict[str, str], warnings: list[str]) -> dict[str, Any]:
    included_files = [
        path.relative_to(run_dir).as_posix()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    ]
    if "manifest.json" not in included_files:
        included_files.append("manifest.json")
    required = {"metadata", "run_config", "gpu_metrics", "logs"}
    recommended = {"raw_trace", "derived_summary", "profiler_trace"}
    missing_required = sorted(required - set(artifacts))
    missing_recommended = sorted(recommended - set(artifacts))
    diagnostic_ready = bool({"raw_trace", "derived_summary", "profiler_trace"} & set(artifacts))
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "included_files": included_files,
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "diagnostic_ready": diagnostic_ready,
        "limited": bool(missing_required or not diagnostic_ready),
        "warnings": list(dict.fromkeys(warnings)),
    }


def _zip_run_dir(run_dir: Path, zip_path: Path) -> str:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}")
    return str(zip_path)


def _print_collection_summary(
    run_dir: Path,
    zip_path: Path | None,
    manifest: dict[str, Any],
    exit_code: int,
    imported: list[str] | None = None,
) -> None:
    status = "completed" if exit_code == 0 else f"failed with exit code {exit_code}"
    print(f"\nfrx collect {status}")
    print(f"Run bundle: {run_dir}")
    if zip_path is not None:
        print(f"Zip bundle: {zip_path}")

    included = manifest.get("included_files", [])
    if included:
        print(f"\nCaptured ({len(included)} files):")
        for f in included:
            tag = " [imported]" if (imported and f in imported) else ""
            print(f"  {f}{tag}")

    if manifest.get("limited"):
        missing = ", ".join(manifest.get("missing_recommended", []))
        print("\nLimited data detected.")
        if missing:
            print(f"Missing recommended artifacts: {missing}")
        print("Instrument the workload with fournex or pass --artifact-dir for richer results.")
    elif manifest.get("missing_recommended"):
        missing = ", ".join(manifest.get("missing_recommended", []))
        print(f"\nMissing optional artifacts: {missing}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(_to_yaml(payload), encoding="utf-8")


def _to_yaml(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_to_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]\n"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_to_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{_yaml_scalar(value)}\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    # V1 intentionally keeps user config support shallow: preserve the source
    # as a payload string rather than depending on a YAML parser for the CLI.
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return {"user_config": {"source_path": str(path), "raw": path.read_text(encoding="utf-8")}}


def _read_run_config_environment(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        return None
    try:
        config = _parse_generated_yaml(config_path.read_text(encoding="utf-8"))
    except OSError:
        return None
    environment = config.get("environment") if isinstance(config, dict) else None
    return environment if isinstance(environment, dict) else None


def _parse_generated_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset emitted by _to_yaml for run_config.yaml."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if stripped.startswith("-"):
            continue

        key, sep, raw_value = stripped.partition(":")
        if not sep:
            continue
        key = key.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1] if stack else root

        if raw_value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_generated_yaml_scalar(raw_value.strip())

    return root


def _parse_generated_yaml_scalar(value: str) -> Any:
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def analyze(args: argparse.Namespace) -> int:
    if _has_comparison_args(args):
        return _analyze_comparison(args)

    if not args.run_path:
        print("Error: provide a path to analyze or use --before/--after for comparison.", file=sys.stderr)
        return 1

    input_path = Path(args.run_path)
    if input_path.is_file() and input_path.suffix.lower() != ".zip":
        return _analyze_evidence_file(input_path, args)

    temp_context = None
    run_path = input_path
    if run_path.is_file() and run_path.suffix.lower() == ".zip":
        temp_context = tempfile.TemporaryDirectory(prefix="frx_analyze_")
        try:
            run_path = _extract_zip_run_bundle(run_path, Path(temp_context.name))
        except ValueError as exc:
            temp_context.cleanup()
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    if not run_path.is_dir():
        if temp_context is not None:
            temp_context.cleanup()
        print(f"Error: run directory not found: {run_path}", file=sys.stderr)
        return 1

    try:
        summary = _load_or_generate_summary(run_path)
        if summary is None:
            print("No trace data found in bundle. Cannot generate analysis.", file=sys.stderr)
            print(f"Expected: {run_path / 'raw' / 'trace.jsonl'} or {run_path / 'derived' / 'summary.json'}", file=sys.stderr)
            return 1

        if args.output_json:
            _print_json_result("run_bundle", summary)
            return 0

        _print_analysis_report(run_path, summary, scope=args.scope)
        return 0
    finally:
        if temp_context is not None:
            temp_context.cleanup()


_CUDA_SOURCE_SUFFIXES = {".cu", ".cuh", ".cuda"}
_COMPARISON_FIELDS = (
    "baseline",
    "optimized",
    "before",
    "after",
    "before_source",
    "after_source",
    "before_ptx",
    "after_ptx",
    "before_ncu",
    "after_ncu",
)


def _has_comparison_args(args: argparse.Namespace) -> bool:
    return any(getattr(args, name, None) for name in _COMPARISON_FIELDS)


def _analyze_evidence_file(path: Path, args: argparse.Namespace) -> int:
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    try:
        text = _read_text_file(path)
        kind = _detect_analysis_input_kind(path, text)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if kind == "ptx":
        from .ptx_analysis import analyze_ptx_text

        result = analyze_ptx_text(text, filename=str(path))
        if args.output_json:
            _print_json_result("ptx", result)
        else:
            _print_ptx_report(result)
        return 0

    if kind == "cuda_source":
        from .cuda_static import inspect_cuda_source

        result = inspect_cuda_source(text, filename=str(path), gpu_model=args.gpu_model)
        if args.output_json:
            _print_json_result("cuda_source", result)
        else:
            _print_cuda_source_report(result)
        return 0

    if kind == "ncu":
        from .ncu_analysis import analyze_ncu_csv_text

        result = analyze_ncu_csv_text(text, environment=_environment_from_args(args))
        validation = result.get("validation", {})
        if args.output_json:
            _print_json_result("ncu", result)
            if validation.get("errors"):
                return 1
        else:
            _print_ncu_report(result)
        return 1 if validation.get("errors") else 0

    print(f"Error: unsupported input file: {path}", file=sys.stderr)
    print("Supported inputs: run directories, .zip bundles, .ptx, .cu/.cuh, and Nsight Compute .csv files.", file=sys.stderr)
    return 1


def _analyze_comparison(args: argparse.Namespace) -> int:
    before_path = args.before or args.baseline
    after_path = args.after or args.optimized
    layer_specific = any(
        getattr(args, name, None)
        for name in ("before_source", "after_source", "before_ptx", "after_ptx", "before_ncu", "after_ncu")
    )

    if (before_path or after_path) and layer_specific:
        print("Error: use either --before/--after or layer-specific comparison flags, not both.", file=sys.stderr)
        return 1

    try:
        if before_path or after_path:
            if not before_path or not after_path:
                print("Error: --before and --after must be provided together.", file=sys.stderr)
                return 1
            return _analyze_auto_comparison(Path(before_path), Path(after_path), args)

        before_input = _build_layered_comparison_input(args, "before")
        after_input = _build_layered_comparison_input(args, "after")
        _validate_layered_comparison(before_input, after_input)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    from .comparison import compare_implementations

    result = compare_implementations(before_input, after_input)
    if args.output_json:
        _print_json_result("comparison", result)
    else:
        _print_cuda_comparison_report(result)
    return 0


def _analyze_auto_comparison(before_path: Path, after_path: Path, args: argparse.Namespace) -> int:
    try:
        before_text = _read_text_file(before_path)
        after_text = _read_text_file(after_path)
        before_kind = _detect_analysis_input_kind(before_path, before_text)
        after_kind = _detect_analysis_input_kind(after_path, after_text)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    before_label = args.before_label or before_path.stem
    after_label = args.after_label or after_path.stem

    if before_kind == "ncu" and after_kind == "ncu":
        from .ncu_comparison import diff_ncu_runs

        result = diff_ncu_runs(
            before_text,
            after_text,
            label_baseline=before_label,
            label_optimized=after_label,
            environment=_environment_from_args(args),
        )
        has_validation_errors = bool(result["baseline"].get("validation", {}).get("errors")) or bool(
            result["optimized"].get("validation", {}).get("errors")
        )
        if args.output_json:
            _print_json_result("comparison", result)
            if has_validation_errors:
                return 1
        else:
            _print_ncu_comparison_report(result)
        return 1 if has_validation_errors else 0

    try:
        before_input = _comparison_input_from_detected_file(before_path, before_kind, before_text, before_label, args)
        after_input = _comparison_input_from_detected_file(after_path, after_kind, after_text, after_label, args)
        _validate_layered_comparison(before_input, after_input)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    from .comparison import compare_implementations

    result = compare_implementations(before_input, after_input)
    if args.output_json:
        _print_json_result("comparison", result)
    else:
        _print_cuda_comparison_report(result)
    return 0


def _build_layered_comparison_input(args: argparse.Namespace, side: str) -> dict[str, Any]:
    label_attr = "before_label" if side == "before" else "after_label"
    payload: dict[str, Any] = {"label": getattr(args, label_attr) or side}

    source_path = getattr(args, f"{side}_source")
    ptx_path = getattr(args, f"{side}_ptx")
    ncu_path = getattr(args, f"{side}_ncu")

    if source_path:
        path = Path(source_path)
        payload["cuda_source"] = _read_text_file(path)
        payload["cuda_filename"] = str(path)
        payload["gpu_model"] = args.gpu_model
        if args.arch_profile:
            payload["arch_profile_overrides"] = _environment_from_args(args).get("arch_profile_overrides")
    if ptx_path:
        path = Path(ptx_path)
        payload["ptx"] = _read_text_file(path)
        payload["ptx_filename"] = str(path)
    if ncu_path:
        path = Path(ncu_path)
        payload["ncu_csv"] = _read_text_file(path)
        payload["gpu_model"] = args.gpu_model
        if args.arch_profile:
            payload["arch_profile_overrides"] = _environment_from_args(args).get("arch_profile_overrides")

    return payload


def _comparison_input_from_detected_file(
    path: Path,
    kind: str,
    text: str,
    label: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"label": label}
    if kind == "cuda_source":
        payload.update({"cuda_source": text, "cuda_filename": str(path), "gpu_model": args.gpu_model})
    elif kind == "ptx":
        payload.update({"ptx": text, "ptx_filename": str(path)})
    elif kind == "ncu":
        payload.update({"ncu_csv": text, "gpu_model": args.gpu_model})
    else:
        raise ValueError(f"unsupported comparison input: {path}")
    if args.arch_profile:
        payload["arch_profile_overrides"] = _environment_from_args(args).get("arch_profile_overrides")
    return payload


def _validate_layered_comparison(before_input: dict[str, Any], after_input: dict[str, Any]) -> None:
    evidence_fields = ("cuda_source", "ptx", "ncu_csv")
    before_layers = {field for field in evidence_fields if before_input.get(field)}
    after_layers = {field for field in evidence_fields if after_input.get(field)}
    if not before_layers or not after_layers:
        raise ValueError("comparison requires at least one input for both before and after.")
    if not (before_layers & after_layers):
        raise ValueError("before and after must share at least one evidence type.")


def _read_text_file(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"file not found: {path}")
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _detect_analysis_input_kind(path: Path, text: str) -> str:
    suffix = path.suffix.lower()
    if suffix == ".ptx":
        return "ptx"
    if suffix in _CUDA_SOURCE_SUFFIXES:
        return "cuda_source"
    if suffix == ".csv":
        return "ncu"

    lowered = text.lower()
    if ".entry" in text or ".version" in text and ".target" in text:
        return "ptx"
    if "__global__" in text or "<<<" in text:
        return "cuda_source"
    if "metric name" in lowered and "metric value" in lowered:
        return "ncu"
    raise ValueError(
        f"unsupported input file: {path}. Supported inputs are .ptx, .cu/.cuh, Nsight Compute .csv, run directories, and .zip bundles."
    )


def _print_json_result(mode: str, result: dict[str, Any]) -> None:
    print(json.dumps({"mode": mode, "result": result}, indent=2))


_SDK_SNIPPET = """\
import fournex as frx

frx.init(job_name="my-run")

for step, batch in enumerate(dataloader):
    with frx.step_context(step=step, batch=batch, model=model):
        # your existing training step here
        pass
"""

_TRAINING_SCRIPT_NAMES = ("train.py", "train.py", "main.py", "run.py", "finetune.py")


def _detect_training_script(cwd: Path) -> Path | None:
    """Return the first likely training script found in cwd, or None."""
    for name in _TRAINING_SCRIPT_NAMES:
        p = cwd / name
        if p.exists():
            return p
    for p in sorted(cwd.glob("train_*.py")):
        return p
    return None


def _already_instrumented(text: str) -> bool:
    return "fournex" in text or "frx.init" in text


def _patch_script(script_path: Path) -> bool:
    """Insert SDK snippet after the last top-level import line, with confirmation."""
    text = script_path.read_text(encoding="utf-8")
    if _already_instrumented(text):
        print(f"\n  {script_path.name} already contains fournex instrumentation — nothing to do.")
        return True

    lines = text.splitlines(keepends=True)
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            last_import_idx = i

    insert_at = last_import_idx + 1 if last_import_idx >= 0 else 0
    snippet_lines = ["\n", "import fournex as frx\n", "frx.init(job_name=\"my-run\")\n", "\n"]
    new_lines = lines[:insert_at] + snippet_lines + lines[insert_at:]

    print(f"\n  Will add these lines after line {insert_at} of {script_path.name}:")
    for line in snippet_lines:
        print(f"  + {line}", end="")
    print()

    answer = input("\n  Apply? [y/N] ").strip().lower()
    if answer == "y":
        script_path.write_text("".join(new_lines), encoding="utf-8")
        print(f"  Patched {script_path}")
        return True
    else:
        print("  Skipped.")
        return False


def init_cmd(args: argparse.Namespace) -> int:
    """Guided onboarding: environment check + SDK snippet + optional script patch."""
    cwd = Path.cwd()

    # ── Environment status ────────────────────────────────────────────────────
    print("\nfrx init\n")

    python_ok = True
    torch_ok = False
    cuda_ok = False
    ncu_ok = bool(shutil.which("ncu"))
    gpu_label = "not detected"

    print(f"  Python {platform.python_version():<10} OK")

    try:
        import torch  # type: ignore
        torch_ok = True
        tv = getattr(torch, "__version__", "?")
        print(f"  PyTorch {tv:<9} OK")
        if torch.cuda.is_available():
            cuda_ok = True
            gpu_name = torch.cuda.get_device_name(0)
            from .arch_profiles import detect_gpu_model
            detected = detect_gpu_model(gpu_name)
            gpu_label = f"{gpu_name}" + (f" -> {detected}" if detected else "")
            print(f"  CUDA              OK  ({gpu_label})")
        else:
            print("  CUDA              WARN  (torch.cuda.is_available() is False)")
    except ImportError:
        print("  PyTorch           FAIL  (not installed — run: pip install torch)")

    if ncu_ok:
        print(f"  ncu               OK  ({shutil.which('ncu')})")
    else:
        print("  ncu               not found  (kernel profiling needs Nsight Compute)")

    # ── Detect training script ────────────────────────────────────────────────
    script = _detect_training_script(cwd)

    # ── Print next steps ──────────────────────────────────────────────────────
    print()
    if torch_ok:
        script_name = script.name if script else "train.py"
        already = script and _already_instrumented(script.read_text(encoding="utf-8"))

        if already:
            print(f"  {script_name} is already instrumented with fournex.")
            print()
            print("  Run:")
            print(f"    frx collect --explain -- python {script_name}")
            print("  Then paste frx_llm_prompt.txt into your LLM.")
        else:
            if script:
                print(f"  Found {script_name}. Add these lines to instrument it:")
            else:
                print("  Add these lines to your training script:")
            print()
            for line in _SDK_SNIPPET.splitlines():
                print(f"    {line}")
            print()
            print("  Then run:")
            print(f"    frx collect --explain -- python {script_name}")
            print("  Paste frx_llm_prompt.txt into your LLM for optimization suggestions.")
            if script:
                print(f"\n  Or run:  frx init --patch {script_name}  to add the snippet automatically.")
    elif ncu_ok:
        print("  PyTorch not found. For CUDA kernel profiling:")
        print()
        print("    frx profile --ncu report.csv --explain")
        print()
        print("  Collect a report with:")
        print("    frx ncu-command full --output report.csv -- ./your_kernel_app")
    else:
        print("  Install PyTorch for training analysis, or Nsight Compute for kernel profiling.")
        print("  Then re-run frx init.")

    print()

    # ── Optional patch ────────────────────────────────────────────────────────
    patch_target = getattr(args, "patch", None)
    if patch_target:
        patch_path = Path(patch_target)
        if not patch_path.exists():
            print(f"  error: file not found: {patch_path}", file=sys.stderr)
            return 1
        _patch_script(patch_path)

    return 0


def doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, str, str]] = []  # (label, status, detail)

    checks.append(("Python", "ok", platform.python_version()))

    try:
        import torch  # type: ignore
        torch_version = getattr(torch, "__version__", "unknown")
        checks.append(("torch", "ok", torch_version))

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "unknown"
            checks.append(("CUDA available", "ok", f"{gpu_name} x{gpu_count}"))
        else:
            checks.append(("CUDA available", "warn", "torch.cuda.is_available() returned False"))
    except ImportError:
        checks.append(("torch", "fail", "not installed - run: pip install torch"))
        checks.append(("CUDA available", "skip", "torch not installed"))

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        checks.append(("nvidia-smi", "ok", nvidia_smi))
    else:
        checks.append(("nvidia-smi", "warn", "not found on PATH — GPU metrics sampling unavailable"))

    try:
        from . import profiler as _profiler  # noqa: F401
        checks.append(("fournex.profiler", "ok", "importable"))
    except Exception as exc:
        checks.append(("fournex.profiler", "fail", str(exc)))

    try:
        from .analysis import summarize_run_with_steady_state as _fn  # noqa: F401
        checks.append(("fournex.analysis", "ok", "importable"))
    except Exception as exc:
        checks.append(("fournex.analysis", "fail", str(exc)))

    status_labels = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]", "skip": "[SKIP]"}
    print("\nfrx doctor\n")
    all_ok = True
    for label, status, detail in checks:
        tag = status_labels.get(status, "[????]")
        print(f"  {tag}  {label:<36} {detail}")
        if status == "fail":
            all_ok = False

    if all_ok:
        print("\nAll checks passed.\n")
        return 0
    else:
        print("\nSome checks failed. See [FAIL] lines above.\n")
        return 1


def _extract_zip_run_bundle(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                _validate_zip_member_path(member.filename, destination_root)
            archive.extractall(destination)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid zip bundle: {zip_path}") from exc

    if _looks_like_run_dir(destination):
        return destination

    top_level_dirs = [path for path in destination.iterdir() if path.is_dir()]
    run_dirs = [path for path in top_level_dirs if _looks_like_run_dir(path)]
    if len(run_dirs) == 1:
        return run_dirs[0]

    raise ValueError(
        "zip bundle does not contain a recognizable run directory "
        "(expected derived/summary.json, raw/trace.jsonl, or profiler artifacts)"
    )


def _validate_zip_member_path(member_name: str, destination_root: Path) -> None:
    normalized = member_name.replace("\\", "/")
    target = (destination_root / normalized).resolve()
    parts = Path(normalized).parts

    if not normalized or Path(normalized).is_absolute() or ".." in parts:
        raise ValueError(f"unsafe zip member path: {member_name}")
    if target != destination_root and destination_root not in target.parents:
        raise ValueError(f"unsafe zip member path: {member_name}")


def _looks_like_run_dir(path: Path) -> bool:
    return (
        (path / "derived" / "summary.json").is_file()
        or (path / "raw" / "trace.jsonl").is_file()
        or any((path / "profiler").glob("*.json"))
        or (path / "metadata.json").is_file()
        or (path / "manifest.json").is_file()
    )


def _load_or_generate_summary(run_dir: Path) -> dict[str, Any] | None:
    derived_path = run_dir / "derived" / "summary.json"
    raw_trace_path = run_dir / "raw" / "trace.jsonl"
    environment = _read_run_config_environment(run_dir)

    if derived_path.exists():
        try:
            return json.loads(derived_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] could not read {derived_path}: {exc}; regenerating from raw trace", file=sys.stderr)

    if raw_trace_path.exists():
        return _generate_summary_from_trace(raw_trace_path, environment=environment)

    events = _events_from_profiler_bundle(run_dir, [])
    if events:
        try:
            from .analysis import summarize_run_with_steady_state
            return summarize_run_with_steady_state(events, environment=environment)
        except (ImportError, ValueError) as exc:
            print(f"[warn] could not generate summary from profiler bundle: {exc}", file=sys.stderr)

    return None


def _generate_summary_from_trace(
    trace_path: Path,
    *,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        from .analysis import summarize_run_with_steady_state
    except ImportError as exc:
        print(f"Warning: analysis module unavailable: {exc}", file=sys.stderr)
        return None

    events = _read_jsonl_events(trace_path)
    if not events:
        return None
    return summarize_run_with_steady_state(events, environment=environment)


def _generate_derived_summary_from_trace(
    run_dir: Path,
    warnings: list[str],
    *,
    environment: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    derived_path = run_dir / "derived" / "summary.json"
    raw_trace_path = run_dir / "raw" / "trace.jsonl"

    # `collect` passes overwrite=True so it can rebuild the summary WITH the
    # nvidia-smi samples it captured. The workload subprocess's SDK auto-persist
    # writes a derived summary first, but from its own events only — it has no GPU
    # samples (sampling runs in the parent), so that summary reports 0% util. Only
    # the parent has gpu_metrics.csv, so the parent must be authoritative.
    if derived_path.exists() and not overwrite:
        return
    if not raw_trace_path.exists():
        return

    try:
        from .analysis import summarize_run_with_steady_state
    except ImportError as exc:
        warnings.append(f"Could not generate derived summary: analysis module unavailable ({exc}).")
        return

    events = _read_jsonl_events(raw_trace_path)
    if not events:
        # Preserve any existing (e.g. subprocess-written) summary rather than
        # clobbering it with nothing.
        warnings.append("Raw trace is empty; derived/summary.json was not generated.")
        return

    # Fold in nvidia-smi samples captured during collection as gpu_sample events.
    # Without this the SDK trace carries no GPU utilization, so the summary reports
    # 0% util and the utilization-keyed rules (underutilized_gpu, launch_bound)
    # can never fire. (_gpu_metrics_csv_to_sdk_events returns [] if the CSV is
    # absent or header-only.)
    events.extend(_gpu_metrics_csv_to_sdk_events(run_dir / "gpu_metrics.csv"))

    summary = summarize_run_with_steady_state(events, environment=environment)
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(derived_path, summary)


def _read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _generate_derived_summary_from_profiler_bundle(
    run_dir: Path,
    warnings: list[str],
    *,
    environment: dict[str, Any] | None = None,
) -> None:
    derived_path = run_dir / "derived" / "summary.json"
    if derived_path.exists():
        return

    events = _events_from_profiler_bundle(run_dir, warnings)
    if not events:
        return

    try:
        from .analysis import summarize_run_with_steady_state
    except ImportError as exc:
        warnings.append(f"Could not generate derived summary from profiler: analysis module unavailable ({exc}).")
        return

    summary = summarize_run_with_steady_state(events, environment=environment)
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(derived_path, summary)


def _events_from_profiler_bundle(run_dir: Path, warnings: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    profiler_files = sorted((run_dir / "profiler").glob("*.json"))
    if profiler_files:
        try:
            events.extend(_profiler_trace_to_sdk_events(profiler_files[0], warnings))
        except Exception as exc:
            warnings.append(f"Could not parse profiler trace {profiler_files[0].name}: {exc}")

    events.extend(_gpu_metrics_csv_to_sdk_events(run_dir / "gpu_metrics.csv"))
    return events


def _profiler_trace_to_sdk_events(trace_path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    raw = json.loads(trace_path.read_text(encoding="utf-8", errors="replace"))
    trace_events = raw.get("traceEvents", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])

    complete = [
        e for e in trace_events
        if isinstance(e, dict) and e.get("ph", "X") == "X" and float(e.get("dur", 0) or 0) > 0
    ]
    if not complete:
        return []

    step_spans = _detect_profiler_step_spans(complete)
    if not step_spans:
        warnings.append(
            f"{trace_path.name}: no step boundaries detected. "
            "Add ProfilerStep#N annotations or top-level record_function wrappers for per-step analysis."
        )
        return []

    sdk_events: list[dict[str, Any]] = []
    for step_num, ts_start_us, ts_end_us in step_spans:
        # Restrict metric extraction to user_annotation events only — avoids double-counting
        # nested cpu_op / python_function spans that share names like "backward" or "dataloader".
        in_step = [
            e for e in complete
            if str(e.get("cat", "")) == "user_annotation"
            and float(e.get("ts", 0)) >= ts_start_us
            and float(e.get("ts", 0)) + float(e.get("dur", 0) or 0) <= ts_end_us + 1
        ]
        kernel_durations_us = [
            float(e.get("dur", 0) or 0)
            for e in complete
            if _is_cuda_kernel_evt(e)
            and float(e.get("ts", 0)) >= ts_start_us
            and float(e.get("ts", 0)) + float(e.get("dur", 0) or 0) <= ts_end_us + 1
        ]

        dl_ns = _max_dur_ns(in_step, _is_dataloader_evt)
        h2d_ns = _sum_dur_ns(in_step, _is_h2d_evt)
        fwd_ns = _max_dur_ns(in_step, _is_forward_evt)
        bwd_ns = _sum_dur_ns(in_step, _is_backward_evt)
        opt_ns = _sum_dur_ns(in_step, _is_optimizer_evt)
        wall_ns = int((ts_end_us - ts_start_us) * 1000)
        kernel_count = len(kernel_durations_us)
        small_kernel_count = sum(1 for duration in kernel_durations_us if duration < 10.0)
        median_kernel_us = _median(kernel_durations_us)

        sdk_events.append({"event_type": "step_start", "step_id": step_num, "duration_ns": 0, "payload": {"step_kind": "train"}})
        sdk_events.append(
            {
                "event_type": "profiler_window",
                "step_id": step_num,
                "duration_ns": wall_ns,
                "payload": {
                    "window_state": "exported",
                    "trace_path": str(trace_path),
                    "kernel_count": kernel_count,
                    "kernel_count_per_step": kernel_count,
                    "median_cuda_kernel_duration_us": median_kernel_us,
                    "small_kernel_fraction": (small_kernel_count / kernel_count) if kernel_count else 0.0,
                },
            }
        )
        if dl_ns > 0:
            sdk_events.append({"event_type": "dataloader_span", "step_id": step_num, "duration_ns": dl_ns, "payload": {"stage": "next"}})
        if h2d_ns > 0:
            sdk_events.append({"event_type": "memcpy_span", "step_id": step_num, "duration_ns": h2d_ns, "payload": {"copy_kind": "h2d"}})
        if fwd_ns > 0:
            sdk_events.append({"event_type": "phase_span", "step_id": step_num, "duration_ns": fwd_ns, "payload": {"phase_name": "forward"}})
        if bwd_ns > 0:
            sdk_events.append({"event_type": "phase_span", "step_id": step_num, "duration_ns": bwd_ns, "payload": {"phase_name": "backward"}})
        if opt_ns > 0:
            sdk_events.append({"event_type": "phase_span", "step_id": step_num, "duration_ns": opt_ns, "payload": {"phase_name": "optimizer"}})
        sdk_events.append({"event_type": "step_end", "step_id": step_num, "duration_ns": wall_ns, "payload": {"status": "ok", "step_kind": "train"}})

    return sdk_events


def _detect_profiler_step_spans(complete: list[dict[str, Any]]) -> list[tuple[int, float, float]]:
    # 1. ProfilerStep#N (standard torch.profiler step numbering)
    numbered: list[tuple[int, float, float]] = []
    for e in complete:
        name = str(e.get("name", ""))
        if name.startswith("ProfilerStep#"):
            try:
                num = int(name.split("#", 1)[1])
                ts = float(e.get("ts", 0))
                dur = float(e.get("dur", 0) or 0)
                if dur > 0:
                    numbered.append((num, ts, ts + dur))
            except (ValueError, IndexError):
                continue
    if numbered:
        return sorted(numbered)

    # 2. Outermost repeating user_annotation (largest average duration)
    user_anns = [e for e in complete if str(e.get("cat", "")) == "user_annotation"]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for e in user_anns:
        by_name.setdefault(str(e.get("name", "")), []).append(e)

    candidates = {name: evts for name, evts in by_name.items() if evts}
    if not candidates:
        return []

    step_name = max(
        candidates,
        key=lambda n: sum(float(e.get("dur", 0) or 0) for e in candidates[n]) / len(candidates[n]),
    )
    step_events = sorted(candidates[step_name], key=lambda e: float(e.get("ts", 0)))
    return [
        (i + 1, float(e.get("ts", 0)), float(e.get("ts", 0)) + float(e.get("dur", 0) or 0))
        for i, e in enumerate(step_events)
    ]


def _is_dataloader_evt(e: dict[str, Any]) -> bool:
    name = str(e.get("name", "")).lower()
    # Match the standard PyTorch DataLoader annotation and common user-defined equivalents.
    # Use _max_dur_ns (not sum) at call site to avoid double-counting nested wrappers.
    return (
        ("dataloader" in name and "__next__" in name)
        or name in ("get_next_batch", "fetch_next_batch", "load_batch", "data_fetch", "data_loading")
    )


def _is_h2d_evt(e: dict[str, Any]) -> bool:
    name = str(e.get("name", "")).lower()
    cat = str(e.get("cat", "")).lower()
    if "dtoh" in name or "d2h" in name or "dtod" in name or "d2d" in name:
        return False
    return "htod" in name or "h2d" in name or "host_to_device" in name or "host to dev" in name or cat == "gpu_memcpy"


def _is_forward_evt(e: dict[str, Any]) -> bool:
    name = str(e.get("name", "")).lower()
    return (
        "forward_backward" in name
        or name == "forward"
        or ("fwd" in name and "backward" not in name and "dataloader" not in name)
    )


def _is_backward_evt(e: dict[str, Any]) -> bool:
    name = str(e.get("name", "")).lower()
    return "backward" in name and "forward_backward" not in name


def _is_optimizer_evt(e: dict[str, Any]) -> bool:
    name = str(e.get("name", "")).lower()
    return "optimizer.step" in name or (
        "step" in name and any(opt in name for opt in ("adam", "sgd", "adamw", "rmsprop", "adagrad"))
    )


def _is_cuda_kernel_evt(e: dict[str, Any]) -> bool:
    cat = str(e.get("cat", "")).lower()
    name = str(e.get("name", "")).lower()
    return "kernel" in cat or "cuda_kernel" in cat or ("cuda" in cat and "kernel" in name)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _sum_dur_ns(events: list[dict[str, Any]], predicate) -> int:
    return sum(int(float(e.get("dur", 0) or 0) * 1000) for e in events if predicate(e))


def _max_dur_ns(events: list[dict[str, Any]], predicate) -> int:
    vals = [int(float(e.get("dur", 0) or 0) * 1000) for e in events if predicate(e)]
    return max(vals) if vals else 0


def _gpu_metrics_csv_to_sdk_events(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                util_gpu = _safe_float(row.get("utilization.gpu") or row.get("gpu_util"))
                util_mem = _safe_float(row.get("utilization.memory") or row.get("memory_util"))
                mem_used = _safe_float(row.get("memory.used") or row.get("memory_used"))
                mem_total = _safe_float(row.get("memory.total") or row.get("memory_total"))
                if util_gpu is None and mem_used is None:
                    continue
                payload: dict[str, Any] = {}
                if util_gpu is not None:
                    payload["utilization_gpu_pct"] = util_gpu
                if util_mem is not None:
                    payload["utilization_mem_pct"] = util_mem
                if mem_used is not None:
                    payload["memory_used_bytes"] = mem_used * 1024 * 1024
                if mem_total is not None:
                    payload["memory_total_bytes"] = mem_total * 1024 * 1024
                events.append({"event_type": "gpu_sample", "step_id": None, "payload": payload})
    except Exception:
        pass
    return events


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def smoke_test(args: argparse.Namespace) -> int:
    import tempfile

    print("frx smoke-test\n")
    checks: list[tuple[str, bool, str]] = []

    with tempfile.TemporaryDirectory(prefix="frx_smoke_") as tmpdir:
        run_dir = Path(tmpdir) / "smoke-run"

        # 1. Write synthetic input-bound profiler trace
        try:
            run_dir.mkdir()
            (run_dir / "profiler").mkdir()
            (run_dir / "raw").mkdir()
            (run_dir / "derived").mkdir()
            _write_smoke_profiler_trace(run_dir)
            checks.append(("write synthetic profiler trace", True, ""))
        except Exception as exc:
            checks.append(("write synthetic profiler trace", False, str(exc)))
            return _print_smoke_results(checks)

        # 2. Generate derived summary from profiler trace
        warnings: list[str] = []
        try:
            _generate_derived_summary_from_profiler_bundle(run_dir, warnings)
            derived_ok = (run_dir / "derived" / "summary.json").exists()
            checks.append(("generate derived/summary.json", derived_ok, "" if derived_ok else "file not created"))
            if warnings:
                checks.append(("no parse warnings", False, "; ".join(warnings)))
            else:
                checks.append(("no parse warnings", True, ""))
        except Exception as exc:
            checks.append(("generate derived/summary.json", False, str(exc)))
            return _print_smoke_results(checks)

        # 3. Load and validate summary structure
        summary = None
        try:
            summary = json.loads((run_dir / "derived" / "summary.json").read_text(encoding="utf-8"))
            checks.append(("load and parse summary JSON", True, ""))
        except Exception as exc:
            checks.append(("load and parse summary JSON", False, str(exc)))
            return _print_smoke_results(checks)

        ss = summary.get("steady_state") or summary
        step_count = ss.get("step_count", 0)
        diagnosis = ss.get("diagnosis", {})
        bottleneck = diagnosis.get("primary_bottleneck")
        recs = diagnosis.get("recommendations", [])

        checks.append(("step_count > 0", step_count > 0, f"got {step_count}"))
        checks.append(("primary_bottleneck == input_bound", bottleneck == "input_bound", f"got {bottleneck!r}"))
        checks.append(("recommendations generated", len(recs) > 0, f"got {len(recs)}"))

        # 4. Verify analyze path works on the same bundle
        try:
            loaded = _load_or_generate_summary(run_dir)
            checks.append(("analyze path loads summary", loaded is not None, ""))
        except Exception as exc:
            checks.append(("analyze path loads summary", False, str(exc)))

    return _print_smoke_results(checks)


def _print_smoke_results(checks: list[tuple[str, bool, str]]) -> int:
    all_ok = True
    for name, passed, detail in checks:
        tag = "[PASS]" if passed else "[FAIL]"
        suffix = f"  ({detail})" if detail else ""
        print(f"  {tag}  {name}{suffix}")
        if not passed:
            all_ok = False
    print()
    if all_ok:
        print("All smoke checks passed.\n")
        return 0
    else:
        print("Smoke test failed. See [FAIL] lines above.\n")
        return 1


def _write_smoke_profiler_trace(run_dir: Path) -> None:
    step_us = 500_000   # 500ms per step
    dl_us = 400_000     # 400ms DataLoader (80% of step)
    fwdbwd_us = 80_000  # 80ms compute
    opt_us = 2_000      # 2ms optimizer
    h2d_us = 500        # 0.5ms H2D copy

    trace_events = []
    ts = 0
    for _ in range(5):
        step_ts = ts
        trace_events += [
            {"ph": "X", "cat": "user_annotation", "name": "train_step",
             "ts": step_ts, "dur": step_us, "pid": 1, "tid": 1},
            {"ph": "X", "cat": "user_annotation",
             "name": "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__",
             "ts": step_ts + 500, "dur": dl_us, "pid": 1, "tid": 1},
            {"ph": "X", "cat": "user_annotation", "name": "host_to_device_copy",
             "ts": step_ts + dl_us + 1_000, "dur": h2d_us, "pid": 1, "tid": 1},
            {"ph": "X", "cat": "user_annotation", "name": "forward_backward_optimizer",
             "ts": step_ts + dl_us + h2d_us + 1_000, "dur": fwdbwd_us, "pid": 1, "tid": 1},
            {"ph": "X", "cat": "user_annotation", "name": "Optimizer.step#SGD.step",
             "ts": step_ts + dl_us + h2d_us + fwdbwd_us - opt_us + 1_000, "dur": opt_us, "pid": 1, "tid": 1},
        ]
        ts += step_us + 5_000

    (run_dir / "profiler" / "profiler_trace.json").write_text(
        json.dumps({"traceEvents": trace_events}, indent=2), encoding="utf-8"
    )


def _print_ptx_report(result: dict[str, Any]) -> None:
    sep = "-" * 60
    summary = result.get("run_summary", {})
    scope = result.get("diagnostic_scope", {})
    print(f"\n{sep}")
    print("  Fournex - PTX Analysis")
    print(f"  File    : {result.get('filename', '<unknown>')}")
    print(f"  Target  : {result.get('target') or 'unknown'}")
    print(sep)

    print("\nVERDICT")
    print(f"  Primary Bottleneck : {result.get('primary_bottleneck') or 'none'}")
    print(f"  Confidence         : {scope.get('confidence', 'unknown')} ({scope.get('type', 'static_ptx')})")
    message = scope.get("message")
    if message:
        print(f"  Scope              : {message}")

    print("\nEVIDENCE")
    print(f"  Kernels            : {summary.get('kernel_count', result.get('kernel_count', 0))}")
    print(f"  Avg Registers      : {summary.get('avg_register_count', 0)}")
    print(f"  Max Registers      : {summary.get('max_register_count', 0)}")
    print(f"  Kernels w/ Spills  : {summary.get('kernels_with_spills', 0)}")
    print(f"  Max Global Mem Mix : {float(summary.get('max_global_memory_ratio', 0.0)):.2f}")
    print(f"  FP64 Kernels       : {summary.get('kernels_with_fp64', 0)}")

    _print_finding_list(result.get("findings", []))
    _print_recommendation_list(result.get("recommendations", []))
    print()


def _print_cuda_source_report(result: dict[str, Any]) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print("  Fournex - CUDA Source Analysis")
    print(f"  GPU Model: {result.get('gpu_model') or 'unspecified'}")
    print(sep)

    print("\nSUMMARY")
    print(f"  Kernels : {result.get('kernel_count', 0)}")
    print(f"  Launches: {result.get('launch_count', 0)}")

    _print_finding_list(result.get("findings", []))

    launch_advice = result.get("launch_advisor", [])
    if launch_advice:
        print("\nLAUNCH ADVISOR")
        for item in launch_advice[:3]:
            candidates = item.get("candidate_block_sizes", [])
            blocks = [str(candidate.get("block_size")) for candidate in candidates if candidate.get("block_size")]
            print(f"  - {item.get('kernel_name', '<kernel>')}: try block sizes {', '.join(blocks) or 'n/a'}")
            notes = item.get("notes", [])
            if notes:
                print(f"    {notes[0]}")
    print()


def _print_ncu_report(result: dict[str, Any]) -> None:
    sep = "-" * 60
    summary = result.get("ncu_run_summary", {})
    scope = result.get("diagnostic_scope", {})
    print(f"\n{sep}")
    print("  Fournex - Nsight Compute Analysis")
    print(f"  Kernels : {result.get('kernel_count', 0)}")
    print(sep)

    print("\nVERDICT")
    print(f"  Primary Bottleneck : {result.get('primary_bottleneck') or 'none'}")
    print(f"  Confidence         : {scope.get('confidence', 'unknown')} ({scope.get('type', 'measured_ncu')})")
    message = scope.get("message")
    if message:
        print(f"  Scope              : {message}")

    _print_ncu_validation(result.get("validation", {}))

    print("\nMEASURED METRICS")
    print(f"  DRAM Throughput    : {_fmt_optional_pct(summary.get('avg_dram_throughput_pct'))}")
    print(f"  L1 Hit Rate        : {_fmt_optional_pct(summary.get('avg_l1_cache_hit_rate_pct'))}")
    print(f"  L2 Hit Rate        : {_fmt_optional_pct(summary.get('avg_l2_cache_hit_rate_pct'))}")
    print(f"  Issue Utilization  : {_fmt_optional_pct(summary.get('avg_issue_slot_utilization_pct'))}")
    print(f"  Occupancy          : {_fmt_optional_pct(summary.get('avg_occupancy_pct'))}")
    print(f"  Eligible Warps/Sched: {_fmt_optional_float(summary.get('avg_eligible_warps_per_scheduler'))}")
    print(f"  Scheduler Active   : {_fmt_optional_pct(summary.get('avg_scheduler_active_pct'))}")
    print(f"  Memory Stall Frac  : {_fmt_optional_float(summary.get('memory_stall_fraction'))}")
    print(f"  Dominant Stall     : {summary.get('dominant_warp_stall', 'unknown')}")
    causes = summary.get("occupancy_limit_causes") or []
    if causes:
        print(f"  Occupancy Limits   : {', '.join(causes)}")

    _print_bottleneck_list(result.get("bottlenecks", []))
    _print_recommendation_list(result.get("recommendations", []))
    print()


def _print_ncu_validation(validation: dict[str, Any]) -> None:
    errors = validation.get("errors", [])
    warnings = validation.get("warnings", [])
    if not errors and not warnings:
        return
    print("\nCSV VALIDATION")
    for error in errors:
        print(f"  [ERROR] {error}")
    for warning in warnings[:4]:
        print(f"  [WARN]  {warning}")


def _print_cuda_comparison_report(result: dict[str, Any]) -> None:
    sep = "-" * 60
    verdict = result.get("verdict", {})
    label_a = result.get("label_a", "before")
    label_b = result.get("label_b", "after")
    winner = verdict.get("overall_winner", "tie")
    winner_label = label_b if winner == "b" else label_a if winner == "a" else "tie"

    print(f"\n{sep}")
    print("  Fournex - CUDA Before/After Comparison")
    print(f"  Before : {label_a}")
    print(f"  After  : {label_b}")
    print(sep)

    print("\nVERDICT")
    print(f"  Winner      : {winner_label}")
    print(f"  Score Before: {_fmt_optional_float(verdict.get('score_a'))}")
    print(f"  Score After : {_fmt_optional_float(verdict.get('score_b'))}")
    print(f"  Delta       : {_fmt_optional_float(verdict.get('score_delta'))}")

    dimensions_b = verdict.get("dimensions_won_by_b", [])
    dimensions_a = verdict.get("dimensions_won_by_a", [])
    if dimensions_b:
        print(f"  Improved    : {', '.join(dimensions_b)}")
    if dimensions_a:
        print(f"  Regressed   : {', '.join(dimensions_a)}")

    tradeoffs = result.get("tradeoffs", [])
    if tradeoffs:
        print("\nTRADEOFFS")
        for tradeoff in tradeoffs[:3]:
            print(f"  - {tradeoff.get('label', 'tradeoff')}: {tradeoff.get('message', '')}")

    for section_name, key in (("STATIC FINDINGS", "static_diff"), ("PTX FINDINGS", "ptx_diff")):
        diff = result.get(key, {})
        findings_diff = diff.get("findings_diff", {}) if diff.get("available") else {}
        resolved = findings_diff.get("resolved_in_b", [])
        new = findings_diff.get("new_in_b", [])
        if resolved or new:
            print(f"\n{section_name}")
            for code in resolved:
                print(f"  [+] Resolved: {code}")
            for code in new:
                print(f"  [-] New     : {code}")

    ncu_diff = result.get("ncu_diff", {})
    if ncu_diff.get("available"):
        print("\nNCU DELTAS")
        for key in ("avg_dram_throughput_pct", "avg_l1_cache_hit_rate_pct", "avg_issue_slot_utilization_pct", "memory_stall_fraction"):
            info = ncu_diff.get(key, {})
            if info.get("delta") is not None:
                print(f"  - {key}: {info.get('a')} -> {info.get('b')} ({info.get('delta'):+.4f})")
    print()


def _print_finding_list(findings: list[dict[str, Any]]) -> None:
    if not findings:
        print("\nFINDINGS")
        print("  No findings above threshold.")
        return
    print(f"\nFINDINGS ({min(len(findings), 5)} of {len(findings)})")
    for finding in findings[:5]:
        location = ""
        if finding.get("filename") and finding.get("line"):
            location = f" ({finding['filename']}:{finding['line']})"
        kernel = f" [{finding['kernel_name']}]" if finding.get("kernel_name") else ""
        print(f"  - {finding.get('severity', 'info').upper()}: {finding.get('code', 'finding')}{kernel}{location}")
        message = finding.get("message")
        if message:
            print(f"    {message}")


def _print_bottleneck_list(bottlenecks: list[dict[str, Any]]) -> None:
    if not bottlenecks:
        return
    print(f"\nBOTTLENECK RANKING ({min(len(bottlenecks), 5)} of {len(bottlenecks)})")
    for item in bottlenecks[:5]:
        print(f"  - {item.get('label', 'unknown')}: score {float(item.get('score', 0.0)):.2f}")


def _print_recommendation_list(recommendations: list[dict[str, Any]]) -> None:
    import textwrap
    if not recommendations:
        print("\nRECOMMENDATIONS")
        print("  No recommendations generated.")
        return
    print(f"\nRECOMMENDATIONS ({min(len(recommendations), 3)} of {len(recommendations)})")
    thin = "-" * 64
    for index, rec in enumerate(recommendations[:3], start=1):
        title = rec.get("title", rec.get("id", "recommendation"))
        priority = rec.get("priority", "low").upper()
        score = rec.get("score", 0.0)
        print(f"\n  {thin}")
        print(f"  {index}. [{priority}] {title}")
        print(f"     Tier: {rec.get('tier', 'next')}   Score: {score:.2f}   Triggered by: {rec.get('triggered_by', 'n/a')}")

        why = rec.get("why")
        if why:
            print(f"\n     Why:")
            for line in textwrap.wrap(why, width=60):
                print(f"       {line}")

        actions = rec.get("actions", [])
        if actions:
            print(f"\n     Actions:")
            for i, action in enumerate(actions[:3], start=1):
                lines = textwrap.wrap(action, width=60)
                print(f"       {i}. {lines[0]}")
                for continuation in lines[1:]:
                    print(f"          {continuation}")

        validation_steps = rec.get("validation_steps") or []
        if validation_steps:
            metrics = ",".join(step["metric"] for step in validation_steps)
            print(f"\n     Validate:")
            print(f"       ncu --metrics {metrics} \\")
            print(f"           --csv ./report.csv ./your_app")
            for step in validation_steps:
                direction_arrow = "<--" if step["direction"] == "decrease" else "-->" if step["direction"] == "increase" else "   "
                label = step.get("label", step["metric"])
                expected = step.get("expected", "")
                threshold = step.get("threshold_good")
                threshold_hint = f" (target: {threshold})" if threshold is not None else ""
                current = step.get("current_value")
                was_hint = f"was {current}; " if current is not None else ""
                lines = textwrap.wrap(f"{direction_arrow} {label}: {was_hint}{expected}{threshold_hint}", width=60)
                print(f"       {lines[0]}")
                for continuation in lines[1:]:
                    print(f"           {continuation}")


def _print_wrapped(text: str, indent: str = "  ", width: int = 72) -> None:
    import textwrap
    lines = textwrap.wrap(text, width=width - len(indent))
    for line in lines:
        print(f"{indent}{line}")


def _ncu_metric_status(value: float | None, warn: float, crit: float, low_is_bad: bool = True) -> str:
    if value is None:
        return "[--]"
    bad = value < crit if low_is_bad else value > crit
    warn_flag = value < warn if low_is_bad else value > warn
    if bad:
        return "[!!]"
    if warn_flag:
        return "[ !]"
    return "[ok]"


def _print_ncu_report_full(result: dict[str, Any], source: str = "", args: Any = None) -> None:
    import textwrap

    sep = "=" * 68
    thin = "-" * 68
    summary = result.get("ncu_run_summary", {})
    scope = result.get("diagnostic_scope", {})
    bottlenecks = result.get("bottlenecks", [])
    recommendations = result.get("recommendations", [])
    kernel_count = result.get("kernel_count", 0)

    print(f"\n{sep}")
    print("  Fournex - CUDA Performance Profile")
    if source:
        print(f"  Source  : {source}")
    print(f"  Kernels : {kernel_count}")
    confidence = scope.get("confidence", "unknown")
    print(f"  Confidence: {confidence}")
    print(sep)

    # ── VERDICT ────────────────────────────────────────────────────────────────
    primary = result.get("primary_bottleneck") or "none"
    print("\nVERDICT")
    print(f"  Primary bottleneck : {primary}")
    secondaries = [b["label"] for b in bottlenecks if b["label"] != primary]
    if secondaries:
        print(f"  Also detected      : {', '.join(secondaries)}")
    scope_msg = scope.get("message")
    if scope_msg:
        print(f"  Note               : {scope_msg}")

    # Show errors immediately; warnings are deferred to the end of the report.
    csv_validation = result.get("validation", {})
    csv_errors = csv_validation.get("errors", [])
    if csv_errors:
        print("\nCSV VALIDATION")
        for error in csv_errors:
            print(f"  [ERROR] {error}")

    # ── MEASURED METRICS ───────────────────────────────────────────────────────
    print(f"\nMEASURED METRICS")
    print(f"  {'Status':<6}  {'Metric':<32} {'Value':<12} Threshold hint")
    print(f"  {thin[:64]}")

    dram = summary.get("avg_dram_throughput_pct")
    st = _ncu_metric_status(dram, warn=60.0, crit=80.0, low_is_bad=False)
    print(f"  {st}  {'DRAM Throughput':<32} {_fmt_optional_pct(dram):<12} high >= 80% -> memory bandwidth bound")

    tc = summary.get("avg_tensor_core_utilization_pct")
    st = _ncu_metric_status(tc, warn=20.0, crit=10.0, low_is_bad=True)
    print(f"  {st}  {'Tensor Core Utilization':<32} {_fmt_optional_pct(tc):<12} low < 10% -> underutilized TC units")

    l1 = summary.get("avg_l1_cache_hit_rate_pct")
    st = _ncu_metric_status(l1, warn=60.0, crit=40.0, low_is_bad=True)
    print(f"  {st}  {'L1 Hit Rate':<32} {_fmt_optional_pct(l1):<12} low < 40% -> L1 cache thrashing")

    l2 = summary.get("avg_l2_cache_hit_rate_pct")
    st = _ncu_metric_status(l2, warn=65.0, crit=50.0, low_is_bad=True)
    print(f"  {st}  {'L2 Hit Rate':<32} {_fmt_optional_pct(l2):<12} low < 50% -> L2 cache thrashing")

    load_sectors = summary.get("avg_global_load_sectors_per_request")
    st = _ncu_metric_status(load_sectors, warn=4.0, crit=8.0, low_is_bad=False)
    ls_str = f"{load_sectors:.1f}" if load_sectors is not None else "n/a"
    print(f"  {st}  {'Load Sectors/Request':<32} {ls_str:<12} high > 4 -> uncoalesced global loads (ideal = 1)")

    isu = summary.get("avg_issue_slot_utilization_pct")
    st = _ncu_metric_status(isu, warn=55.0, crit=40.0, low_is_bad=True)
    print(f"  {st}  {'Issue Slot Utilization':<32} {_fmt_optional_pct(isu):<12} low < 60% -> low ILP / underutilized SMs")

    occ = summary.get("avg_occupancy_pct")
    st = _ncu_metric_status(occ, warn=50.0, crit=30.0, low_is_bad=True)
    print(f"  {st}  {'Occupancy':<32} {_fmt_optional_pct(occ):<12} low < 30% -> too few warps to hide latency")

    eligible = summary.get("avg_eligible_warps_per_scheduler")
    st = _ncu_metric_status(eligible, warn=1.0, crit=0.5, low_is_bad=True)
    elig_str = f"{eligible:.2f}" if eligible is not None else "n/a"
    print(f"  {st}  {'Eligible Warps/Scheduler':<32} {elig_str:<12} low < 0.5 -> warp scheduler starved")

    sched_active = summary.get("avg_scheduler_active_pct")
    st = _ncu_metric_status(sched_active, warn=50.0, crit=35.0, low_is_bad=True)
    print(f"  {st}  {'Scheduler Active':<32} {_fmt_optional_pct(sched_active):<12} low < 35% -> warp scheduler underutilized")

    dom_stall = summary.get("dominant_warp_stall", "unknown")
    dom_pct = summary.get("dominant_warp_stall_pct", 0.0)
    st = _ncu_metric_status(dom_pct, warn=20.0, crit=30.0, low_is_bad=False)
    dom_str = f"{dom_stall} ({dom_pct:.1f}%)" if dom_stall != "unknown" else "unknown"
    print(f"  {st}  {'Dominant Warp Stall':<32} {dom_str}")

    causes = [c for c in (summary.get("occupancy_limit_causes") or []) if c != "unknown_threads_per_block"]
    if causes and occ is not None and occ < 50.0:
        print(f"\n  Occupancy limited by: {', '.join(causes)}")

    missing_count = sum(1 for v in [dram, tc, l1, l2, load_sectors, isu, occ, eligible, sched_active] if v is None)
    if missing_count >= 4:
        print(f"\n  Note: {missing_count} metrics not collected -- re-run with --preset full to capture all.")

    # ── BOTTLENECKS ────────────────────────────────────────────────────────────
    if bottlenecks:
        print(f"\nBOTTLENECKS DETECTED ({len(bottlenecks)})")
        for b in bottlenecks:
            score_bar = "#" * int(b.get("score", 0.0) * 10)
            print(f"  [{score_bar:<10}] {b['label']:<36} score {float(b.get('score', 0.0)):.2f}")

    # ── RECOMMENDATIONS ────────────────────────────────────────────────────────
    if not recommendations:
        print("\nRECOMMENDATIONS")
        print("  No recommendations generated.")
    else:
        print(f"\nRECOMMENDATIONS ({len(recommendations)})")
        for idx, rec in enumerate(recommendations, start=1):
            priority = rec.get("priority", "low").upper()
            tier = rec.get("tier", "next")
            score = rec.get("score", 0.0)
            title = rec.get("title", rec.get("id", "recommendation"))
            print(f"\n  {'-' * 64}")
            print(f"  {idx}. [{priority}] {title}")
            print(f"     Tier: {tier}   Score: {score:.2f}   Triggered by: {rec.get('triggered_by', 'n/a')}")

            why = rec.get("why")
            if why:
                print(f"\n     Why:")
                _print_wrapped(why, indent="       ")

            actions = rec.get("actions") or []
            if actions:
                print(f"\n     Actions:")
                for i, action in enumerate(actions, start=1):
                    lines = textwrap.wrap(action, width=60)
                    print(f"       {i}. {lines[0]}")
                    for continuation in lines[1:]:
                        print(f"          {continuation}")

            validation_steps = rec.get("validation_steps") or []
            if validation_steps:
                metrics = ",".join(step["metric"] for step in validation_steps)
                print(f"\n     Validate:")
                print(f"       ncu --metrics {metrics} \\")
                print(f"           --csv ./report.csv ./your_app")
                for step in validation_steps:
                    direction_arrow = "<--" if step["direction"] == "decrease" else "-->" if step["direction"] == "increase" else "   "
                    label = step.get("label", step["metric"])
                    expected = step.get("expected", "")
                    threshold = step.get("threshold_good")
                    threshold_hint = f" (target: {threshold})" if threshold is not None else ""
                    current = step.get("current_value")
                    was_hint = f"was {current}; " if current is not None else ""
                    lines = textwrap.wrap(f"{direction_arrow} {label}: {was_hint}{expected}{threshold_hint}", width=60)
                    print(f"       {lines[0]}")
                    for continuation in lines[1:]:
                        print(f"           {continuation}")
            elif rec.get("validation"):
                validation = rec["validation"]
                print(f"\n     Validation:")
                for v in validation:
                    lines = textwrap.wrap(v, width=60)
                    print(f"       - {lines[0]}")
                    for continuation in lines[1:]:
                        print(f"         {continuation}")

            risks = rec.get("risks") or rec.get("caveats") or []
            if isinstance(risks, str):
                risks = [risks]
            if risks:
                print(f"\n     Risks/Caveats:")
                for r in risks:
                    lines = textwrap.wrap(r, width=60)
                    print(f"       (!) {lines[0]}")
                    for continuation in lines[1:]:
                        print(f"         {continuation}")

    # ── DATA NOTES (deferred warnings) ────────────────────────────────────────
    # Only show preset-completeness warnings when the corresponding bottleneck
    # family was actually detected. Suppressing irrelevant missing-metric warnings
    # keeps the output focused — a kernel diagnosed as sync-bound doesn't need
    # a warning about missing tensor core metrics.
    csv_warnings = csv_validation.get("warnings", [])
    if csv_warnings:
        _MEMORY_BN = {"memory_bandwidth_bound", "warp_stall_memory", "l1_cache_thrashing", "l2_cache_thrashing", "uncoalesced_access"}
        _STALL_BN  = {"warp_stall_memory", "warp_stall_sync"}
        _TC_BN     = {"tensor_core_underutilized"}
        _OCC_BN    = {"occupancy_limited", "occupancy_limited_by_registers", "occupancy_limited_by_shared_memory", "occupancy_limited_by_block_size", "low_warp_scheduler_utilization"}
        _bn_labels = {b["label"] for b in bottlenecks}
        _WARNING_GATES = {
            "Memory diagnosis":     bool(_MEMORY_BN & _bn_labels),
            "Warp stall diagnosis": bool(_STALL_BN  & _bn_labels),
            "Tensor core diagnosis":bool(_TC_BN     & _bn_labels),
            "Occupancy diagnosis":  bool(_OCC_BN    & _bn_labels),
        }
        relevant = [
            w for w in csv_warnings
            if not any(w.startswith(prefix) for prefix in _WARNING_GATES)
            or any(w.startswith(prefix) and active for prefix, active in _WARNING_GATES.items())
        ]
        if relevant:
            print(f"\nDATA NOTES")
            for w in relevant:
                print(f"  [warn] {w}")

    # ── NEXT STEPS ─────────────────────────────────────────────────────────────
    print(f"\n{thin}")
    print("NEXT STEPS")
    if source and not source.startswith("ncu ") and not source.startswith("live"):
        print(f"  Re-run analysis:  frx profile --ncu {source}")
    else:
        print(f"  Re-run after changes:  frx profile -- <your_workload_command>")
    if recommendations:
        top = recommendations[0]
        print(f"  Top priority fix:  {top.get('title', top.get('id', ''))}")
    print()


def _fmt_optional_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1f}%"


def _fmt_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _analyze_ncu_diff(args: argparse.Namespace) -> int:
    if not args.baseline or not args.optimized:
        print("Error: --baseline and --optimized must both be provided together.", file=sys.stderr)
        return 1

    baseline_path  = Path(args.baseline)
    optimized_path = Path(args.optimized)

    for path in (baseline_path, optimized_path):
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            return 1

    try:
        from .ncu_comparison import diff_ncu_runs
    except ImportError as exc:
        print(f"Error: ncu_comparison module unavailable: {exc}", file=sys.stderr)
        return 1

    baseline_text  = baseline_path.read_text(encoding="utf-8-sig", errors="replace")
    optimized_text = optimized_path.read_text(encoding="utf-8-sig", errors="replace")

    result = diff_ncu_runs(
        baseline_text,
        optimized_text,
        label_baseline=baseline_path.stem,
        label_optimized=optimized_path.stem,
        environment=_environment_from_args(args),
    )

    if args.output_json:
        print(json.dumps(result, indent=2))
        return 0

    _print_ncu_comparison_report(result)
    return 0


def _print_validation_delta_table(
    deltas: dict[str, Any],
    verdict: dict[str, Any],
    label_before: str,
    label_after: str,
) -> None:
    if not deltas:
        return

    _OUTCOME_RESULT = {
        "improved":  "optimization validated",
        "regressed": "regression detected",
        "mixed":     "mixed results",
        "neutral":   "no significant change",
    }
    _DIR_TAG = {"improved": "[+]", "regressed": "[-]", "neutral": "[=]"}

    col_metric  = 30
    col_val     = 10

    before_hdr = label_before[:col_val].rjust(col_val)
    after_hdr  = label_after[:col_val].rjust(col_val)
    print(f"\nBEFORE / AFTER VALIDATION")
    print(f"\n  {'Metric':<{col_metric}}  {before_hdr}  {after_hdr}  Change")
    print(f"  {'-' * (col_metric + col_val * 2 + 24)}")

    for info in deltas.values():
        lbl  = info["label"]
        unit = info["unit"]
        a    = info["baseline"]
        b    = info["optimized"]
        d    = info["delta"]
        direction = info["direction"] or "n/a"
        tag  = _DIR_TAG.get(direction, "   ")

        def _fmt(v: Any) -> str:
            if v is None:
                return "n/a"
            return f"{float(v):.1f}{unit}"

        a_str = _fmt(a).rjust(col_val)
        b_str = _fmt(b).rjust(col_val)

        if d is not None and abs(d) >= 0.001:
            delta_str = f"{d:+.1f}{unit}"
            change = f"{tag} {direction}  ({delta_str})"
        else:
            change = f"{tag} {direction}"

        print(f"  {lbl:<{col_metric}}  {a_str}  {b_str}  {change}")

    outcome      = verdict.get("outcome", "neutral")
    resolved     = verdict.get("bottlenecks_resolved", 0)
    new_count    = verdict.get("bottlenecks_new", 0)
    result_label = _OUTCOME_RESULT.get(outcome, outcome)
    detail       = f"({resolved} bottleneck{'s' if resolved != 1 else ''} resolved, {new_count} new)"
    print(f"\n  Result: {result_label}  {detail}")


def _print_ncu_comparison_report(result: dict[str, Any]) -> None:
    sep = "-" * 60
    label_b = result["label_baseline"]
    label_o = result["label_optimized"]
    verdict = result["verdict"]
    bdiff   = result["bottleneck_diff"]
    deltas  = result["metric_deltas"]

    outcome_labels = {
        "improved":  "IMPROVED",
        "regressed": "REGRESSED",
        "mixed":     "MIXED  (some resolved, some new)",
        "neutral":   "NEUTRAL  (no change in bottleneck profile)",
    }

    print(f"\n{sep}")
    print(f"  GPU Autopilot - NCU Before/After Comparison")
    print(f"  Baseline  : {label_b}")
    print(f"  Optimized : {label_o}")
    print(sep)

    print(f"\nVERDICT: {outcome_labels.get(verdict['outcome'], verdict['outcome'].upper())}")
    print(f"  Bottlenecks resolved  : {verdict['bottlenecks_resolved']}")
    print(f"  Bottlenecks new       : {verdict['bottlenecks_new']}")
    print(f"  Persistent (improved) : {verdict['bottlenecks_improved']} of {verdict['bottlenecks_persistent']}")

    if bdiff["resolved"]:
        print(f"\nRESOLVED")
        for label in bdiff["resolved"]:
            print(f"  [+]  {label}")

    if bdiff["new"]:
        print(f"\nNEW REGRESSIONS")
        for label in bdiff["new"]:
            print(f"  [-]  {label}")

    if bdiff["persistent"]:
        improved_set = set(bdiff["improved"])
        score_deltas = bdiff["score_deltas"]
        print(f"\nPERSISTENT")
        for label in bdiff["persistent"]:
            d = score_deltas.get(label, 0.0)
            tag = "  (score improved)" if label in improved_set else ""
            print(f"  [~]  {label:<42}  score {d:+.2f}{tag}")

    _print_validation_delta_table(deltas, verdict, label_b, label_o)

    baseline_validation = result.get("baseline", {}).get("validation", {})
    optimized_validation = result.get("optimized", {}).get("validation", {})
    if baseline_validation.get("errors") or optimized_validation.get("errors"):
        print(f"\nCSV VALIDATION")
        for lbl, validation in ((label_b, baseline_validation), (label_o, optimized_validation)):
            for error in validation.get("errors", []):
                print(f"  [ERROR] {lbl}: {error}")
    print()


def _print_analysis_report(run_dir: Path, summary: dict[str, Any], scope: str = "auto") -> None:
    run_id = run_dir.name
    has_steady_state = "steady_state" in summary
    has_run = "run" in summary

    if scope == "auto":
        scope_data = summary.get("steady_state") or summary.get("run") or summary
    elif scope == "steady_state" and has_steady_state:
        scope_data = summary["steady_state"]
    elif scope == "run" and has_run:
        scope_data = summary["run"]
    else:
        scope_data = summary

    scope_name = scope_data.get("scope", {}).get("name", scope)
    diagnosis = scope_data.get("diagnosis", {})
    run_summary = scope_data.get("run_summary", {})
    per_step = scope_data.get("per_step", [])

    sep = "-" * 56
    print(f"\n{sep}")
    print(f"  GPU Autopilot - Run Analysis")
    print(f"  Run  : {run_id}")
    print(f"  Scope: {scope_name}  ({scope_data.get('step_count', len(per_step))} steps)")
    print(sep)

    primary = diagnosis.get("primary_bottleneck")
    user_facing = diagnosis.get("user_facing_bottleneck", primary)
    confidence = diagnosis.get("confidence", {})
    step_count = scope_data.get("step_count", len(per_step))
    step_avg_ns = run_summary.get("step_time_avg_ns", 0)
    analysis_incomplete = not primary and not scope_data.get("bottlenecks") and (
        step_count == 0 or not step_avg_ns
    )
    print(f"\nVERDICT")
    if primary:
        display_label = user_facing if user_facing and user_facing != primary else primary
        print(f"  Primary Bottleneck : {display_label}")
        if user_facing and user_facing != primary:
            print(f"  Internal Signal    : {primary} (symptom)")
        print(f"  Confidence         : {confidence.get('level', 'unknown')} ({confidence.get('score', 0.0):.2f})")
        print(f"  Reason             : {confidence.get('reason', '')}")
    elif analysis_incomplete:
        print(f"  Analysis incomplete: missing or non-measurable trace data.")
        print(f"  Reason             : import raw/trace.jsonl, derived/summary.json, or profiler/profiler_trace.json before trusting the diagnosis.")
    else:
        print(f"  No bottleneck detected above threshold.")
        print(f"  Confidence         : {confidence.get('level', 'low')}: {confidence.get('reason', '')}")

    why = diagnosis.get("why", [])
    if why:
        print(f"\nEVIDENCE")
        for bullet in why:
            print(f"  - {bullet}")

    why_not = diagnosis.get("why_not_others", [])
    if why_not:
        print(f"\nSECONDARY SIGNALS")
        for bullet in why_not:
            print(f"  - {bullet}")

    print(f"\nPERFORMANCE SNAPSHOT")
    avg_gpu = run_summary.get("average_gpu_utilization_pct", 0.0)
    avg_mem = run_summary.get("average_memory_utilization_pct", 0.0)
    throughput = run_summary.get("throughput_steps_per_sec", 0.0)
    stall = run_summary.get("dominant_stall_type", "unknown")
    mem_pressure = run_summary.get("memory_pressure_peak_ratio", 0.0)

    step_avg_ms = step_avg_ns / 1_000_000 if step_avg_ns else 0.0
    print(f"  Avg GPU Utilization : {avg_gpu:.1f}%")
    print(f"  Avg Memory Util     : {avg_mem:.1f}%")
    print(f"  Peak Memory Pressure: {mem_pressure:.2f}")
    if step_avg_ms >= 0.001:
        print(f"  Avg Step Time       : {step_avg_ms:.3f} ms")
    elif step_avg_ns > 0:
        print(f"  Avg Step Time       : {step_avg_ns} ns")
    if throughput > 0:
        print(f"  Throughput          : {throughput:,.1f} steps/sec")
    print(f"  Dominant Stall      : {stall}")

    tax = scope_data.get("framework_abstraction_tax")
    if tax:
        print(f"\nFRAMEWORK ABSTRACTION TAX")
        print(f"  Score              : {tax.get('score', 0)}/100 ({tax.get('severity', 'low')})")
        contributors = tax.get("contributors", [])
        if contributors:
            print(f"  Contributors:")
            for c in contributors:
                tag = " (inferred)" if c.get("inferred") else ""
                print(f"   - {c.get('name', '')}{tag}")

    recommendations = diagnosis.get("recommendations", [])
    if recommendations:
        print(f"\nTOP RECOMMENDATIONS ({min(len(recommendations), 3)} of {len(recommendations)})")
        for i, rec in enumerate(recommendations[:3], start=1):
            priority = rec.get("priority", "low").upper()
            title = rec.get("title", rec.get("id", ""))
            effort = rec.get("effort", "?")
            risk = rec.get("risk", "?")
            score = rec.get("score", 0.0)
            print(f"\n  {i}. [{priority}] {title}")
            print(f"     Effort: {effort}  |  Risk: {risk}  |  Score: {score:.2f}")
            triggered_by = rec.get("triggered_by")
            if triggered_by:
                print(f"     Triggered by: {triggered_by}")
            why_text = rec.get("why", "")
            if why_text:
                print(f"     Why: {why_text}")
            ranked = rec.get("why_ranked", [])
            if ranked:
                print(f"     Ranked: {', '.join(ranked[:2])}")
            actions = rec.get("actions", [])
            if actions:
                print(f"     Actions:")
                for action in actions[:3]:
                    print(f"       - {action}")
            validation = rec.get("validation", [])
            if validation:
                print(f"     Validate: {validation[0]}")
    else:
        print(f"\nNo recommendations generated.")
        if not primary:
            print("  Instrument the workload with fournex for richer trace data.")

    withheld = diagnosis.get("withheld_recommendations", [])
    if withheld:
        print(f"\nWHY NOT")
        for item in withheld[:3]:
            title = item.get("title", item.get("id", "recommendation"))
            reason = item.get("reason", "")
            evidence = _format_report_evidence(item.get("evidence", {}))
            print(f"  - {title}: {reason}")
            if evidence:
                print(f"    Evidence: {evidence}")

    event_count = summary.get("event_count", scope_data.get("event_count", 0))
    print(f"\n  ({event_count} trace events analyzed)\n")


def _format_report_evidence(evidence: dict[str, Any]) -> str:
    parts = []
    for key, value in evidence.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
