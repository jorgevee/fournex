from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from typing import Any


BUNDLE_SCHEMA_VERSION = "0.1.0"
EXPECTED_ARTIFACTS = (
    "metadata.json",
    "run_config.yaml",
    "gpu_metrics.csv",
    "optional_logs.txt",
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "collect":
        args.workload_command = _normalize_workload_command(args.workload_command)
        args.workload_command = _resolve_workload_command(args.workload_command)
        if not args.workload_command:
            parser.error("collect requires a workload command after --")
        return collect(args)
    elif args.command == "analyze":
        return analyze(args)
    elif args.command == "doctor":
        return doctor(args)
    elif args.command == "smoke-test":
        return smoke_test(args)
    elif args.command == "tune":
        args.workload_command = _normalize_workload_command(args.workload_command)
        args.workload_command = _resolve_workload_command(args.workload_command)
        if not args.workload_command:
            parser.error("tune requires a workload command after --")
        return tune(args)
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
    _generate_derived_summary_from_trace(run_dir, warnings)
    _generate_derived_summary_from_profiler_bundle(run_dir, warnings)
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
        verbose=True,
    )
    report = runner.run()
    print(format_report(report))
    return 0 if report.improved else 1


def _build_parser() -> argparse.ArgumentParser:
    import sys
    _stem = Path(sys.argv[0]).stem.lower().replace(".exe", "")
    prog = "fournex" if "fournex" in _stem else "frx"
    parser = argparse.ArgumentParser(prog=prog)
    subparsers = parser.add_subparsers(dest="command")

    collect_parser = subparsers.add_parser("collect", help="run a workload and package a run bundle")
    collect_parser.add_argument("--name", default=None, help="human-readable job name")
    collect_parser.add_argument("--out", default="runs", help="output directory for run folders")
    collect_parser.add_argument("--config", default=None, help="optional user run_config.yaml to merge")
    collect_parser.add_argument("--sample-interval-ms", type=int, default=1000)
    collect_parser.add_argument("--run-id", default=None)
    collect_parser.add_argument("--no-zip", action="store_true")
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

    analyze_parser = subparsers.add_parser("analyze", help="analyze a collected run bundle and print diagnosis")
    analyze_parser.add_argument("run_path", help="path to a run directory (e.g. runs/run-abc123)")
    analyze_parser.add_argument(
        "--scope",
        choices=["run", "steady_state", "auto"],
        default="auto",
        help="which analysis scope to report (default: steady_state when available)",
    )
    analyze_parser.add_argument("--json", dest="output_json", action="store_true", help="output raw JSON")

    subparsers.add_parser("doctor", help="check environment for frx requirements")

    subparsers.add_parser("smoke-test", help="run a synthetic workload and verify end-to-end bundle generation")

    tune_parser = subparsers.add_parser(
        "tune",
        help="run safe autopilot: sweep configs and recommend the fastest one",
    )
    tune_parser.add_argument("--name", default="frx-tune", help="job name")
    tune_parser.add_argument("--out", default="runs", help="output directory")
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
        env["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        env["framework"] = "unknown"
        env["framework_detection_error"] = str(exc)
    return env


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
            except Exception as exc:
                warnings.append(f"nvidia-smi sampling failed: {exc}")
                return
            stop.wait(max(interval_ms, 100) / 1000.0)


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


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def analyze(args: argparse.Namespace) -> int:
    run_path = Path(args.run_path)

    if run_path.is_file() and run_path.suffix == ".zip":
        print("Error: zip analysis not yet supported. Unzip the bundle first.", file=sys.stderr)
        return 1
    if not run_path.is_dir():
        print(f"Error: run directory not found: {run_path}", file=sys.stderr)
        return 1

    summary = _load_or_generate_summary(run_path)
    if summary is None:
        print("No trace data found in bundle. Cannot generate analysis.", file=sys.stderr)
        print(f"Expected: {run_path / 'raw' / 'trace.jsonl'} or {run_path / 'derived' / 'summary.json'}", file=sys.stderr)
        return 1

    if args.output_json:
        print(json.dumps(summary, indent=2))
        return 0

    _print_analysis_report(run_path, summary, scope=args.scope)
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


def _load_or_generate_summary(run_dir: Path) -> dict[str, Any] | None:
    derived_path = run_dir / "derived" / "summary.json"
    raw_trace_path = run_dir / "raw" / "trace.jsonl"

    if derived_path.exists():
        try:
            return json.loads(derived_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if raw_trace_path.exists():
        return _generate_summary_from_trace(raw_trace_path)

    events = _events_from_profiler_bundle(run_dir, [])
    if events:
        try:
            from .analysis import summarize_run_with_steady_state
            return summarize_run_with_steady_state(events)
        except Exception:
            pass

    return None


def _generate_summary_from_trace(trace_path: Path) -> dict[str, Any] | None:
    try:
        from .analysis import summarize_run_with_steady_state
    except ImportError as exc:
        print(f"Warning: analysis module unavailable: {exc}", file=sys.stderr)
        return None

    events = _read_jsonl_events(trace_path)
    if not events:
        return None
    return summarize_run_with_steady_state(events)


def _generate_derived_summary_from_trace(run_dir: Path, warnings: list[str]) -> None:
    derived_path = run_dir / "derived" / "summary.json"
    raw_trace_path = run_dir / "raw" / "trace.jsonl"

    if derived_path.exists():
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
        warnings.append("Raw trace is empty; derived/summary.json was not generated.")
        return

    summary = summarize_run_with_steady_state(events)
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


def _generate_derived_summary_from_profiler_bundle(run_dir: Path, warnings: list[str]) -> None:
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

    summary = summarize_run_with_steady_state(events)
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
            why_text = rec.get("why", "")
            if why_text:
                print(f"     {why_text}")
            actions = rec.get("actions", [])
            if actions:
                print(f"     Actions:")
                for action in actions[:3]:
                    print(f"       - {action}")
    else:
        print(f"\nNo recommendations generated.")
        if not primary:
            print("  Instrument the workload with fournex for richer trace data.")

    event_count = summary.get("event_count", scope_data.get("event_count", 0))
    print(f"\n  ({event_count} trace events analyzed)\n")


if __name__ == "__main__":
    raise SystemExit(main())
