from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .actions import PromotionThresholds, TrialConfig, TrialResult
from .benchmark import BenchmarkWindow, apply_benchmark_window
from .guards import check_guards, extract_metrics_from_summary
from .quality import extract_quality_metrics


class LocalTrialExecutor:
    def __init__(
        self,
        workload_command: list[str],
        job_name: str = "frx-tune",
        time_budget_s: int = 60,
        warmup_steps: int = 5,
        measure_steps: int = 20,
        repeat_count: int = 1,
        benchmark_window: BenchmarkWindow | None = None,
        sample_interval_ms: int = 1000,
        thresholds: PromotionThresholds | None = None,
        reuse_context: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> None:
        self.workload_command = workload_command
        self.job_name = job_name
        self.benchmark_window = benchmark_window or BenchmarkWindow(
            warmup_steps=warmup_steps,
            measurement_steps=measure_steps,
            repeat_count=repeat_count,
            timeout_s=time_budget_s,
        )
        self.time_budget_s = self.benchmark_window.timeout_s
        self.warmup_steps = self.benchmark_window.warmup_steps
        self.measure_steps = self.benchmark_window.measurement_steps
        self.repeat_count = self.benchmark_window.repeat_count
        self.sample_interval_ms = sample_interval_ms
        self.thresholds = thresholds or PromotionThresholds()
        self.reuse_context = reuse_context or {}
        self.verbose = verbose

    def run(self, trial_config: TrialConfig, trial_dir: Path) -> TrialResult:
        trial_dir.mkdir(parents=True, exist_ok=True)

        self._write_config(trial_config, trial_dir / "config.yaml")
        self._write_benchmark_window(trial_dir / "benchmark_window.json")

        env = self._build_env(trial_config, trial_dir)
        exit_code = self._exec_with_timeout(env, trial_dir)
        summary = self._load_or_derive_summary(trial_dir, trial_dir / "raw")

        if summary is None:
            result = TrialResult(
                config_id=trial_config.config_id,
                label=trial_config.display_label,
                exit_code=exit_code,
                throughput_steps_per_sec=0.0,
                avg_gpu_utilization_pct=0.0,
                avg_step_time_ms=0.0,
                peak_memory_ratio=0.0,
                dominant_stall="unknown",
                step_count=0,
                passed_guards=False,
                guard_failures=["no trace data - workload did not emit frx events"],
                env_vars=trial_config.env,
                quality_metrics={},
                artifacts_path=str(trial_dir),
                artifact_paths=self._artifact_paths(trial_dir),
            )
            self._write_metrics(result, trial_dir / "metrics.json")
            result.artifact_paths = self._artifact_paths(trial_dir)
            self._write_metrics(result, trial_dir / "metrics.json")
            return result

        metrics = extract_metrics_from_summary(summary)
        quality_metrics = extract_quality_metrics(summary)
        candidate = TrialResult(
            config_id=trial_config.config_id,
            label=trial_config.display_label,
            exit_code=exit_code,
            throughput_steps_per_sec=metrics["throughput_steps_per_sec"],
            avg_gpu_utilization_pct=metrics["avg_gpu_utilization_pct"],
            avg_step_time_ms=metrics["avg_step_time_ms"],
            peak_memory_ratio=metrics["peak_memory_ratio"],
            dominant_stall=metrics["dominant_stall"],
            step_count=metrics["step_count"],
            passed_guards=True,
            guard_failures=[],
            raw_summary=summary,
            env_vars=trial_config.env,
            quality_metrics=quality_metrics,
            artifacts_path=str(trial_dir),
            artifact_paths=self._artifact_paths(trial_dir),
        )

        dummy_baseline = TrialResult(
            config_id="dummy",
            label="",
            exit_code=0,
            throughput_steps_per_sec=candidate.throughput_steps_per_sec,
            avg_gpu_utilization_pct=candidate.avg_gpu_utilization_pct,
            avg_step_time_ms=candidate.avg_step_time_ms,
            peak_memory_ratio=0.0,
            dominant_stall="",
            step_count=max(candidate.step_count, self.thresholds.require_sufficient_steps),
            passed_guards=True,
            guard_failures=[],
            quality_metrics=quality_metrics,
        )
        passed, failures = check_guards(candidate, dummy_baseline, self.thresholds)
        candidate.passed_guards = passed
        candidate.guard_failures = failures
        self._write_metrics(candidate, trial_dir / "metrics.json")
        candidate.artifact_paths = self._artifact_paths(trial_dir)
        self._write_metrics(candidate, trial_dir / "metrics.json")
        return candidate

    def record_skipped(
        self,
        trial_config: TrialConfig,
        trial_dir: Path,
        reasons: list[str],
    ) -> TrialResult:
        trial_dir.mkdir(parents=True, exist_ok=True)
        self._write_config(trial_config, trial_dir / "config.yaml")
        self._write_benchmark_window(trial_dir / "benchmark_window.json")
        (trial_dir / "stdout.log").write_text("", encoding="utf-8")
        (trial_dir / "stderr.log").write_text(
            "\n".join(f"pre-run safety validation: {reason}" for reason in reasons) + "\n",
            encoding="utf-8",
        )
        result = TrialResult(
            config_id=trial_config.config_id,
            label=trial_config.display_label,
            exit_code=-3,
            throughput_steps_per_sec=0.0,
            avg_gpu_utilization_pct=0.0,
            avg_step_time_ms=0.0,
            peak_memory_ratio=0.0,
            dominant_stall="skipped",
            step_count=0,
            passed_guards=False,
            guard_failures=[f"pre-run safety validation: {reason}" for reason in reasons],
            env_vars=trial_config.env,
            quality_metrics={},
            artifacts_path=str(trial_dir),
            artifact_paths=self._artifact_paths(trial_dir),
        )
        self._write_metrics(result, trial_dir / "metrics.json")
        result.artifact_paths = self._artifact_paths(trial_dir)
        self._write_metrics(result, trial_dir / "metrics.json")
        return result

    def _build_env(self, trial_config: TrialConfig, trial_dir: Path) -> dict[str, str]:
        raw_dir = trial_dir / "raw"
        derived_dir = trial_dir / "derived"
        raw_dir.mkdir(parents=True, exist_ok=True)
        derived_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update(self.benchmark_window.env_vars())
        env.update(
            {
                "FRX_RUN_ID": trial_config.config_id,
                "FRX_JOB_NAME": self.job_name,
                "FRX_OUTPUT_DIR": str(trial_dir),
                "FRX_RAW_TRACE_PATH": str(raw_dir / "trace.jsonl"),
                "FRX_DERIVED_SUMMARY_PATH": str(derived_dir / "summary.json"),
                "FRX_AUTO_PERSIST": "1",
                "FRX_SAMPLE_INTERVAL_MS": str(self.sample_interval_ms),
            }
        )
        env.update(trial_config.env)
        return env

    def _exec_with_timeout(self, env: dict[str, str], trial_dir: Path) -> int:
        stdout_path = trial_dir / "stdout.log"
        stderr_path = trial_dir / "stderr.log"
        try:
            result = subprocess.run(
                self.workload_command,
                env=env,
                timeout=self.time_budget_s,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout_path.write_text(result.stdout or "", encoding="utf-8")
            stderr_path.write_text(result.stderr or "", encoding="utf-8")
            if self.verbose:
                if result.stdout:
                    print(result.stdout, end="")
                if result.stderr:
                    print(result.stderr, end="")
            return result.returncode
        except subprocess.TimeoutExpired as exc:
            stdout_path.write_text(_timeout_text(exc.stdout), encoding="utf-8")
            stderr_path.write_text(_timeout_text(exc.stderr), encoding="utf-8")
            if self.verbose:
                print(f"       (killed after {self.time_budget_s}s time budget)")
            return -1
        except Exception as exc:
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(f"subprocess error: {exc}\n", encoding="utf-8")
            if self.verbose:
                print(f"       (subprocess error: {exc})")
            return -2

    def _load_or_derive_summary(
        self, trial_dir: Path, raw_dir: Path
    ) -> dict[str, Any] | None:
        derived_path = trial_dir / "derived" / "summary.json"
        if derived_path.exists():
            try:
                summary = json.loads(derived_path.read_text(encoding="utf-8"))
                return self._apply_and_persist_window(summary, derived_path)
            except Exception:
                pass

        trace_path = raw_dir / "trace.jsonl"
        if not trace_path.exists():
            return None

        try:
            from ..analysis import summarize_run_with_steady_state

            events = _read_jsonl(trace_path)
            if not events:
                return None
            summary = summarize_run_with_steady_state(
                events,
                skip_first_n=self.benchmark_window.warmup_steps,
                last_k=self.benchmark_window.measurement_steps,
            )
            return self._apply_and_persist_window(summary, derived_path)
        except Exception:
            return None

    def _apply_and_persist_window(
        self,
        summary: dict[str, Any],
        derived_path: Path,
    ) -> dict[str, Any]:
        summary = apply_benchmark_window(summary, self.benchmark_window)
        derived_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _write_config(self, trial_config: TrialConfig, path: Path) -> None:
        path.write_text(self.config_text(trial_config), encoding="utf-8")

    def config_text(self, trial_config: TrialConfig) -> str:
        payload = {
            "name": trial_config.name,
            "label": trial_config.display_label,
            "risk_level": trial_config.risk_level,
            "patch": trial_config.patch,
            "env": trial_config.env,
            "workload_command": self.workload_command,
            "benchmark_window": self.benchmark_window.to_dict(),
            "sample_interval_ms": self.sample_interval_ms,
            "thresholds": asdict(self.thresholds),
            "reuse_context": self.reuse_context,
        }
        return _simple_yaml(payload)

    def _write_benchmark_window(self, path: Path) -> None:
        path.write_text(json.dumps(self.benchmark_window.to_dict(), indent=2), encoding="utf-8")

    def _write_metrics(self, result: TrialResult, path: Path) -> None:
        payload = asdict(result)
        payload.pop("raw_summary", None)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _artifact_paths(trial_dir: Path) -> dict[str, str]:
        candidates = {
            "config": trial_dir / "config.yaml",
            "metrics": trial_dir / "metrics.json",
            "benchmark_window": trial_dir / "benchmark_window.json",
            "stdout": trial_dir / "stdout.log",
            "stderr": trial_dir / "stderr.log",
            "raw_trace": trial_dir / "raw" / "trace.jsonl",
            "derived_summary": trial_dir / "derived" / "summary.json",
        }
        return {
            name: path.relative_to(trial_dir).as_posix()
            for name, path in candidates.items()
            if path.exists()
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass
    return events


def _timeout_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _simple_yaml(value: Any, indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_simple_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_simple_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
    else:
        lines.append(f"{prefix}{_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text or any(ch in text for ch in ":#\n[]{}"):
        return json.dumps(text)
    return text
