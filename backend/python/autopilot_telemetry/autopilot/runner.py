from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .actions import CandidateConfig, PromotionThresholds, TrialResult
from .guards import check_guards, extract_metrics_from_summary
from .report import TuneReport, build_report, format_report
from .tuners import generate_all_candidates


class ExperimentRunner:
    """
    Orchestrates the safe autopilot tune loop.

    Flow:
        1. Capture baseline — run workload unmodified, record metrics.
        2. Generate candidates — staged search across dataloader, batch size, AMP.
        3. Run each candidate — inject env vars, capture trace, extract metrics.
        4. Validate — run correctness guards on each trial.
        5. Pick winner — highest throughput that clears promotion thresholds.
        6. Return TuneReport.

    The workload subprocess is wrapped by the frx collect machinery so every
    trial produces a derived/summary.json we can parse for metrics.
    """

    def __init__(
        self,
        workload_command: list[str],
        job_name: str = "frx-tune",
        out_dir: str = "runs",
        max_trials: int = 12,
        safe_only: bool = True,
        time_budget_s: int = 60,
        warmup_steps: int = 5,
        measure_steps: int = 20,
        sample_interval_ms: int = 1000,
        thresholds: PromotionThresholds | None = None,
        environment: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> None:
        self.workload_command = workload_command
        self.job_name = job_name
        self.out_dir = Path(out_dir)
        self.max_trials = max_trials
        self.safe_only = safe_only
        self.time_budget_s = time_budget_s
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.sample_interval_ms = sample_interval_ms
        self.thresholds = thresholds or PromotionThresholds()
        self.environment = environment or {}
        self.verbose = verbose

    def run(self) -> TuneReport:
        tune_id = f"tune-{uuid.uuid4().hex[:8]}"
        tune_dir = self.out_dir / tune_id
        tune_dir.mkdir(parents=True, exist_ok=True)

        self._log(f"\nfrx autopilot — starting tune run {tune_id}")
        self._log(f"Workload : {' '.join(self.workload_command)}")
        self._log(f"Max trials: {self.max_trials}  |  Time budget: {self.time_budget_s}s/trial\n")

        # ── 1. Baseline ───────────────────────────────────────────────────────
        self._log("Running baseline...")
        baseline = self._run_trial(
            config_id="baseline",
            label="baseline",
            extra_env={},
            trial_dir=tune_dir / "baseline",
        )
        self._log(f"  Baseline: {baseline.throughput_steps_per_sec:.2f} steps/sec  "
                  f"(exit={baseline.exit_code}, steps={baseline.step_count})\n")

        # ── 2. Generate candidates ─────────────────────────────────────────────
        baseline_bs: int | None = None  # not yet detectable from trace; extend later
        candidates = generate_all_candidates(
            environment=self.environment,
            baseline_batch_size=baseline_bs,
            max_total=self.max_trials,
            safe_only=self.safe_only,
        )
        self._log(f"Generated {len(candidates)} candidates\n")

        # ── 3 & 4. Run + validate ──────────────────────────────────────────────
        trial_results: list[TrialResult] = []
        for i, candidate in enumerate(candidates, start=1):
            self._log(f"  [{i}/{len(candidates)}] {candidate.label} ...")
            trial_dir = tune_dir / candidate.config_id
            result = self._run_trial(
                config_id=candidate.config_id,
                label=candidate.label,
                extra_env=candidate.env_vars,
                trial_dir=trial_dir,
            )
            self._annotate_delta(result, baseline)
            tag = _result_tag(result)
            self._log(f"  {tag}  {result.label:<36} "
                      f"{result.throughput_delta:+.1%}  "
                      f"(exit={result.exit_code}, steps={result.step_count})")
            if result.guard_failures:
                for f in result.guard_failures:
                    self._log(f"       ! {f}")
            trial_results.append(result)

        # ── 5. Build report ────────────────────────────────────────────────────
        report = build_report(
            job_name=self.job_name,
            baseline=baseline,
            trials=trial_results,
            thresholds=self.thresholds,
        )

        report_path = tune_dir / "autopilot_report.json"
        report.save(report_path)
        self._log(f"\nReport saved: {report_path}")

        return report

    def _run_trial(
        self,
        config_id: str,
        label: str,
        extra_env: dict[str, str],
        trial_dir: Path,
    ) -> TrialResult:
        trial_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update({
            "FRX_TUNE_WARMUP_STEPS": str(self.warmup_steps),
            "FRX_TUNE_MAX_STEPS": str(self.warmup_steps + self.measure_steps),
        })
        env.update(extra_env)

        run_id = f"{config_id}-{uuid.uuid4().hex[:6]}"
        derived_dir = trial_dir / "derived"
        raw_dir = trial_dir / "raw"
        derived_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        env.update({
            "FRX_RUN_ID": run_id,
            "FRX_JOB_NAME": self.job_name,
            "FRX_OUTPUT_DIR": str(trial_dir),
            "FRX_RAW_TRACE_PATH": str(raw_dir / "trace.jsonl"),
            "FRX_DERIVED_SUMMARY_PATH": str(derived_dir / "summary.json"),
            "FRX_AUTO_PERSIST": "1",
            "FRX_SAMPLE_INTERVAL_MS": str(self.sample_interval_ms),
        })

        exit_code = self._exec_with_timeout(self.workload_command, env, trial_dir)

        # Try to generate derived summary from trace if not already present.
        summary = self._load_or_derive_summary(trial_dir, raw_dir)

        if summary is None:
            return TrialResult(
                config_id=config_id,
                label=label,
                exit_code=exit_code,
                throughput_steps_per_sec=0.0,
                avg_gpu_utilization_pct=0.0,
                avg_step_time_ms=0.0,
                peak_memory_ratio=0.0,
                dominant_stall="unknown",
                step_count=0,
                passed_guards=False,
                guard_failures=["no trace data — workload did not emit frx events"],
                env_vars=extra_env,
            )

        metrics = extract_metrics_from_summary(summary)
        # Construct a stub baseline for guard check (will be replaced with real baseline
        # in the delta annotation step — pass a permissive dummy here).
        dummy_baseline = TrialResult(
            config_id="dummy", label="", exit_code=0,
            throughput_steps_per_sec=metrics["throughput_steps_per_sec"],
            avg_gpu_utilization_pct=0, avg_step_time_ms=0,
            peak_memory_ratio=0, dominant_stall="", step_count=100,
            passed_guards=True, guard_failures=[],
        )
        passed, failures = check_guards(
            TrialResult(
                config_id=config_id, label=label, exit_code=exit_code,
                throughput_steps_per_sec=metrics["throughput_steps_per_sec"],
                avg_gpu_utilization_pct=metrics["avg_gpu_utilization_pct"],
                avg_step_time_ms=metrics["avg_step_time_ms"],
                peak_memory_ratio=metrics["peak_memory_ratio"],
                dominant_stall=metrics["dominant_stall"],
                step_count=metrics["step_count"],
                passed_guards=True, guard_failures=[],
            ),
            dummy_baseline,
            self.thresholds,
        )

        return TrialResult(
            config_id=config_id,
            label=label,
            exit_code=exit_code,
            throughput_steps_per_sec=metrics["throughput_steps_per_sec"],
            avg_gpu_utilization_pct=metrics["avg_gpu_utilization_pct"],
            avg_step_time_ms=metrics["avg_step_time_ms"],
            peak_memory_ratio=metrics["peak_memory_ratio"],
            dominant_stall=metrics["dominant_stall"],
            step_count=metrics["step_count"],
            passed_guards=passed,
            guard_failures=failures,
            raw_summary=summary,
            env_vars=extra_env,
        )

    def _exec_with_timeout(
        self, command: list[str], env: dict[str, str], cwd: Path
    ) -> int:
        try:
            result = subprocess.run(
                command,
                env=env,
                timeout=self.time_budget_s,
                capture_output=not self.verbose,
                text=True,
            )
            return result.returncode
        except subprocess.TimeoutExpired:
            self._log(f"       (killed after {self.time_budget_s}s time budget)")
            return -1
        except Exception as exc:
            self._log(f"       (subprocess error: {exc})")
            return -2

    def _load_or_derive_summary(
        self, trial_dir: Path, raw_dir: Path
    ) -> dict[str, Any] | None:
        derived_path = trial_dir / "derived" / "summary.json"
        if derived_path.exists():
            try:
                return json.loads(derived_path.read_text(encoding="utf-8"))
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
            summary = summarize_run_with_steady_state(events)
            derived_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary
        except Exception:
            return None

    @staticmethod
    def _annotate_delta(result: TrialResult, baseline: TrialResult) -> None:
        if baseline.throughput_steps_per_sec > 0 and result.throughput_steps_per_sec > 0:
            result.throughput_delta = (
                (result.throughput_steps_per_sec - baseline.throughput_steps_per_sec)
                / baseline.throughput_steps_per_sec
            )
        # Re-run guards with actual baseline now that we have it.
        from .guards import check_guards as _check
        passed, failures = _check(result, baseline)
        result.passed_guards = passed
        result.guard_failures = failures

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        pass
    return events


def _result_tag(result: TrialResult) -> str:
    if not result.is_viable:
        return "[FAIL]"
    if result.throughput_delta >= 0.08:
        return "[PASS]"
    return "[    ]"
