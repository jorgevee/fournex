from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from .actions import PromotionThresholds, TrialConfig, TrialResult
from .benchmark import BenchmarkWindow
from .guards import check_guards
from .local_executor import LocalTrialExecutor
from .quality import QualityPolicy, check_quality_regression
from .report import TuneReport, build_report, format_report
from .safety import SafetyPolicy, validate_candidate
from .tuners import generate_all_candidates


class ExperimentRunner:
    """
    Orchestrates the safe autopilot tune loop.

    Flow:
        1. Capture baseline by running the workload unmodified.
        2. Generate candidates across dataloader, batch size, and AMP knobs.
        3. Run each candidate through the local trial executor.
        4. Validate each trial with promotion guards.
        5. Pick the fastest candidate that clears the thresholds.
        6. Persist JSON and Markdown reports.
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
        repeat_count: int = 1,
        benchmark_window: BenchmarkWindow | None = None,
        sample_interval_ms: int = 1000,
        thresholds: PromotionThresholds | None = None,
        environment: dict[str, Any] | None = None,
        bottleneck_diagnosis: str | dict[str, Any] | None = None,
        safety_policy: SafetyPolicy | None = None,
        quality_policy: QualityPolicy | None = None,
        verbose: bool = True,
    ) -> None:
        self.workload_command = workload_command
        self.job_name = job_name
        self.out_dir = Path(out_dir)
        self.max_trials = max_trials
        self.safe_only = safe_only
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
        self.environment = environment or {}
        self.bottleneck_diagnosis = bottleneck_diagnosis
        self.safety_policy = safety_policy or SafetyPolicy()
        self.quality_policy = quality_policy or QualityPolicy()
        self.verbose = verbose

    def run(self) -> TuneReport:
        tune_id = f"tune-{uuid.uuid4().hex[:8]}"
        tune_dir = self.out_dir / tune_id
        tune_dir.mkdir(parents=True, exist_ok=True)
        executor = self._build_executor()

        self._log(f"\nfrx autopilot - starting tune run {tune_id}")
        self._log(f"Workload : {' '.join(self.workload_command)}")
        self._log(f"Max trials: {self.max_trials}  |  Time budget: {self.time_budget_s}s/trial\n")

        self._log("Running baseline...")
        baseline = self._run_trial(
            executor,
            TrialConfig(name="baseline", label="baseline", patch={"baseline": True}),
            tune_dir / "baseline",
        )
        self._log(
            f"  Baseline: {baseline.throughput_steps_per_sec:.2f} steps/sec  "
            f"(exit={baseline.exit_code}, steps={baseline.step_count})\n"
        )

        baseline_bs: int | None = None
        bottleneck = self.bottleneck_diagnosis or _diagnosis_from_summary(baseline.raw_summary)
        candidates = generate_all_candidates(
            environment=self.environment,
            baseline_batch_size=baseline_bs,
            max_total=self.max_trials,
            safe_only=self.safe_only,
            bottleneck_diagnosis=bottleneck,
        )
        self._log(f"Generated {len(candidates)} candidates\n")

        trial_results: list[TrialResult] = []
        for i, candidate in enumerate(candidates, start=1):
            trial_config = TrialConfig.from_candidate(candidate)
            self._log(f"  [{i}/{len(candidates)}] {trial_config.display_label} ...")
            validation = validate_candidate(
                candidate,
                baseline=baseline,
                environment=self.environment,
                policy=self.safety_policy,
            )
            if validation.passed:
                result = self._run_trial(executor, trial_config, tune_dir / trial_config.config_id)
            else:
                result = executor.record_skipped(
                    trial_config,
                    tune_dir / trial_config.config_id,
                    validation.reasons,
                )
            self._annotate_delta(result, baseline)
            tag = _result_tag(result, self.thresholds)
            self._log(
                f"  {tag}  {result.label:<36} "
                f"{result.throughput_delta:+.1%}  "
                f"(exit={result.exit_code}, steps={result.step_count})"
            )
            if result.guard_failures:
                for failure in result.guard_failures:
                    self._log(f"       ! {failure}")
            trial_results.append(result)

        report = build_report(
            job_name=self.job_name,
            baseline=baseline,
            trials=trial_results,
            thresholds=self.thresholds,
        )

        report_path = tune_dir / "autopilot_report.json"
        report.save(report_path)
        markdown_path = tune_dir / "report.md"
        markdown_path.write_text(format_report(report), encoding="utf-8")
        self._log(f"\nReport saved: {report_path}")
        self._log(f"Markdown report saved: {markdown_path}")

        return report

    def _build_executor(self) -> LocalTrialExecutor:
        return LocalTrialExecutor(
            workload_command=self.workload_command,
            job_name=self.job_name,
            benchmark_window=self.benchmark_window,
            sample_interval_ms=self.sample_interval_ms,
            thresholds=self.thresholds,
            verbose=self.verbose,
        )

    def _run_trial(
        self,
        executor: LocalTrialExecutor,
        trial_config: TrialConfig,
        trial_dir: Path,
    ) -> TrialResult:
        return executor.run(trial_config, trial_dir)

    def _annotate_delta(self, result: TrialResult, baseline: TrialResult) -> None:
        if baseline.throughput_steps_per_sec > 0 and result.throughput_steps_per_sec > 0:
            result.throughput_delta = (
                (result.throughput_steps_per_sec - baseline.throughput_steps_per_sec)
                / baseline.throughput_steps_per_sec
            )
        if result.exit_code == -3:
            return
        passed, failures = check_guards(result, baseline, self.thresholds)
        quality_passed, quality_failures = check_quality_regression(
            baseline,
            result,
            self.quality_policy,
        )
        passed = passed and quality_passed
        failures.extend(quality_failures)
        result.passed_guards = passed
        result.guard_failures = failures

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


def _result_tag(result: TrialResult, thresholds: PromotionThresholds) -> str:
    if not result.is_viable:
        return "[FAIL]"
    if result.throughput_delta >= thresholds.min_speedup:
        return "[PASS]"
    return "[    ]"


def _diagnosis_from_summary(summary: dict[str, Any]) -> dict[str, Any] | None:
    if not summary:
        return None
    if "steady_state" in summary and isinstance(summary["steady_state"], dict):
        steady_state = summary["steady_state"]
        if isinstance(steady_state.get("diagnosis"), dict):
            return steady_state
    if "diagnosis" in summary and isinstance(summary["diagnosis"], dict):
        return summary
    if "run" in summary and isinstance(summary["run"], dict):
        run = summary["run"]
        if isinstance(run.get("diagnosis"), dict):
            return run
    return None
