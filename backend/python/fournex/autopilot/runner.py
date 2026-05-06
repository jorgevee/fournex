from __future__ import annotations

import uuid
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .actions import CandidateConfig, PromotionThresholds, TrialConfig, TrialResult
from .benchmark import BenchmarkWindow
from .comparison import aggregate_repeats, annotate_noise_comparison
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
        race_enabled: bool = True,
        race_warmup_steps: int = 1,
        race_measure_steps: int = 5,
        race_promote_count: int = 3,
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
        self.race_enabled = race_enabled
        self.race_warmup_steps = race_warmup_steps
        self.race_measure_steps = race_measure_steps
        self.race_promote_count = race_promote_count
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
        baseline = self._run_repeated_trial(
            executor,
            TrialConfig(name="baseline", label="baseline", patch={"baseline": True}),
            tune_dir / "baseline",
        )
        self._log(
            f"  Baseline: {baseline.throughput_steps_per_sec:.2f} steps/sec  "
            f"(exit={baseline.exit_code}, steps={baseline.step_count})\n"
        )

        baseline_bs = _infer_batch_size_from_summary(baseline.raw_summary)
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
        if baseline_bs:
            self._log(f"Detected baseline batch size: {baseline_bs}\n")

        candidates_to_benchmark = candidates
        if self._should_run_race(candidates):
            race_results, candidates_to_benchmark = self._run_race_stage(
                candidates,
                baseline=baseline,
                tune_dir=tune_dir,
            )
            trial_results.extend(race_results)

        for i, candidate in enumerate(candidates_to_benchmark, start=1):
            result = self._run_candidate(
                executor,
                candidate,
                baseline=baseline,
                trial_dir=tune_dir / candidate.config_id,
                index=i,
                total=len(candidates_to_benchmark),
                stage="full",
            )
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

    def _build_race_executor(self) -> LocalTrialExecutor:
        window = self._race_window()
        return LocalTrialExecutor(
            workload_command=self.workload_command,
            job_name=self.job_name,
            benchmark_window=window,
            sample_interval_ms=self.sample_interval_ms,
            thresholds=self.thresholds,
            verbose=self.verbose,
        )

    def _race_window(self) -> BenchmarkWindow:
        warmup_steps = min(self.benchmark_window.warmup_steps, max(0, self.race_warmup_steps))
        measure_steps = min(self.benchmark_window.measurement_steps, max(1, self.race_measure_steps))
        timeout_s = min(self.benchmark_window.timeout_s, max(10, int(self.benchmark_window.timeout_s * 0.5)))
        return BenchmarkWindow(
            warmup_steps=warmup_steps,
            measurement_steps=measure_steps,
            repeat_count=1,
            timeout_s=timeout_s,
        )

    def _should_run_race(self, candidates: list[CandidateConfig]) -> bool:
        if not self.race_enabled:
            return False
        if self.repeat_count > 1:
            return False
        if self.race_promote_count <= 0:
            return False
        if len(candidates) <= self.race_promote_count:
            return False
        if self.benchmark_window.measurement_steps <= max(1, self.race_measure_steps):
            return False
        return True

    def _run_race_stage(
        self,
        candidates: list[CandidateConfig],
        *,
        baseline: TrialResult,
        tune_dir: Path,
    ) -> tuple[list[TrialResult], list[CandidateConfig]]:
        race_executor = self._build_race_executor()
        race_dir = tune_dir / "race"
        self._log(
            "Running quick race stage "
            f"({self._race_window().warmup_steps} warmup + "
            f"{self._race_window().measurement_steps} measure steps)..."
        )

        pairs: list[tuple[CandidateConfig, TrialResult]] = []
        for i, candidate in enumerate(candidates, start=1):
            result = self._run_candidate(
                race_executor,
                candidate,
                baseline=baseline,
                trial_dir=race_dir / candidate.config_id,
                index=i,
                total=len(candidates),
                stage="race",
            )
            pairs.append((candidate, result))

        ranked = sorted(
            pairs,
            key=lambda item: (
                item[1].exit_code == 0,
                item[1].passed_guards,
                item[1].throughput_delta,
                item[1].throughput_steps_per_sec,
            ),
            reverse=True,
        )
        finalists = [
            candidate
            for candidate, result in ranked
            if result.exit_code == 0 and result.passed_guards and result.throughput_steps_per_sec > 0
        ][: self.race_promote_count]
        finalist_ids = {candidate.config_id for candidate in finalists}

        for candidate, result in pairs:
            result.eligible_for_promotion = False
            result.benchmark_stage = "race"
            if candidate.config_id in finalist_ids:
                result.screening_decision = "promoted to full benchmark"
            elif result.exit_code == 0 and result.passed_guards:
                result.screening_decision = "screened out by quicker candidates"
            else:
                result.screening_decision = "screened out by race guard failure"
            self._rewrite_result_metrics(result)

        self._log(f"Quick race promoted {len(finalists)} of {len(candidates)} candidates\n")
        return [result for _, result in pairs], finalists

    def _run_candidate(
        self,
        executor: LocalTrialExecutor,
        candidate: CandidateConfig,
        *,
        baseline: TrialResult,
        trial_dir: Path,
        index: int,
        total: int,
        stage: str,
    ) -> TrialResult:
        trial_config = TrialConfig.from_candidate(candidate)
        prefix = "race" if stage == "race" else "full"
        self._log(f"  [{index}/{total}] {prefix}: {trial_config.display_label} ...")
        validation = validate_candidate(
            candidate,
            baseline=baseline,
            environment=self.environment,
            policy=self.safety_policy,
        )
        if validation.passed:
            result = self._run_repeated_trial(executor, trial_config, trial_dir)
        else:
            result = executor.record_skipped(
                trial_config,
                trial_dir,
                validation.reasons,
            )
        self._annotate_delta(result, baseline)
        result.benchmark_stage = stage
        result.eligible_for_promotion = stage == "full"
        self._rewrite_result_metrics(result)
        tag = _result_tag(result, self.thresholds)
        self._log(
            f"  {tag}  {result.label:<36} "
            f"{result.throughput_delta:+.1%}  "
            f"(exit={result.exit_code}, steps={result.step_count})"
        )
        if result.guard_failures:
            for failure in result.guard_failures:
                self._log(f"       ! {failure}")
        return result

    def _run_trial(
        self,
        executor: LocalTrialExecutor,
        trial_config: TrialConfig,
        trial_dir: Path,
    ) -> TrialResult:
        return executor.run(trial_config, trial_dir)

    def _run_repeated_trial(
        self,
        executor: LocalTrialExecutor,
        trial_config: TrialConfig,
        trial_dir: Path,
    ) -> TrialResult:
        if self.repeat_count <= 1:
            return self._run_trial(executor, trial_config, trial_dir)

        trial_dir.mkdir(parents=True, exist_ok=True)
        repeats: list[TrialResult] = []
        for repeat_index in range(1, self.repeat_count + 1):
            repeat_dir = trial_dir / f"repeat_{repeat_index:03d}"
            repeats.append(self._run_trial(executor, trial_config, repeat_dir))

        aggregate = aggregate_repeats(
            config_id=trial_config.config_id,
            label=trial_config.display_label,
            repeats=repeats,
            artifacts_path=str(trial_dir),
            env_vars=trial_config.env,
        )
        self._write_aggregate_metrics(aggregate, trial_dir)
        return aggregate

    def _annotate_delta(self, result: TrialResult, baseline: TrialResult) -> None:
        annotate_noise_comparison(baseline, result, self.thresholds)
        if result.exit_code == -3:
            return
        existing_failures = list(result.guard_failures)
        existing_passed = result.passed_guards
        passed, failures = check_guards(result, baseline, self.thresholds)
        quality_passed, quality_failures = check_quality_regression(
            baseline,
            result,
            self.quality_policy,
        )
        passed = existing_passed and passed and quality_passed and not existing_failures
        failures = existing_failures + failures + quality_failures
        result.passed_guards = passed
        result.guard_failures = failures

    @staticmethod
    def _write_aggregate_metrics(result: TrialResult, trial_dir: Path) -> None:
        payload = asdict(result)
        payload.pop("raw_summary", None)
        (trial_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _rewrite_result_metrics(result: TrialResult) -> None:
        if not result.artifacts_path:
            return
        path = Path(result.artifacts_path) / "metrics.json"
        if not path.parent.exists():
            return
        payload = asdict(result)
        payload.pop("raw_summary", None)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


def _result_tag(result: TrialResult, thresholds: PromotionThresholds) -> str:
    if result.benchmark_stage == "race":
        if result.exit_code == 0 and result.passed_guards and result.throughput_steps_per_sec > 0:
            return "[RACE]"
        return "[FAIL]"
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


def _infer_batch_size_from_summary(summary: dict[str, Any]) -> int | None:
    if not summary:
        return None

    sizes: list[int] = []
    for scope in _summary_scopes(summary):
        for step in scope.get("per_step", []) or []:
            raw = step.get("batch_size")
            try:
                batch_size = int(raw)
            except (TypeError, ValueError):
                continue
            if batch_size > 0:
                sizes.append(batch_size)

    if not sizes:
        return None

    counts: dict[int, int] = {}
    for size in sizes:
        counts[size] = counts.get(size, 0) + 1
    return max(counts, key=lambda size: (counts[size], size))


def _summary_scopes(summary: dict[str, Any]) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    for key in ("measurement_window", "steady_state", "run"):
        value = summary.get(key)
        if isinstance(value, dict):
            scopes.append(value)
    if summary.get("per_step"):
        scopes.append(summary)
    return scopes
