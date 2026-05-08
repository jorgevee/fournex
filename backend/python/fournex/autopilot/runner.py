from __future__ import annotations

import uuid
import json
from dataclasses import asdict, fields
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
        resume_dir: str | None = None,
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
        self.resume_dir = Path(resume_dir) if resume_dir else None
        self.reuse_artifacts = self.resume_dir is not None
        self.verbose = verbose

    def run(self) -> TuneReport:
        if self.resume_dir is not None:
            tune_dir = self.resume_dir
            tune_id = tune_dir.name
        else:
            tune_id = f"tune-{uuid.uuid4().hex[:8]}"
            tune_dir = self.out_dir / tune_id
        tune_dir.mkdir(parents=True, exist_ok=True)
        executor = self._build_executor()

        action = "resuming" if self.reuse_artifacts else "starting"
        self._log(f"\nfrx autopilot - {action} tune run {tune_id}")
        self._log(f"Workload : {' '.join(self.workload_command)}")
        self._log(f"Max trials: {self.max_trials}  |  Time budget: {self.time_budget_s}s/trial\n")

        self._log("Running baseline...")
        baseline, baseline_reused = self._run_repeated_trial(
            executor,
            TrialConfig(name="baseline", label="baseline", patch={"baseline": True}),
            tune_dir / "baseline",
        )
        if baseline_reused:
            self._log("  Baseline: reused existing artifacts")
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
            reuse_context=self._reuse_context(),
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
            reuse_context=self._reuse_context(),
            verbose=self.verbose,
        )

    def _reuse_context(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "quality_policy": asdict(self.quality_policy),
        }

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
            result, reused = self._run_repeated_trial(executor, trial_config, trial_dir)
        else:
            result = executor.record_skipped(
                trial_config,
                trial_dir,
                validation.reasons,
            )
            reused = False
        if reused:
            self._log("       reused existing artifacts")
            self._reset_cached_comparison_state(result)
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
    ) -> tuple[TrialResult, bool]:
        cached = self._load_reusable_result(executor, trial_config, trial_dir)
        if cached is not None:
            return cached, True
        return executor.run(trial_config, trial_dir), False

    def _run_repeated_trial(
        self,
        executor: LocalTrialExecutor,
        trial_config: TrialConfig,
        trial_dir: Path,
    ) -> tuple[TrialResult, bool]:
        cached = self._load_reusable_result(executor, trial_config, trial_dir)
        if cached is not None:
            return cached, True

        if self.repeat_count <= 1:
            return self._run_trial(executor, trial_config, trial_dir)

        trial_dir.mkdir(parents=True, exist_ok=True)
        repeats: list[TrialResult] = []
        reused_repeats = 0
        for repeat_index in range(1, self.repeat_count + 1):
            repeat_dir = trial_dir / f"repeat_{repeat_index:03d}"
            repeat_result, reused = self._run_trial(executor, trial_config, repeat_dir)
            repeats.append(repeat_result)
            reused_repeats += int(reused)

        aggregate = aggregate_repeats(
            config_id=trial_config.config_id,
            label=trial_config.display_label,
            repeats=repeats,
            artifacts_path=str(trial_dir),
            env_vars=trial_config.env,
        )
        self._write_aggregate_metrics(aggregate, trial_dir)
        return aggregate, reused_repeats == self.repeat_count

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

    def _load_reusable_result(
        self,
        executor: LocalTrialExecutor,
        trial_config: TrialConfig,
        trial_dir: Path,
    ) -> TrialResult | None:
        if not self.reuse_artifacts:
            return None

        config_path = trial_dir / "config.yaml"
        benchmark_path = trial_dir / "benchmark_window.json"
        metrics_path = trial_dir / "metrics.json"
        if not config_path.exists() or not benchmark_path.exists() or not metrics_path.exists():
            return None

        try:
            if config_path.read_text(encoding="utf-8") != executor.config_text(trial_config):
                return None
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            if benchmark != executor.benchmark_window.to_dict():
                return None
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if payload.get("exit_code") == -3:
                return None
        except Exception:
            return None

        result = _trial_result_from_payload(payload)
        result.artifacts_path = str(trial_dir)
        result.artifact_paths = executor._artifact_paths(trial_dir)
        summary_path = trial_dir / "derived" / "summary.json"
        if summary_path.exists():
            try:
                result.raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                result.raw_summary = {}
        return result

    @staticmethod
    def _reset_cached_comparison_state(result: TrialResult) -> None:
        result.passed_guards = True
        result.guard_failures = []
        result.comparison_notes = []
        result.throughput_delta = 0.0

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


def _trial_result_from_payload(payload: dict[str, Any]) -> TrialResult:
    names = {field.name for field in fields(TrialResult)}
    values = {name: payload[name] for name in names if name in payload}
    return TrialResult(**values)


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
