from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .actions import CandidateConfig, PromotionThresholds, TrialResult


@dataclass
class TuneReport:
    job_name: str
    baseline: TrialResult
    trials: list[TrialResult]
    winner: TrialResult | None
    thresholds: PromotionThresholds
    total_trials: int
    promoted_count: int

    @property
    def improved(self) -> bool:
        return self.winner is not None and self.winner.config_id != "baseline"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_to_dict(self), indent=2), encoding="utf-8")


def select_winner(
    baseline: TrialResult,
    trials: list[TrialResult],
    thresholds: PromotionThresholds | None = None,
) -> tuple[TrialResult | None, list[TrialResult]]:
    """
    Returns (winner, promoted_candidates).

    Winner must be viable + clear min_speedup threshold.
    Among viable candidates, pick the one with the highest throughput.
    Returns (baseline, []) if nothing improves enough.
    """
    t = thresholds or PromotionThresholds()
    promoted = [
        r for r in trials
        if r.eligible_for_promotion and r.is_viable and r.throughput_delta >= t.min_speedup
    ]
    if not promoted:
        return None, []
    promoted.sort(key=lambda r: r.throughput_steps_per_sec, reverse=True)
    return promoted[0], promoted


def build_report(
    job_name: str,
    baseline: TrialResult,
    trials: list[TrialResult],
    thresholds: PromotionThresholds | None = None,
) -> TuneReport:
    t = thresholds or PromotionThresholds()
    winner, promoted = select_winner(baseline, trials, t)
    return TuneReport(
        job_name=job_name,
        baseline=baseline,
        trials=trials,
        winner=winner,
        thresholds=t,
        total_trials=len(trials),
        promoted_count=len(promoted),
    )


def format_report(report: TuneReport) -> str:
    lines: list[str] = []
    sep = "─" * 58

    lines.append(f"\n{sep}")
    lines.append("  frx autopilot — Tune Report")
    lines.append(f"  Job    : {report.job_name}")
    lines.append(f"  Trials : {report.total_trials} candidates + baseline")
    lines.append(sep)

    b = report.baseline
    lines.append(f"\nBASELINE")
    lines.append(f"  Throughput   : {b.throughput_steps_per_sec:,.2f} steps/sec")
    lines.append(f"  Avg step     : {b.avg_step_time_ms:.1f} ms")
    lines.append(f"  GPU util     : {b.avg_gpu_utilization_pct:.1f}%")
    lines.append(f"  Dominant stall: {b.dominant_stall}")
    if b.repeat_count > 1:
        lines.append(f"  Repeats      : {b.repeat_count}")

    lines.append(f"\nTRIAL RESULTS")
    screened = [r for r in report.trials if r.benchmark_stage == "race"]
    viable = [r for r in report.trials if r.benchmark_stage != "race" and r.is_viable]
    failed = [r for r in report.trials if r.benchmark_stage != "race" and not r.is_viable]

    for r in sorted(viable, key=lambda x: x.throughput_delta, reverse=True):
        delta_str = f"{r.throughput_delta:+.1%}"
        marker = " ✓" if r.throughput_delta >= report.thresholds.min_speedup else ""
        confidence = f"  confidence={r.confidence_label}" if r.repeat_count > 1 else ""
        lines.append(f"  {r.label:<36} {delta_str}{marker}{confidence}")

    for r in sorted(screened, key=lambda x: x.throughput_delta, reverse=True):
        delta_str = f"{r.throughput_delta:+.1%}"
        decision = r.screening_decision or "quick race result"
        lines.append(f"  {r.label:<36} [RACE] {delta_str}  {decision}")

    for r in failed:
        reasons = "; ".join(r.guard_failures[:2])
        lines.append(f"  {r.label:<36} [FAILED] {reasons}")

    if report.winner and report.improved:
        w = report.winner
        lines.append(f"\nWINNER")
        lines.append(f"  Config       : {w.label}")
        lines.append(f"  Throughput   : {w.throughput_steps_per_sec:,.2f} steps/sec  "
                     f"({w.throughput_delta:+.1%} vs baseline)")
        lines.append(f"  Avg step     : {w.avg_step_time_ms:.1f} ms")
        lines.append(f"  GPU util     : {w.avg_gpu_utilization_pct:.1f}%")
        if w.repeat_count > 1:
            lines.append(f"  Confidence   : {w.confidence_label}")
            lines.append(f"  Noise band   : +/-{w.noise_band:.1%}")
        lines.append(f"\nENV VARS TO APPLY")
        for k, v in w.env_vars.items():
            lines.append(f"  {k}={v}")
        lines.append(f"\nApplied: No — recommendation only")
        lines.append(f"To apply: set the env vars above before launching your workload.")
        lines.append(f"To persist: write them to a .env file or your launcher config.")
    else:
        lines.append(f"\nNo candidate improved throughput by ≥{report.thresholds.min_speedup:.0%}.")
        lines.append(f"Recommendation: keep current config.")
        if not viable:
            lines.append(f"Note: all {len(failed)} trial(s) failed guards. "
                         f"Instrument your workload with autopilot_telemetry for richer traces.")

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


def _to_dict(report: TuneReport) -> dict[str, Any]:
    def trial_dict(r: TrialResult) -> dict[str, Any]:
        d = asdict(r)
        d.pop("raw_summary", None)
        return d

    return {
        "job_name": report.job_name,
        "total_trials": report.total_trials,
        "promoted_count": report.promoted_count,
        "improved": report.improved,
        "baseline": trial_dict(report.baseline),
        "winner": trial_dict(report.winner) if report.winner else None,
        "trials": [trial_dict(r) for r in report.trials],
        "thresholds": asdict(report.thresholds),
    }
