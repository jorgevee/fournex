from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .actions import TrialResult


@dataclass
class QualityPolicy:
    max_final_loss_regression: float = 0.05
    max_loss_divergence: float = 0.50
    output_abs_tolerance: float = 0.005
    require_finite_loss: bool = True


def extract_quality_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    scope = summary.get("measurement_window") or summary.get("steady_state") or summary.get("run") or summary
    run_summary = scope.get("run_summary", {})
    per_step = scope.get("per_step", [])

    losses = [_to_float(step.get("loss")) for step in per_step if isinstance(step, dict)]
    finite_losses = [loss for loss in losses if loss is not None and math.isfinite(loss)]
    nan_count = sum(1 for loss in losses if loss is not None and math.isnan(loss))
    inf_count = sum(1 for loss in losses if loss is not None and math.isinf(loss))

    metrics: dict[str, Any] = {
        "loss_count": len(finite_losses),
        "nan_count": int(run_summary.get("nan_count", 0) or 0) + nan_count,
        "inf_count": int(run_summary.get("inf_count", 0) or 0) + inf_count,
    }

    if finite_losses:
        metrics.update(
            {
                "initial_loss": finite_losses[0],
                "final_loss": finite_losses[-1],
                "min_loss": min(finite_losses),
                "max_loss": max(finite_losses),
                "loss_slope": finite_losses[-1] - finite_losses[0],
            }
        )

    for key in (
        "initial_loss",
        "final_loss",
        "validation_loss",
        "accuracy",
        "perplexity",
        "gradient_norm",
        "parameter_norm",
        "max_output_abs_diff",
        "max_output_rel_diff",
    ):
        value = _to_float(run_summary.get(key))
        if value is not None:
            metrics[key] = value

    quality = scope.get("quality")
    if isinstance(quality, dict):
        for key, value in quality.items():
            number = _to_float(value)
            metrics[key] = number if number is not None else value

    return metrics


def check_quality_regression(
    baseline: TrialResult,
    trial: TrialResult,
    policy: QualityPolicy | None = None,
) -> tuple[bool, list[str]]:
    q = policy or QualityPolicy()
    failures: list[str] = []
    b = baseline.quality_metrics or {}
    t = trial.quality_metrics or {}

    nan_count = int(t.get("nan_count", 0) or 0)
    inf_count = int(t.get("inf_count", 0) or 0)
    if q.require_finite_loss and nan_count > 0:
        failures.append(f"quality regression: detected {nan_count} NaN loss value(s)")
    if q.require_finite_loss and inf_count > 0:
        failures.append(f"quality regression: detected {inf_count} Inf loss value(s)")

    baseline_final = _to_float(b.get("final_loss"))
    trial_final = _to_float(t.get("final_loss"))
    if baseline_final is not None and trial_final is not None and baseline_final > 0:
        max_allowed = baseline_final * (1.0 + q.max_final_loss_regression)
        if trial_final > max_allowed:
            failures.append(
                "quality regression: final loss "
                f"{trial_final:.6g} exceeds baseline {baseline_final:.6g} "
                f"by more than {q.max_final_loss_regression:.0%}"
            )

    trial_initial = _to_float(t.get("initial_loss"))
    if trial_initial is not None and trial_final is not None and trial_initial > 0:
        max_allowed = trial_initial * (1.0 + q.max_loss_divergence)
        if trial_final > max_allowed:
            failures.append(
                "quality regression: trial loss diverged "
                f"from {trial_initial:.6g} to {trial_final:.6g}"
            )

    output_diff = _to_float(t.get("max_output_abs_diff"))
    if output_diff is not None and output_diff > q.output_abs_tolerance:
        failures.append(
            "quality regression: max output absolute difference "
            f"{output_diff:.6g} exceeds tolerance {q.output_abs_tolerance:.6g}"
        )

    return not failures, failures


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
