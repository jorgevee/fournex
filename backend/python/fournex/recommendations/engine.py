from __future__ import annotations

import functools
import pathlib
from typing import Any

import yaml

from .signals import extract_ncu_signals, extract_signals

_CATALOG_PATH = pathlib.Path(__file__).parent / "catalog.yaml"
_RULES_PATH = pathlib.Path(__file__).parent / "rules.yaml"

_DEFAULT_SPEEDUP_RANGE = {
    "high": (10.0, 25.0),
    "medium": (5.0, 12.0),
    "low": (1.0, 5.0),
}
_DEPENDENCY_ORDER = {
    "rec_launch_cuda_graphs": ("rec_launch_torch_compile",),
    "rec_launch_fuse_ops": ("rec_launch_torch_compile",),
    "rec_shape_disable_dynamic": ("rec_shape_bucket_inputs",),
}
_EQUIVALENT_RECOMMENDATIONS = {
    "rec_copy_pinned_memory": ("rec_input_pinned_memory",),
}

_BUNDLE_LABELS = {
    "input_pipeline": "Input Pipeline Optimization",
    "copy": "Host-to-Device Transfer Optimization",
    "synchronization": "Synchronization Overhead Reduction",
    "occupancy": "GPU Occupancy Improvement",
    "kernel_launch": "Kernel Launch Overhead Reduction",
    "memory": "Memory Pressure Relief",
    "shape_stability": "Shape Stability Improvement",
    "telemetry": "Telemetry & Observability",
    "ncu_memory": "Memory Bandwidth & Cache Optimization",
    "ncu_compute": "Compute Efficiency Improvement",
    "ncu_sync": "Kernel Synchronization Reduction",
    "ptx_static": "PTX Static Optimization",
}


@functools.cache
def _load_catalog() -> dict[str, dict[str, Any]]:
    entries = yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in entries}


@functools.cache
def _load_rules() -> list[dict[str, Any]]:
    return yaml.safe_load(_RULES_PATH.read_text(encoding="utf-8"))


def generate_recommendations(
    bottlenecks: list[dict[str, Any]],
    run_summary: dict[str, Any],
    per_step: list[dict[str, Any]] | None = None,
    environment: dict[str, Any] | None = None,
    *,
    signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    catalog = _load_catalog()
    rules = _load_rules()
    if signals is None:
        signals = extract_signals(run_summary, bottlenecks, per_step or [], environment)

    bottleneck_map = {b["label"]: b for b in bottlenecks}

    # ── Evaluate rules ───────────────────────────────────────────────────────
    # rec_id → best (priority_boost, rule, bottleneck_score)
    candidates: dict[str, tuple[float, dict[str, Any], float]] = {}

    for rule in rules:
        bottleneck_label = rule["bottleneck"]
        matched_bottleneck = bottleneck_map.get(bottleneck_label)
        if matched_bottleneck is None:
            continue
        if matched_bottleneck["score"] < rule.get("min_score", 0.0):
            continue
        if not _signals_match(rule.get("signals") or {}, signals):
            continue
        suppressed_if = rule.get("suppressed_if") or {}
        if suppressed_if and _any_signal_matches(suppressed_if, signals):
            continue

        boost = float(rule.get("priority_boost", 0.0))
        b_score = float(matched_bottleneck["score"])

        for rec_id in rule.get("recommendations", []):
            existing = candidates.get(rec_id)
            if existing is None or boost > existing[0]:
                candidates[rec_id] = (boost, rule, b_score)

    if not candidates:
        withheld = _build_withheld_recommendations(catalog, bottlenecks, signals, set())
        return {"recommendations": [], "bundles": [], "withheld_recommendations": withheld}

    # ── Score and build recommendation objects ────────────────────────────────
    scored_by_id: dict[str, tuple[float, dict[str, Any]]] = {}

    for rec_id, (boost, rule, b_score) in candidates.items():
        entry = catalog.get(rec_id)
        if entry is None:
            continue

        score, components, guardrails = _score_recommendation(entry, b_score, boost, signals)

        if score >= 0.75:
            priority = "high"
        elif score >= 0.50:
            priority = "medium"
        else:
            priority = "low"
        tier = _tier_for_recommendation(score, components, guardrails)

        rec = {
            "id": rec_id,
            "title": entry["title"],
            "priority": priority,
            "score": score,
            "roi_score": score,
            "tier": tier,
            "confidence": round(b_score, 4),
            "expected_impact": entry.get("impact", "medium"),
            "effort": entry.get("effort", "medium"),
            "risk": _risk_level(entry),
            "category": entry.get("category", "general"),
            "why": rule.get("explanation", "").strip(),
            "why_ranked": _ranking_explanation(components, guardrails),
            "roi_components": components,
            "guardrails_applied": guardrails,
            "prerequisites": entry.get("prerequisites", {}),
            "actions": entry.get("action_templates", []),
            "validation": entry.get("validation_templates", []),
            "risks": entry.get("caveats", []),
            "triggered_by": rule["id"],
        }
        scored_by_id[rec_id] = (score, rec)

    _dedupe_equivalent_recommendations(scored_by_id)
    _enforce_dependency_order(scored_by_id)
    scored = list(scored_by_id.values())
    scored.sort(key=lambda x: x[0], reverse=True)
    recommendations = [rec for _, rec in scored]
    active_ids = {rec["id"] for rec in recommendations}

    # ── Build bundles ─────────────────────────────────────────────────────────
    bundles = _build_bundles(recommendations)
    withheld = _build_withheld_recommendations(catalog, bottlenecks, signals, active_ids)

    return {
        "recommendations": recommendations,
        "bundles": bundles,
        "withheld_recommendations": withheld,
    }


def _signals_match(conditions: dict[str, Any], signals: dict[str, Any]) -> bool:
    """Return True if ALL conditions match the signals dict (AND logic)."""
    for key, expected in conditions.items():
        if signals.get(key) != expected:
            return False
    return True


def _any_signal_matches(conditions: dict[str, Any], signals: dict[str, Any]) -> bool:
    """Return True if ANY condition matches the signals dict (OR logic).

    Used for suppressed_if: a rule is suppressed if any one of the listed
    signals is active, not only when all of them are simultaneously true.
    """
    for key, expected in conditions.items():
        if signals.get(key) == expected:
            return True
    return False


def _score_recommendation(
    entry: dict[str, Any],
    confidence: float,
    priority_boost: float,
    signals: dict[str, Any],
) -> tuple[float, dict[str, float], list[str]]:
    speedup_score = _speedup_score(entry)
    confidence_score = _clamp(confidence)
    cost_savings_score = _cost_savings_score(speedup_score, signals)
    ease_score = _effort_score(entry)
    safety_score = _safety_score(entry)

    raw_score = (
        0.30 * speedup_score
        + 0.20 * confidence_score
        + 0.20 * cost_savings_score
        + 0.15 * ease_score
        + 0.15 * safety_score
    )
    boost = min(priority_boost, 0.20)
    score = raw_score + boost

    guardrails: list[str] = []
    if confidence_score < 0.40:
        score *= 0.75
        guardrails.append("low_confidence_demoted")
    if safety_score < 0.40 and speedup_score < 0.70:
        score *= 0.85
        guardrails.append("risk_demoted")

    components = {
        "speedup": round(speedup_score, 4),
        "confidence": round(confidence_score, 4),
        "cost_savings": round(cost_savings_score, 4),
        "ease": round(ease_score, 4),
        "safety": round(safety_score, 4),
        "priority_boost": round(boost, 4),
    }
    return round(_clamp(score), 4), components, guardrails


def _speedup_score(entry: dict[str, Any]) -> float:
    speedup_min = entry.get("estimated_speedup_pct_min")
    speedup_max = entry.get("estimated_speedup_pct_max")
    if speedup_min is None or speedup_max is None:
        speedup_min, speedup_max = _DEFAULT_SPEEDUP_RANGE.get(
            entry.get("impact", "medium"), _DEFAULT_SPEEDUP_RANGE["medium"]
        )
    expected_speedup = (float(speedup_min) + float(speedup_max)) / 2.0
    if expected_speedup < 2.0:
        return 0.10
    if expected_speedup < 5.0:
        return 0.25
    if expected_speedup < 10.0:
        return 0.45
    if expected_speedup < 20.0:
        return 0.70
    if expected_speedup < 35.0:
        return 0.90
    return 1.00


def _effort_score(entry: dict[str, Any]) -> float:
    effort = entry.get("implementation_effort", entry.get("effort", "medium"))
    return {
        "config": 1.00,
        "low": 0.90,
        "medium": 0.55,
        "high": 0.20,
        "custom_cuda": 0.10,
    }.get(str(effort), 0.55)


def _safety_score(entry: dict[str, Any]) -> float:
    safety = entry.get("safety", "medium")
    risk = entry.get("risk_level")
    if risk is not None:
        return {"low": 0.90, "medium": 0.55, "high": 0.20}.get(str(risk), 0.55)
    return {"high": 0.90, "medium": 0.55, "low": 0.20}.get(str(safety), 0.55)


def _cost_savings_score(speedup_score: float, signals: dict[str, Any]) -> float:
    num_gpus = int(signals.get("num_gpus", 1))
    if num_gpus >= 64:
        exposure_score = 0.95
    elif num_gpus >= 8:
        exposure_score = 0.80
    elif num_gpus >= 2:
        exposure_score = 0.55
    else:
        exposure_score = 0.35
    return round(_clamp((0.60 * speedup_score) + (0.40 * exposure_score)), 4)


def _tier_for_recommendation(
    score: float,
    components: dict[str, float],
    guardrails: list[str],
) -> str:
    if "low_confidence_demoted" in guardrails:
        return "advanced"
    if score >= 0.75 and components["ease"] >= 0.75 and components["safety"] >= 0.55:
        return "try_now"
    if score >= 0.50:
        return "next"
    return "advanced"


def _risk_level(entry: dict[str, Any]) -> str:
    explicit = entry.get("risk_level")
    if explicit is not None:
        return str(explicit)
    safety = entry.get("safety", "medium")
    return {"high": "low", "medium": "medium", "low": "high"}.get(str(safety), "medium")


def _ranking_explanation(components: dict[str, float], guardrails: list[str]) -> list[str]:
    reasons = []
    if components["speedup"] >= 0.70:
        reasons.append("High expected speedup")
    elif components["speedup"] >= 0.45:
        reasons.append("Moderate expected speedup")
    if components["confidence"] >= 0.75:
        reasons.append("Strong diagnostic confidence")
    if components["ease"] >= 0.75:
        reasons.append("Low implementation effort")
    if components["safety"] >= 0.75:
        reasons.append("Low operational risk")
    if components["cost_savings"] >= 0.70:
        reasons.append("Meaningful cost exposure")
    if not reasons:
        reasons.append("Ranked by balanced ROI score")
    for guardrail in guardrails:
        if guardrail == "low_confidence_demoted":
            reasons.append("Demoted because diagnostic confidence is low")
        elif guardrail == "risk_demoted":
            reasons.append("Demoted because risk is high relative to expected gain")
    return reasons


def _enforce_dependency_order(scored_by_id: dict[str, tuple[float, dict[str, Any]]]) -> None:
    for rec_id, prerequisites in _DEPENDENCY_ORDER.items():
        current = scored_by_id.get(rec_id)
        if current is None:
            continue
        for prerequisite_id in prerequisites:
            prerequisite = scored_by_id.get(prerequisite_id)
            if prerequisite is None:
                continue
            current_score, rec = current
            prerequisite_score, _ = prerequisite
            if current_score >= prerequisite_score:
                adjusted_score = round(max(0.0, prerequisite_score - 0.0001), 4)
                rec["score"] = adjusted_score
                rec["roi_score"] = adjusted_score
                rec["guardrails_applied"] = rec["guardrails_applied"] + ["dependency_ordered"]
                rec["why_ranked"] = rec["why_ranked"] + [
                    f"Ranked after prerequisite recommendation {prerequisite_id}"
                ]
                current = (adjusted_score, rec)
                scored_by_id[rec_id] = current


def _dedupe_equivalent_recommendations(scored_by_id: dict[str, tuple[float, dict[str, Any]]]) -> None:
    for canonical_id, duplicate_ids in _EQUIVALENT_RECOMMENDATIONS.items():
        if canonical_id not in scored_by_id:
            continue
        for duplicate_id in duplicate_ids:
            scored_by_id.pop(duplicate_id, None)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _build_bundles(recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_categories: dict[str, list[str]] = {}
    for rec in recommendations:
        cat = rec["category"]
        seen_categories.setdefault(cat, []).append(rec["id"])

    bundles = []
    for cat, rec_ids in seen_categories.items():
        if len(rec_ids) >= 2:
            bundles.append({
                "label": _BUNDLE_LABELS.get(cat, cat.replace("_", " ").title()),
                "category": cat,
                "recommendation_ids": rec_ids,
            })
    return bundles


def _build_withheld_recommendations(
    catalog: dict[str, dict[str, Any]],
    bottlenecks: list[dict[str, Any]],
    signals: dict[str, Any],
    active_ids: set[str],
) -> list[dict[str, Any]]:
    bottleneck_labels = {b["label"] for b in bottlenecks}
    withheld: list[dict[str, Any]] = []

    if (
        "input_bound" in bottleneck_labels
        and signals.get("input_pipeline_stalled")
        and signals.get("h2d_copy_low")
        and "rec_input_pinned_memory" not in active_ids
        and "rec_copy_pinned_memory" not in active_ids
    ):
        withheld.append(_withheld_entry(
            catalog,
            "rec_input_pinned_memory",
            reason=(
                "DataLoader wait is elevated, but H2D copy time is low; pinned memory "
                "is unlikely to improve a CPU-side input stall."
            ),
            blocked_by=["h2d_copy_low"],
            evidence={
                "input_frac": round(float(signals.get("input_frac", 0.0)), 4),
                "h2d_frac": round(float(signals.get("h2d_frac", 0.0)), 4),
            },
        ))

    return withheld


def _withheld_entry(
    catalog: dict[str, dict[str, Any]],
    rec_id: str,
    *,
    reason: str,
    blocked_by: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    entry = catalog.get(rec_id, {})
    return {
        "id": rec_id,
        "title": entry.get("title", rec_id),
        "category": entry.get("category", "general"),
        "reason": reason,
        "blocked_by": blocked_by,
        "evidence": evidence,
    }
