from __future__ import annotations

import functools
import pathlib
from typing import Any

import yaml

from .signals import extract_signals

_CATALOG_PATH = pathlib.Path(__file__).parent / "catalog.yaml"
_RULES_PATH = pathlib.Path(__file__).parent / "rules.yaml"

_IMPACT_SCORE = {"high": 1.0, "medium": 0.6, "low": 0.3}
_EFFORT_BONUS = {"low": 1.0, "medium": 0.5, "high": 0.0}
_SAFETY_BONUS = {"high": 1.0, "medium": 0.5, "low": 0.0}

_BUNDLE_LABELS = {
    "input_pipeline": "Input Pipeline Optimization",
    "copy": "Host-to-Device Transfer Optimization",
    "synchronization": "Synchronization Overhead Reduction",
    "occupancy": "GPU Occupancy Improvement",
    "kernel_launch": "Kernel Launch Overhead Reduction",
    "memory": "Memory Pressure Relief",
    "shape_stability": "Shape Stability Improvement",
    "telemetry": "Telemetry & Observability",
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
) -> dict[str, Any]:
    catalog = _load_catalog()
    rules = _load_rules()
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
        return {"recommendations": [], "bundles": []}

    # ── Score and build recommendation objects ────────────────────────────────
    scored: list[tuple[float, dict[str, Any]]] = []

    for rec_id, (boost, rule, b_score) in candidates.items():
        entry = catalog.get(rec_id)
        if entry is None:
            continue

        impact_score = _IMPACT_SCORE.get(entry.get("impact", "medium"), 0.6)
        effort_bonus = _EFFORT_BONUS.get(entry.get("effort", "medium"), 0.5)
        safety_bonus = _SAFETY_BONUS.get(entry.get("safety", "medium"), 0.5)

        score = (
            0.40 * b_score
            + 0.25 * boost
            + 0.20 * impact_score
            + 0.10 * effort_bonus
            + 0.05 * safety_bonus
            + boost
        )
        score = round(min(1.0, score), 4)

        if score >= 0.75:
            priority = "high"
        elif score >= 0.50:
            priority = "medium"
        else:
            priority = "low"

        rec = {
            "id": rec_id,
            "title": entry["title"],
            "priority": priority,
            "score": score,
            "confidence": round(b_score, 4),
            "expected_impact": entry.get("impact", "medium"),
            "effort": entry.get("effort", "medium"),
            "category": entry.get("category", "general"),
            "why": rule.get("explanation", "").strip(),
            "actions": entry.get("action_templates", []),
            "validation": entry.get("validation_templates", []),
            "risks": entry.get("caveats", []),
            "triggered_by": rule["id"],
        }
        scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    recommendations = [rec for _, rec in scored]

    # ── Build bundles ─────────────────────────────────────────────────────────
    bundles = _build_bundles(recommendations)

    return {"recommendations": recommendations, "bundles": bundles}


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
