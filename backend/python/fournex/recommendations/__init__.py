from __future__ import annotations

import functools
import pathlib
from typing import Any

import yaml

_CATALOG_PATH = pathlib.Path(__file__).parent / "catalog.yaml"
_RULES_PATH = pathlib.Path(__file__).parent / "rules.yaml"

_EFFORT_RANK = {"config": 0, "low": 1, "medium": 2, "high": 3, "custom_cuda": 4}
_IMPACT_RANK = {"high": 0, "medium": 1, "low": 2}


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
    bottleneck_map = {b["label"]: b for b in bottlenecks}

    seen: dict[str, float] = {}
    for rule in rules:
        matched = bottleneck_map.get(rule["bottleneck"])
        if matched is None:
            continue
        b_score = float(matched.get("score", 0.0))
        if b_score < rule.get("min_score", 0.0):
            continue
        boost = float(rule.get("priority_boost", 0.0))
        for rec_id in rule.get("recommendations", []):
            existing = seen.get(rec_id, -1.0)
            if b_score + boost > existing:
                seen[rec_id] = b_score + boost

    if not seen:
        return {"recommendations": [], "bundles": []}

    recommendations: list[dict[str, Any]] = []
    for rec_id, score in seen.items():
        entry = catalog.get(rec_id)
        if entry is None:
            continue
        impact = entry.get("impact", "medium")
        effort = entry.get("implementation_effort", entry.get("effort", "medium"))
        risk_level = entry.get("risk_level") or (
            {"high": "low", "medium": "medium", "low": "high"}.get(str(entry.get("safety", "medium")), "medium")
        )
        priority = "high" if score >= 0.70 else "medium" if score >= 0.45 else "low"
        tier = "try_now" if priority == "high" and effort in ("config", "low") else "next" if priority != "low" else "advanced"

        # find the first matching rule's explanation for this rec
        why = ""
        for rule in rules:
            if rec_id in rule.get("recommendations", []) and bottleneck_map.get(rule["bottleneck"]):
                why = rule.get("explanation", "").strip()
                break

        recommendations.append({
            "id": rec_id,
            "title": entry["title"],
            "priority": priority,
            "score": round(min(1.0, score), 4),
            "roi_score": round(min(1.0, score), 4),
            "tier": tier,
            "confidence": round(min(1.0, score), 4),
            "expected_impact": impact,
            "effort": effort,
            "risk": risk_level,
            "category": entry.get("category", "general"),
            "why": why,
            "why_ranked": [],
            "roi_components": {},
            "guardrails_applied": [],
            "prerequisites": entry.get("prerequisites", {}),
            "actions": entry.get("action_templates", []),
            "validation": entry.get("validation_templates", []),
            "risks": entry.get("caveats", []),
            "triggered_by": "",
        })

    recommendations.sort(key=lambda r: (
        _IMPACT_RANK.get(r["expected_impact"], 1),
        _EFFORT_RANK.get(str(r["effort"]), 2),
        -r["score"],
    ))

    seen_categories: dict[str, list[str]] = {}
    for rec in recommendations:
        seen_categories.setdefault(rec["category"], []).append(rec["id"])

    bundles = [
        {"label": cat.replace("_", " ").title(), "category": cat, "recommendation_ids": ids}
        for cat, ids in seen_categories.items()
        if len(ids) >= 2
    ]

    return {"recommendations": recommendations, "bundles": bundles}
