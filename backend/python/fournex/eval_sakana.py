"""Evaluate Fournex's analyzer on the SakanaAI/AI-CUDA-Engineer-Archive.

Answers one question: *can Fournex produce useful, evidence-backed judgments on
messy generated CUDA kernels without human labels?* For each kernel we run the
NCU adapter + static source analyzer, reconcile them, and record what Fournex
concluded, how confident it was, what evidence it lacked, and whether it raised a
correctness/safety flag. We then aggregate honest, clearly-labeled metrics.

Design stance — we test Fournex, not Sakana:
  * The dataset has **no ground-truth bottleneck labels**, so the only "accuracy"
    numbers come from a small hand-labeled gold set (``data/sakana/gold.yaml``).
    Everything else is either objective (coverage, correctness recall vs the
    ``Correct`` flag) or a clearly-flagged NCU-derived **weak label** used for
    self-consistency, never presented as truth.
  * Correctness is first-class: speedup != success. Fournex never infers
    correctness from a profile (a wrong kernel profiles fine). Its correctness
    signal comes only from the build/runtime ``Error``, Clang-Tidy *errors*, and
    static correctness-risk findings — and the report says so. Silent numerical
    mismatches (``Error`` null, high ``Max_Diff``) are intentionally counted as a
    documented blind spot, not a Fournex failure.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .cuda_static import inspect_cuda_source
from .explain import build_explain_result, render_summary_txt
from .ncu_analysis import analyze_ncu_profile_dict
from .sakana_ncu_adapter import ABSENT_SECTIONS, parse_ncu_profile

_DATA_DIR = Path(__file__).resolve().parent / "data" / "sakana"
DEFAULT_SUBSET = _DATA_DIR / "subset.jsonl"
DEFAULT_GOLD = _DATA_DIR / "gold.yaml"

# Confidence ordering for the gold "ceiling" check (low to high).
_CONFIDENCE_ORDER = ["low-medium", "medium", "medium-high", "high", "confirmed"]

# Static finding codes that indicate a genuine correctness/safety risk (not perf,
# and not robustness lints like missing bounds guards that fire on correct-by-
# construction kernels). Kept deliberately narrow to stay high-precision.
_CORRECTNESS_RISK_CODES = frozenset({
    "conditional_syncthreads",  # __syncthreads() inside a divergent branch -> hang/UB
})


def _confidence_rank(label: str | None) -> int:
    try:
        return _CONFIDENCE_ORDER.index(label or "")
    except ValueError:
        return -1


# ── Row plumbing ────────────────────────────────────────────────────────────

def row_key(row: dict) -> str:
    """Stable identifier for a kernel variant within the archive."""
    idx = row.get("__index_level_0__")
    return f"L{row.get('Level_ID')}/T{row.get('Task_ID')}/{row.get('Kernel_Name')}#{idx}"


def _as_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def load_rows(
    source: str | Path | None = None,
    *,
    sample: int | None = None,
    level: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """Load rows from a JSONL subset (defaults to the packaged cached fixture).

    Sampling is deterministic for a given ``seed`` so eval runs are reproducible.
    """
    path = Path(source) if source else DEFAULT_SUBSET
    if not path.exists():
        raise FileNotFoundError(
            f"Sakana subset not found at {path}. Run scripts/fetch_sakana.py to build it."
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if level is not None:
        rows = [r for r in rows if r.get("Level_ID") == level]
    if sample is not None and sample < len(rows):
        rows = random.Random(seed).sample(rows, sample)
    return rows


# ── Correctness signal (never from the profile) ──────────────────────────────

def _clang_has_errors(clang_raw: Any) -> bool:
    parsed = parse_ncu_profile(clang_raw)  # same Python-repr format; reuse the safe parser
    text = ""
    if isinstance(parsed, dict):
        text = str(parsed.get("stdout", "")) + str(parsed.get("stderr", ""))
    elif isinstance(clang_raw, str):
        text = clang_raw
    return ": error:" in text


def correctness_signal(row: dict, static: dict) -> dict:
    """Fournex's correctness/safety judgment for one kernel.

    ``status`` is a standing caveat (Fournex never executes the kernel, so it
    cannot certify correctness). ``warning`` is a positive red flag with reasons.
    """
    reasons: list[str] = []
    error = row.get("Error")
    if isinstance(error, str) and error.strip():
        reasons.append("build/runtime error reported during compilation or execution")
    if _clang_has_errors(row.get("Clang_Tidy")):
        reasons.append("clang-tidy reported error-level diagnostics")
    risk_codes = sorted(
        {f.get("code", "") for f in static.get("findings", [])} & _CORRECTNESS_RISK_CODES
    )
    reasons.extend(f"static correctness-risk: {c}" for c in risk_codes)
    return {
        "status": "not_verified_by_fournex",
        "warning": bool(reasons),
        "reasons": reasons,
    }


# ── NCU-only weak label (flagged heuristic, never truth) ──────────────────────

def derive_weak_label(ncu: dict | None) -> str:
    """A coarse NCU-only bottleneck guess for self-consistency reporting only.

    Circular by construction (Fournex reads the same metrics), so this is never
    presented as ground truth — only to see how often Fournex's reconciled
    primary lines up with the raw Speed-of-Light reading.
    """
    if not ncu:
        return "no_ncu"
    s = ncu.get("ncu_run_summary", {})
    occ = s.get("avg_occupancy_pct")
    dram = s.get("avg_dram_throughput_pct")
    mem_busy = s.get("avg_memory_busy_pct")
    sm = s.get("avg_sm_throughput_pct")
    l1 = s.get("avg_l1_cache_hit_rate_pct")
    l2 = s.get("avg_l2_cache_hit_rate_pct")
    issue = s.get("avg_issue_slot_utilization_pct")
    if occ is not None and occ < 40:
        return "low_occupancy"
    if dram is not None and dram > 60 and (mem_busy or 0) >= (sm or 0):
        return "memory_bound"
    if (l1 is not None and l1 < 40) or (l2 is not None and l2 < 50):
        return "cache_thrashing"
    if issue is not None and issue < 60:
        return "low_issue_efficiency"
    return "inconclusive"


# ── Per-row evaluation ────────────────────────────────────────────────────────

def evaluate_row(row: dict, *, environment: dict | None = None) -> dict:
    name = str(row.get("Kernel_Name") or "kernel")
    cuda_code = row.get("CUDA_Code") or ""

    ncu = None
    if parse_ncu_profile(row.get("NCU_Profile")) is not None:
        ncu = analyze_ncu_profile_dict(row.get("NCU_Profile"), kernel_name=name, environment=environment)

    static = inspect_cuda_source(cuda_code, filename=f"{name}.cu") if cuda_code else {"findings": [], "kernels": []}
    explain = build_explain_result(ncu_result=ncu, static_result=static, environment=environment)

    diagnoses = explain.get("diagnoses", [])
    primary = explain.get("primary_diagnosis")
    primary_diag = next((d for d in diagnoses if d.get("label") == primary), None)
    confidence = primary_diag.get("confidence") if primary_diag else None

    # Which classifier-usable sections this dataset's profile lacked (explains why
    # certain diagnoses can't be confirmed and why confidence stays bounded).
    absent = list(ABSENT_SECTIONS) if ncu is not None else ["ncu_profile_absent"]
    missing_metrics = sorted({
        m.get("metric", "")
        for d in diagnoses
        for m in (d.get("missing_evidence") or {}).get("metrics", [])
    } - {""})

    correctness = correctness_signal(row, static)
    speedup = _as_float(row.get("CUDA_Speedup_Native"))

    summary_txt = render_summary_txt(explain, src_filename=f"{name}.cu")
    headline = summary_txt.strip().splitlines()[0] if summary_txt.strip() else ""

    return {
        "row_key": row_key(row),
        "op_name": row.get("Op_Name"),
        "level": row.get("Level_ID"),
        "kernel_name": name,
        # Fournex output
        "primary_diagnosis": primary,
        "confidence": confidence,
        "diagnoses": [
            {"label": d["label"], "confidence": d["confidence"], "severity": d["severity"],
             "layers_confirming": d["layers_confirming"]}
            for d in diagnoses
        ],
        "layers_available": explain.get("layers_available", []),
        "missing_evidence_metrics": missing_metrics,
        "absent_ncu_sections": absent,
        "weak_label": derive_weak_label(ncu),
        "correctness": correctness,
        "summary_headline": headline,
        # Ground-truth outcomes (from the dataset, for scoring/correlation)
        "ground_truth": {
            "correct": bool(row.get("Correct")),
            "max_diff": _as_float(row.get("Max_Diff")),
            "has_error_text": bool(isinstance(row.get("Error"), str) and row.get("Error").strip()),
            "speedup_native": speedup,
            "speedup_compile": _as_float(row.get("CUDA_Speedup_Compile")),
        },
    }


# ── Gold set ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoldEntry:
    row_key: str
    expected_primary: str | None
    confidence_ceiling: str | None
    expect_correctness_warning: bool | None
    notes: str = ""


def load_gold(path: str | Path | None = None) -> dict[str, GoldEntry]:
    import yaml

    p = Path(path) if path else DEFAULT_GOLD
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    gold: dict[str, GoldEntry] = {}
    for item in raw:
        gold[item["row_key"]] = GoldEntry(
            row_key=item["row_key"],
            expected_primary=item.get("expected_primary"),
            confidence_ceiling=item.get("confidence_ceiling"),
            expect_correctness_warning=item.get("expect_correctness_warning"),
            notes=item.get("notes", ""),
        )
    return gold


# ── Aggregation / leaderboard ─────────────────────────────────────────────────

def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def aggregate(records: list[dict], gold: dict[str, GoldEntry]) -> dict:
    n = len(records)

    # Coverage: produced a concrete (non-inconclusive, non-null) primary diagnosis.
    concrete = [r for r in records if r["primary_diagnosis"] not in (None, "inconclusive")]

    # Self-consistency vs the NCU-only weak label (HEURISTIC — flagged below).
    weak_pairs = [r for r in records if r["weak_label"] not in ("no_ncu", "inconclusive")]
    weak_agree = sum(1 for r in weak_pairs if _weak_agrees(r["primary_diagnosis"], r["weak_label"]))

    # Correctness recall, bucketed by detectability (the honest split).
    incorrect = [r for r in records if not r["ground_truth"]["correct"]]
    build_fail = [r for r in incorrect if r["ground_truth"]["has_error_text"]]
    silent = [r for r in incorrect if not r["ground_truth"]["has_error_text"]]
    correct = [r for r in records if r["ground_truth"]["correct"]]
    warned = lambda rs: sum(1 for r in rs if r["correctness"]["warning"])

    # Speedup alignment (no over-claiming on fast kernels; explanation of slow ones).
    fast = [r for r in records if (r["ground_truth"]["speedup_native"] or 0) >= 1.0]
    slow = [r for r in records if (r["ground_truth"]["speedup_native"] or 0) < 1.0]
    fast_overclaimed = sum(
        1 for r in fast
        if any(d["severity"] == "high" and _confidence_rank(d["confidence"]) >= _confidence_rank("high")
               for d in r["diagnoses"])
    )
    slow_explained = sum(1 for r in slow if r["primary_diagnosis"] not in (None, "inconclusive"))

    # Confidence ceiling: this dataset's profiles lack warp-stall data, so no
    # diagnosis may exceed "medium-high". Track both the guard and the actual
    # strongest claim Fournex made (the ceiling is a cap, not what it emits).
    over_high = sum(1 for r in records if _confidence_rank(r["confidence"]) > _confidence_rank("medium-high"))
    emitted_ranks = [
        _confidence_rank(d["confidence"])
        for r in records for d in r["diagnoses"]
        if _confidence_rank(d["confidence"]) >= 0
    ]
    max_emitted = _CONFIDENCE_ORDER[max(emitted_ranks)] if emitted_ranks else None

    leaderboard: dict[str, Any] = {
        "rows_evaluated": n,
        "coverage_concrete_diagnosis": {
            "value": _rate(len(concrete), n), "basis": "objective",
            "note": "fraction with a concrete (non-inconclusive) primary diagnosis",
        },
        "no_crash_rate": {"value": 1.0 if n else None, "basis": "objective",
                          "note": "every row analyzed without raising"},
        "primary_diagnosis_distribution": _distribution(r["primary_diagnosis"] for r in records),
        "self_consistency_vs_weak_label": {
            "value": _rate(weak_agree, len(weak_pairs)), "basis": "heuristic-circular",
            "note": "agreement vs NCU-only Speed-of-Light reading; NOT ground truth",
            "n": len(weak_pairs),
        },
        "correctness_warning_recall": {
            "build_or_runtime_error": {"value": _rate(warned(build_fail), len(build_fail)),
                                        "n": len(build_fail), "basis": "objective"},
            "silent_numerical_mismatch": {"value": _rate(warned(silent), len(silent)),
                                           "n": len(silent), "basis": "objective",
                                           "note": "documented blind spot: undetectable by profile/static"},
            "warning_rate_on_correct_kernels": {"value": _rate(warned(correct), len(correct)),
                                                 "n": len(correct), "basis": "objective",
                                                 "note": "warnings on kernels that passed; may be legitimate safety flags (e.g. conditional __syncthreads), not necessarily false positives"},
        },
        "speedup_alignment": {
            "fast_kernels_not_overclaimed": {
                "value": _rate(len(fast) - fast_overclaimed, len(fast)), "n": len(fast), "basis": "objective",
                "note": "fast kernels (speedup>=1) NOT tagged high-severity/high-confidence",
            },
            "slow_kernels_explained": {
                "value": _rate(slow_explained, len(slow)), "n": len(slow), "basis": "objective",
                "note": "slow kernels (speedup<1) given a concrete primary diagnosis",
            },
        },
        "confidence_ceiling_respected": {
            "value": _rate(n - over_high, n), "basis": "objective",
            "ceiling": "medium-high",
            "note": "fraction of rows whose confidence stayed at/below the medium-high ceiling (no warp-stall data exists in this dataset to justify more)",
        },
        "max_confidence_emitted": {
            "value": max_emitted, "basis": "objective",
            "note": "the single strongest confidence label Fournex actually attached to any diagnosis across the run",
        },
    }

    if gold:
        leaderboard["gold"] = _score_gold(records, gold)

    return leaderboard


def _weak_agrees(primary: str | None, weak: str) -> bool:
    """Loose family match between a reconciled diagnosis label and a weak label."""
    if not primary:
        return False
    p = primary.lower()
    families = {
        "memory_bound": ("memory_bandwidth", "roofline_memory", "global_memory"),
        "low_occupancy": ("occupancy", "roofline_low_mfu"),
        "cache_thrashing": ("cache",),
        "low_issue_efficiency": ("issue", "low_mfu"),
    }
    return any(tok in p for tok in families.get(weak, ()))


def _distribution(labels: Iterable[str | None]) -> dict[str, int]:
    out: dict[str, int] = {}
    for lab in labels:
        key = lab or "none"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _score_gold(records: list[dict], gold: dict[str, GoldEntry]) -> dict:
    by_key = {r["row_key"]: r for r in records}
    matched = [(g, by_key[k]) for k, g in gold.items() if k in by_key]

    prim_total = prim_ok = 0
    ceil_total = ceil_ok = 0
    warn_total = warn_ok = 0
    failures: list[dict] = []

    for g, r in matched:
        if g.expected_primary is not None:
            prim_total += 1
            ok = (r["primary_diagnosis"] == g.expected_primary) or (
                g.expected_primary == "inconclusive" and r["primary_diagnosis"] in (None, "inconclusive")
            )
            prim_ok += ok
            if not ok:
                failures.append({"row_key": g.row_key, "check": "primary",
                                 "expected": g.expected_primary, "got": r["primary_diagnosis"]})
        if g.confidence_ceiling is not None:
            ceil_total += 1
            ok = _confidence_rank(r["confidence"]) <= _confidence_rank(g.confidence_ceiling)
            ceil_ok += ok
            if not ok:
                failures.append({"row_key": g.row_key, "check": "confidence_ceiling",
                                 "ceiling": g.confidence_ceiling, "got": r["confidence"]})
        if g.expect_correctness_warning is not None:
            warn_total += 1
            ok = r["correctness"]["warning"] == g.expect_correctness_warning
            warn_ok += ok
            if not ok:
                failures.append({"row_key": g.row_key, "check": "correctness_warning",
                                 "expected": g.expect_correctness_warning, "got": r["correctness"]["warning"]})

    return {
        "gold_rows_in_subset": len(matched),
        "gold_rows_total": len(gold),
        "primary_bottleneck_accuracy": {"value": _rate(prim_ok, prim_total), "n": prim_total, "basis": "vs-truth"},
        "confidence_ceiling_respected": {"value": _rate(ceil_ok, ceil_total), "n": ceil_total, "basis": "vs-truth"},
        "correctness_warning_accuracy": {"value": _rate(warn_ok, warn_total), "n": warn_total, "basis": "vs-truth"},
        "failures": failures,
    }


# ── Top-level run ─────────────────────────────────────────────────────────────

def run_eval(
    *,
    source: str | Path | None = None,
    sample: int | None = None,
    level: int | None = None,
    seed: int = 0,
    environment: dict | None = None,
    gold_path: str | Path | None = None,
    use_gold: bool = True,
) -> dict:
    rows = load_rows(source, sample=sample, level=level, seed=seed)
    records = [evaluate_row(r, environment=environment) for r in rows]
    gold = load_gold(gold_path) if use_gold else {}
    leaderboard = aggregate(records, gold)
    return {
        "schema": "sakana_eval_v1",
        "dataset": "SakanaAI/AI-CUDA-Engineer-Archive",
        "confidence_scale": {
            "ordered": list(_CONFIDENCE_ORDER),
            "meaning": "how many independent analysis layers (source/ptx/ncu/profiler) confirm a diagnosis",
            "low-medium": "1 of 2+ available layers confirms",
            "medium": "1 layer confirms and it is the only one available",
            "medium-high": "2 layers confirm with a 3rd+ still available",
            "high": "2 of exactly 2 available layers confirm",
            "confirmed": "3+ layers confirm",
        },
        "dataset_meta": {
            "source": str(Path(source) if source else DEFAULT_SUBSET),
            "rows": len(records),
            "level": level,
            "sample": sample,
            "seed": seed,
        },
        "caveats": [
            "No ground-truth bottleneck labels exist in this dataset; accuracy is measured only against a small hand-labeled gold set.",
            "Fournex never infers correctness from a profile; silent numerical mismatches are a documented blind spot.",
            "This dataset's NCU profiles lack warp-stall, coalescing, and tensor-pipe sections, so several bottleneck classes cannot be confirmed here.",
            "weak_label agreement is heuristic/circular and is not a measure of accuracy.",
        ],
        "leaderboard": leaderboard,
        "per_row": records,
    }
