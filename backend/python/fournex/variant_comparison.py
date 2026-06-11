"""Multi-variant comparison: combine NCU profiles with measured throughput.

Links per-variant Nsight Compute CSVs to measured performance (GFLOP/s, steps/s,
or any throughput metric) and produces a ranked report that explains *why* each
optimization produced its measured delta — not just which metrics changed.

Input format (results.csv):

    variant,ncu_csv,throughput_gflops,notes
    bad_gemm,bad_gemm_ncu.csv,12.3,naive
    coalesced,coalesced_ncu.csv,50.5,coalesced access
    tiled,tiled_ncu.csv,46.0,16x16 tile
    tiled_unroll,tiled_unroll_ncu.csv,70.0,tiled + loop unroll

- ``variant``           — display name
- ``ncu_csv``           — NCU CSV path, relative to the results.csv directory
- ``throughput_gflops`` — measured throughput (any unit; used as ground truth ranking)
- ``notes``             — optional description

Usage::

    from fournex.variant_comparison import load_variants_csv, analyze_variants
    variants = load_variants_csv(Path("results.csv"))
    report = analyze_variants(variants)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .ncu_analysis import analyze_ncu_csv_text
from .ncu_comparison import diff_ncu_runs


def load_variants_csv(path: str | Path) -> list[dict[str, Any]]:
    """Parse a results.csv manifest and return a list of variant dicts.

    Each entry has: ``variant``, ``ncu_csv`` (resolved Path), ``throughput``,
    ``notes``.

    Raises ``FileNotFoundError`` if the manifest or any referenced NCU CSV is
    absent. Raises ``ValueError`` for malformed rows (missing required columns).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"results manifest not found: {path}")

    base_dir = path.parent
    variants: list[dict[str, Any]] = []

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"variant", "ncu_csv", "throughput_gflops"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"results.csv must have columns: variant, ncu_csv, throughput_gflops. "
                f"Got: {list(reader.fieldnames or [])}"
            )
        for i, row in enumerate(reader, start=2):
            variant = row.get("variant", "").strip()
            ncu_csv_rel = row.get("ncu_csv", "").strip()
            throughput_str = row.get("throughput_gflops", "").strip()
            if not variant or not ncu_csv_rel or not throughput_str:
                raise ValueError(f"results.csv line {i}: missing required field")
            try:
                throughput = float(throughput_str)
            except ValueError:
                raise ValueError(
                    f"results.csv line {i}: throughput_gflops must be a number, got {throughput_str!r}"
                )
            ncu_path = base_dir / ncu_csv_rel
            if not ncu_path.exists():
                raise FileNotFoundError(f"NCU CSV not found: {ncu_path} (referenced from {path})")
            variants.append({
                "variant": variant,
                "ncu_csv": ncu_path,
                "throughput": throughput,
                "notes": row.get("notes", "").strip(),
            })

    if len(variants) < 2:
        raise ValueError("results.csv must contain at least 2 variants for comparison")
    return variants


def analyze_variants(
    variants: list[dict[str, Any]],
    *,
    baseline_variant: str | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze multiple kernel variants and produce a ranked comparison report.

    ``variants`` is the list returned by ``load_variants_csv()``.
    ``baseline_variant`` names the reference variant (default: lowest throughput).

    Returns a ``variant_comparison_v1`` dict with:
    - ``variants_ranked``: all variants sorted by throughput descending, each with
      NCU analysis, throughput delta vs baseline, and bottleneck summary
    - ``transitions``: pairwise diff for each adjacent pair in throughput order,
      using ``diff_ncu_runs()`` to explain metric and bottleneck changes
    - ``baseline``: the chosen reference variant name
    - ``top_recommendation``: the most frequent high-priority recommendation
      across all variants (the "what to fix next" summary)
    """
    # Analyze each variant
    analyzed: list[dict[str, Any]] = []
    for v in variants:
        ncu_text = v["ncu_csv"].read_text(encoding="utf-8", errors="replace")
        ncu_result = analyze_ncu_csv_text(ncu_text, environment=environment)
        analyzed.append({
            "variant": v["variant"],
            "ncu_csv": str(v["ncu_csv"]),
            "throughput": v["throughput"],
            "notes": v["notes"],
            "ncu_result": ncu_result,
            "primary_bottleneck": ncu_result.get("primary_bottleneck"),
        })

    # Determine baseline
    if baseline_variant:
        baseline_entry = next((a for a in analyzed if a["variant"] == baseline_variant), None)
        if baseline_entry is None:
            raise ValueError(
                f"baseline variant {baseline_variant!r} not found in results.csv. "
                f"Available: {[a['variant'] for a in analyzed]}"
            )
    else:
        baseline_entry = min(analyzed, key=lambda a: a["throughput"])

    baseline_name = baseline_entry["variant"]
    baseline_throughput = baseline_entry["throughput"]

    # Rank by throughput descending
    ranked = sorted(analyzed, key=lambda a: a["throughput"], reverse=True)

    variants_ranked = []
    for entry in ranked:
        delta_x = (
            round(entry["throughput"] / baseline_throughput, 2)
            if baseline_throughput > 0 else None
        )
        variants_ranked.append({
            "variant": entry["variant"],
            "throughput": entry["throughput"],
            "notes": entry["notes"],
            "delta_vs_baseline_x": delta_x,
            "is_baseline": entry["variant"] == baseline_name,
            "primary_bottleneck": entry["primary_bottleneck"],
            "ncu_result": entry["ncu_result"],
        })

    # Build pairwise transitions (sorted by throughput ascending, then diff adjacent)
    ascending = sorted(analyzed, key=lambda a: a["throughput"])
    transitions = []
    for i in range(len(ascending) - 1):
        src = ascending[i]
        dst = ascending[i + 1]
        src_text = src["ncu_csv"]
        dst_text = dst["ncu_csv"]
        try:
            diff = diff_ncu_runs(
                Path(src_text).read_text(encoding="utf-8", errors="replace"),
                Path(dst_text).read_text(encoding="utf-8", errors="replace"),
                label_baseline=src["variant"],
                label_optimized=dst["variant"],
                environment=environment,
            )
        except Exception:
            diff = None

        throughput_delta_x = (
            round(dst["throughput"] / src["throughput"], 2)
            if src["throughput"] > 0 else None
        )
        # Pick the most-changed metric for the "headline" of this transition
        headline_metric = _pick_headline_metric(diff) if diff else None

        transitions.append({
            "from_variant": src["variant"],
            "to_variant": dst["variant"],
            "from_throughput": src["throughput"],
            "to_throughput": dst["throughput"],
            "throughput_delta_x": throughput_delta_x,
            "bottleneck_resolved": diff["bottleneck_diff"]["resolved"] if diff else [],
            "bottleneck_new": diff["bottleneck_diff"]["new"] if diff else [],
            "headline_metric": headline_metric,
            "diff": diff,
        })

    # Top recommendation: most frequent high-priority rec across all variants
    top_recommendation = _top_recommendation(analyzed)

    return {
        "schema": "variant_comparison_v1",
        "baseline": baseline_name,
        "variant_count": len(analyzed),
        "variants_ranked": variants_ranked,
        "transitions": transitions,
        "top_recommendation": top_recommendation,
    }


def _pick_headline_metric(diff: dict[str, Any]) -> dict[str, Any] | None:
    """Return the metric with the largest absolute % change from the diff."""
    deltas = diff.get("metric_deltas", {})
    best_key = None
    best_abs = 0.0
    for key, info in deltas.items():
        delta = info.get("delta")
        if delta is not None and abs(delta) > best_abs:
            best_abs = abs(delta)
            best_key = key
    if best_key is None:
        return None
    info = deltas[best_key]
    return {
        "key": best_key,
        "label": info.get("label", best_key),
        "baseline": info.get("baseline"),
        "optimized": info.get("optimized"),
        "delta": info.get("delta"),
        "direction": info.get("direction"),
        "unit": info.get("unit", ""),
    }


def _top_recommendation(analyzed: list[dict[str, Any]]) -> str | None:
    """Return the title of the most common high-priority recommendation across all variants."""
    counts: dict[str, int] = {}
    for entry in analyzed:
        recs = entry["ncu_result"].get("recommendations", [])
        for r in recs:
            if r.get("priority") == "high":
                title = r.get("title", "")
                if title:
                    counts[title] = counts.get(title, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.__getitem__)
