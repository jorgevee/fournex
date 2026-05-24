"""Human-readable accuracy report for the Fournex NCU diagnostic pipeline.

Runs all fixture scenarios and prints a summary table with true positives
and false positives. No GPU required.

Usage:
    python backend/tests/evals/run_accuracy_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as fn

FIXTURES = Path(__file__).parent / "fixtures"

_SCENARIOS = [
    {
        "name": "uncoalesced_dram_bound",
        "fixture": "uncoalesced_dram_bound.csv",
        "expected": {"memory_bandwidth_bound", "uncoalesced_access", "l1_cache_thrashing"},
        "must_be_absent": {"warp_stall_sync"},
        "description": "Stride-K access: DRAM 89%, L1 22%, sectors/req 9.3",
    },
    {
        "name": "tensor_core_idle",
        "fixture": "tensor_core_idle.csv",
        "expected": {"tensor_core_underutilized"},
        "must_be_absent": {"warp_stall_sync", "memory_bandwidth_bound"},
        "description": "FP32 GEMM: TC util 6%, occ 62%",
    },
    {
        "name": "excessive_sync",
        "fixture": "excessive_sync.csv",
        "expected": {"warp_stall_sync"},
        "must_be_absent": {"tensor_core_underutilized", "memory_bandwidth_bound"},
        "description": "Spurious barriers: barrier stall 42%, wait 18%",
    },
    {
        "name": "register_pressure",
        "fixture": "register_pressure.csv",
        "expected": {"occupancy_limited"},
        "must_be_absent": set(),
        "description": "High register count: occupancy 22%",
    },
    {
        "name": "well_optimized",
        "fixture": "well_optimized.csv",
        "expected": set(),
        "must_be_absent": set(),
        "description": "True negative: DRAM 32%, TC 68%, L1 78% — no bottlenecks",
    },
]


def _run_scenario(scenario: dict) -> dict:
    csv = (FIXTURES / scenario["fixture"]).read_text()
    result = fn.analyze_ncu_csv_text(csv)
    detected = {b["label"] for b in result["bottlenecks"]}
    expected = scenario["expected"]
    must_absent = scenario["must_be_absent"]

    tp = expected & detected
    fn_set = expected - detected
    fp = must_absent & detected

    return {
        "detected": detected,
        "tp": tp,
        "fn": fn_set,
        "fp": fp,
        "tp_count": len(tp),
        "expected_count": len(expected),
        "fp_count": len(fp),
    }


def _abbreviate(labels: set[str], width: int = 32) -> str:
    if not labels:
        return "(none)"
    joined = ", ".join(sorted(labels))
    return joined[:width] + "…" if len(joined) > width else joined


def main() -> int:
    sep = "-" * 78
    header_fmt = f"{'Scenario':<28}  {'Expected':<22}  {'TP':>5}  {'FP':>5}"

    print()
    print("  Fournex NCU Diagnostic Pipeline - Accuracy Eval")
    print(f"  {sep}")
    print(f"  {header_fmt}")
    print(f"  {sep}")

    total_expected = 0
    total_tp = 0
    total_fp = 0
    all_pass = True

    for sc in _SCENARIOS:
        r = _run_scenario(sc)
        total_expected += r["expected_count"]
        total_tp += r["tp_count"]
        total_fp += r["fp_count"]

        tp_str = f"{r['tp_count']}/{r['expected_count']}"
        fp_str = str(r["fp_count"])

        status = ""
        if r["fn"]:
            status = f"  MISS: {_abbreviate(r['fn'], 40)}"
            all_pass = False
        if r["fp"]:
            status += f"  FP: {_abbreviate(r['fp'], 40)}"
            all_pass = False

        expected_str = _abbreviate(sc["expected"], 22)
        print(f"  {sc['name']:<28}  {expected_str:<22}  {tp_str:>5}  {fp_str:>5}{status}")

    print(f"  {sep}")
    overall = "PASS" if all_pass else "FAIL"
    print(
        f"  {overall}  --  {total_tp}/{total_expected} expected bottlenecks detected, "
        f"{total_fp} false positives"
    )
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
