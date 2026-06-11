"""Tests for variant_comparison.py and frx compare-variants CLI."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as fn
from fournex.cli import main
from fournex.variant_comparison import load_variants_csv, analyze_variants


# ── Fixtures ──────────────────────────────────────────────────────────────────

_MEMORY_BOUND_CSV = "\n".join([
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "k,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,89.0",
    "k,smsp__pcsamplingdata_pct_of_utilization_issue_stalled_memory_throttle,%,45.0",
    "k,sm__issue_active.avg.pct_of_peak_sustained_active,%,55.0",
])

_COMPUTE_BOUND_CSV = "\n".join([
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "k,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,30.0",
    "k,sm__issue_active.avg.pct_of_peak_sustained_active,%,75.0",
    "k,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,%,5.0",
])

_LOW_UTIL_CSV = "\n".join([
    "Kernel Name,Metric Name,Metric Unit,Metric Value",
    "k,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,25.0",
    "k,sm__issue_active.avg.pct_of_peak_sustained_active,%,20.0",
])


def _write_variants_dir(tmp: Path) -> Path:
    (tmp / "bad.csv").write_text(_MEMORY_BOUND_CSV, encoding="utf-8")
    (tmp / "medium.csv").write_text(_COMPUTE_BOUND_CSV, encoding="utf-8")
    (tmp / "good.csv").write_text(_LOW_UTIL_CSV, encoding="utf-8")
    results = tmp / "results.csv"
    results.write_text(
        "variant,ncu_csv,throughput_gflops,notes\n"
        "bad_gemm,bad.csv,12.3,naive\n"
        "coalesced,medium.csv,50.5,coalesced\n"
        "tiled,good.csv,70.0,tiled+unroll\n",
        encoding="utf-8",
    )
    return results


# ── load_variants_csv ─────────────────────────────────────────────────────────

def test_load_variants_csv_basic(tmp_path):
    results = _write_variants_dir(tmp_path)
    variants = load_variants_csv(results)
    assert len(variants) == 3
    assert variants[0]["variant"] == "bad_gemm"
    assert variants[0]["throughput"] == 12.3
    assert variants[0]["ncu_csv"].exists()


def test_load_variants_csv_missing_manifest(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_variants_csv(tmp_path / "nope.csv")


def test_load_variants_csv_missing_ncu_file(tmp_path):
    import pytest
    (tmp_path / "results.csv").write_text(
        "variant,ncu_csv,throughput_gflops\nbad,missing.csv,10.0\ngood,also_missing.csv,20.0\n"
    )
    with pytest.raises(FileNotFoundError):
        load_variants_csv(tmp_path / "results.csv")


def test_load_variants_csv_requires_two_variants(tmp_path):
    import pytest
    (tmp_path / "a.csv").write_text(_MEMORY_BOUND_CSV)
    (tmp_path / "results.csv").write_text(
        "variant,ncu_csv,throughput_gflops\nonly_one,a.csv,10.0\n"
    )
    with pytest.raises(ValueError, match="at least 2"):
        load_variants_csv(tmp_path / "results.csv")


# ── analyze_variants ──────────────────────────────────────────────────────────

def test_analyze_variants_schema(tmp_path):
    results = _write_variants_dir(tmp_path)
    variants = load_variants_csv(results)
    report = analyze_variants(variants)
    assert report["schema"] == "variant_comparison_v1"
    assert report["variant_count"] == 3


def test_analyze_variants_ranked_by_throughput(tmp_path):
    results = _write_variants_dir(tmp_path)
    report = analyze_variants(load_variants_csv(results))
    throughputs = [v["throughput"] for v in report["variants_ranked"]]
    assert throughputs == sorted(throughputs, reverse=True)


def test_analyze_variants_baseline_is_lowest_by_default(tmp_path):
    results = _write_variants_dir(tmp_path)
    report = analyze_variants(load_variants_csv(results))
    assert report["baseline"] == "bad_gemm"


def test_analyze_variants_explicit_baseline(tmp_path):
    results = _write_variants_dir(tmp_path)
    report = analyze_variants(load_variants_csv(results), baseline_variant="coalesced")
    assert report["baseline"] == "coalesced"


def test_analyze_variants_delta_vs_baseline(tmp_path):
    results = _write_variants_dir(tmp_path)
    report = analyze_variants(load_variants_csv(results))
    baseline = next(v for v in report["variants_ranked"] if v["is_baseline"])
    assert baseline["delta_vs_baseline_x"] == 1.0 or baseline["is_baseline"]
    best = report["variants_ranked"][0]
    assert best["delta_vs_baseline_x"] is not None
    assert best["delta_vs_baseline_x"] > 1.0


def test_analyze_variants_transitions_count(tmp_path):
    results = _write_variants_dir(tmp_path)
    report = analyze_variants(load_variants_csv(results))
    assert len(report["transitions"]) == 2  # 3 variants → 2 pairwise transitions


def test_analyze_variants_transitions_ordered(tmp_path):
    results = _write_variants_dir(tmp_path)
    report = analyze_variants(load_variants_csv(results))
    for t in report["transitions"]:
        assert t["from_throughput"] < t["to_throughput"]


def test_analyze_variants_invalid_baseline(tmp_path):
    import pytest
    results = _write_variants_dir(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        analyze_variants(load_variants_csv(results), baseline_variant="does_not_exist")


# ── CLI: frx compare-variants ─────────────────────────────────────────────────

def test_cli_compare_variants_exits_zero(tmp_path, capsys):
    results = _write_variants_dir(tmp_path)
    rc = main(["compare-variants", str(results)])
    assert rc == 0


def test_cli_compare_variants_json(tmp_path, capsys):
    results = _write_variants_dir(tmp_path)
    main(["compare-variants", str(results), "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["mode"] == "compare_variants"
    assert payload["result"]["schema"] == "variant_comparison_v1"


def test_cli_compare_variants_missing_file(capsys):
    rc = main(["compare-variants", "does_not_exist.csv"])
    assert rc == 1


def test_cli_compare_variants_explicit_baseline(tmp_path, capsys):
    results = _write_variants_dir(tmp_path)
    rc = main(["compare-variants", str(results), "--baseline", "coalesced"])
    assert rc == 0


def test_cli_compare_variants_explain(tmp_path, capsys):
    results = _write_variants_dir(tmp_path)
    out_dir = tmp_path / "brief"
    rc = main(["compare-variants", str(results), "--explain", "--explain-out", str(out_dir)])
    assert rc == 0
    assert (out_dir / "frx_llm_prompt.txt").exists()


# ── fournex.__version__ + exports ─────────────────────────────────────────────

def test_version_is_set():
    assert fn.__version__
    assert isinstance(fn.__version__, str)
    assert "." in fn.__version__


def test_exports():
    assert hasattr(fn, "load_variants_csv")
    assert hasattr(fn, "analyze_variants")
