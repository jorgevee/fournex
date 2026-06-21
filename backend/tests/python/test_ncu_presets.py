import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.ncu_presets import (
    build_ncu_command,
    filter_metrics_for_sm,
    get_ncu_preset,
    pc_sampling_supported,
)

_PC = "smsp__pcsamplingdata_"


def _has_pc(metrics) -> bool:
    return any(m.startswith(_PC) for m in metrics)


def test_memory_preset_contains_pc_sampling_by_default():
    # Guards the premise of the gating: the memory preset really does carry
    # PC-sampling metrics that Blackwell rejects.
    assert _has_pc(get_ncu_preset("memory").metrics)


def test_pc_sampling_supported_by_arch():
    assert pc_sampling_supported(None) is True       # unknown -> don't drop
    assert pc_sampling_supported("sm_80") is True     # Ampere
    assert pc_sampling_supported("sm_90") is True      # Hopper
    assert pc_sampling_supported(90) is True
    assert pc_sampling_supported("sm_100") is False    # Blackwell DC
    assert pc_sampling_supported("sm_120") is False    # Blackwell consumer (RTX 50xx)
    assert pc_sampling_supported(120) is False


def test_filter_drops_pc_sampling_only_on_blackwell():
    metrics = get_ncu_preset("memory").metrics
    # Hopper: unchanged.
    assert filter_metrics_for_sm(metrics, "sm_90") == metrics
    # Blackwell: PC-sampling stripped, non-PC metrics retained.
    filtered = filter_metrics_for_sm(metrics, "sm_120")
    assert not _has_pc(filtered)
    assert "dram__throughput.avg.pct_of_peak_sustained_elapsed" in filtered
    assert len(filtered) < len(metrics)


def test_build_ncu_command_gates_metrics_for_sm_120():
    cmd_hopper = build_ncu_command("memory", ["./app"], sm_version="sm_90")
    cmd_black = build_ncu_command("memory", ["./app"], sm_version="sm_120")
    assert _PC in ",".join(cmd_hopper)
    assert _PC not in ",".join(cmd_black)
    # Still a valid, non-empty metrics request on Blackwell.
    assert "--metrics" in cmd_black
    assert "dram__throughput.avg.pct_of_peak_sustained_elapsed" in ",".join(cmd_black)


def test_build_ncu_command_unknown_arch_keeps_all_metrics():
    cmd = build_ncu_command("full", ["./app"])  # sm_version=None
    assert _PC in ",".join(cmd)


def test_memory_and_full_presets_contain_kernel_duration():
    # gpu__time_duration.sum is the basis for a trustworthy bench speedup verdict.
    assert "gpu__time_duration.sum" in get_ncu_preset("memory").metrics
    assert "gpu__time_duration.sum" in get_ncu_preset("full").metrics


def test_kernel_duration_survives_blackwell_gating():
    # It is not a PC-sampling metric, so it must NOT be dropped on sm_120.
    filtered = filter_metrics_for_sm(get_ncu_preset("full").metrics, "sm_120")
    assert "gpu__time_duration.sum" in filtered
