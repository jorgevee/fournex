import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import fournex as fn
from fournex.arch_profiles import detect_gpu_model, get_arch_profile, list_known_gpus, load_arch_profile_overrides, resolve_sm_version
from fournex.cuda_rules.engine import extract_source_signals, load_rules, match_rules
from fournex.cuda_static import inspect_cuda_source, parse_cuda_kernels
from fournex.kernel_inspector import device_limits_for_gpu


# ── resolve_sm_version ────────────────────────────────────────────────────────

def test_resolve_h100():
    assert resolve_sm_version("h100") == "sm_90"

def test_resolve_sm90_passthrough():
    assert resolve_sm_version("sm_90") == "sm_90"

def test_resolve_sm90_case_insensitive():
    assert resolve_sm_version("SM_90") == "sm_90"

def test_resolve_rtx4090():
    assert resolve_sm_version("rtx4090") == "sm_89"

def test_resolve_t4():
    assert resolve_sm_version("t4") == "sm_75"


# ── detect_gpu_model ──────────────────────────────────────────────────────────

def test_detect_h100_full_name():
    assert detect_gpu_model("NVIDIA H100 80GB HBM3") == "h100"

def test_detect_a100_sxm():
    assert detect_gpu_model("NVIDIA A100-SXM4-80GB") == "a100"

def test_detect_rtx5060():
    assert detect_gpu_model("NVIDIA GeForce RTX 5060") == "rtx5060"

def test_detect_rtx4090():
    assert detect_gpu_model("NVIDIA GeForce RTX 4090") == "rtx4090"

def test_detect_t4():
    assert detect_gpu_model("Tesla T4") == "t4"

def test_detect_l4():
    assert detect_gpu_model("NVIDIA L4") == "l4"

def test_detect_prefers_longer_match():
    # "rtx3090ti" must win over "rtx3090" for a 3090 Ti device
    assert detect_gpu_model("NVIDIA GeForce RTX 3090 Ti") == "rtx3090ti"

def test_detect_unknown_returns_none():
    assert detect_gpu_model("NVIDIA Quadro P4000") is None

def test_detect_none_returns_none():
    assert detect_gpu_model(None) is None

def test_detect_exported_from_fournex():
    import fournex as fn
    assert hasattr(fn, "detect_gpu_model")
    assert fn.detect_gpu_model("NVIDIA H100 80GB HBM3") == "h100"

def test_resolve_unknown_returns_none():
    assert resolve_sm_version("unknown_gpu_xyz") is None

def test_resolve_none_returns_none():
    assert resolve_sm_version(None) is None

def test_resolve_sm120():
    assert resolve_sm_version("sm_120") == "sm_120"

def test_resolve_rtx5090():
    assert resolve_sm_version("rtx5090") == "sm_120"

def test_resolve_a100():
    assert resolve_sm_version("a100") == "sm_80"

def test_resolve_l4():
    assert resolve_sm_version("l4") == "sm_89"


# ── get_arch_profile ──────────────────────────────────────────────────────────

def test_profile_h100_is_hopper():
    profile = get_arch_profile("h100")
    assert profile["arch_family"] == "hopper"

def test_profile_t4_no_bf16():
    profile = get_arch_profile("t4")
    assert profile["bf16_supported"] is False

def test_profile_rtx4090_has_fp8():
    profile = get_arch_profile("rtx4090")
    assert profile["fp8_supported"] is True

def test_profile_none_returns_default_ampere():
    profile = get_arch_profile(None)
    assert profile["arch_family"] == "ampere"

def test_profile_hopper_higher_register_threshold():
    turing = get_arch_profile("t4")
    hopper = get_arch_profile("h100")
    assert hopper["register_pressure_var_threshold"] > turing["register_pressure_var_threshold"]

def test_profile_hopper_higher_shared_memory_limit():
    ampere = get_arch_profile("a100")
    hopper = get_arch_profile("h100")
    assert hopper["shared_memory_static_limit"] > ampere["shared_memory_static_limit"]

def test_profile_hopper_tensor_core_min_dim_64():
    assert get_arch_profile("h100")["tensor_core_min_dim"] == 64

def test_profile_override_dict_by_product_name():
    profile = get_arch_profile("h100", {"profiles": {"h100": {"peak_fp32_tflops": 61.5}}})
    assert profile["arch_family"] == "hopper"
    assert profile["peak_fp32_tflops"] == 61.5

def test_profile_override_dict_by_sm_version():
    profile = get_arch_profile("h100", {"profiles": {"sm_90": {"peak_memory_bw_gbps": 3900.0}}})
    assert profile["peak_memory_bw_gbps"] == 3900.0

def test_profile_override_top_level_scalars():
    profile = get_arch_profile("a100", {"peak_fp16_tflops": 280.0})
    assert profile["peak_fp16_tflops"] == 280.0

def test_load_arch_profile_overrides_yaml(tmp_path):
    path = tmp_path / "arch.yaml"
    path.write_text("profiles:\n  h100:\n    peak_fp32_tflops: 60.0\n", encoding="utf-8")
    overrides = load_arch_profile_overrides(path)
    assert overrides["profiles"]["h100"]["peak_fp32_tflops"] == 60.0

def test_get_arch_profile_accepts_yaml_path(tmp_path):
    path = tmp_path / "arch.yaml"
    path.write_text("profiles:\n  h100:\n    peak_fp32_tflops: 60.0\n", encoding="utf-8")
    assert get_arch_profile("h100", path)["peak_fp32_tflops"] == 60.0

def test_list_known_gpus_includes_products_and_sm():
    gpus = list_known_gpus()
    assert "h100" in gpus
    assert "sm_90" in gpus
    assert "t4" in gpus


# ── Architecture overrides in rule matching ───────────────────────────────────

_SHARED_MEM_65KB = "__global__ void k(float* a) { __shared__ float tile[16384]; int i = threadIdx.x; a[i] = tile[i]; }"
# 16384 * 4 = 65536 bytes = 64 KB

def test_large_shared_flags_on_turing():
    r = inspect_cuda_source(_SHARED_MEM_65KB, gpu_model="t4")
    codes = {f["code"] for f in r["findings"]}
    assert "large_static_shared_memory" in codes

def test_large_shared_does_not_flag_on_hopper():
    r = inspect_cuda_source(_SHARED_MEM_65KB, gpu_model="h100")
    codes = {f["code"] for f in r["findings"]}
    assert "large_static_shared_memory" not in codes

_HIGH_REG_KERNEL = "__global__ void k(float* a) { " + " ".join(f"float v{i} = a[{i}];" for i in range(25)) + " a[0] = v0; }"

def test_register_pressure_flags_on_turing():
    r = inspect_cuda_source(_HIGH_REG_KERNEL, gpu_model="t4")
    codes = {f["code"] for f in r["findings"]}
    assert "high_register_pressure" in codes

def test_register_pressure_does_not_flag_on_hopper():
    # 25 vars is > 20 (Turing threshold) but < 32 (Hopper threshold)
    r = inspect_cuda_source(_HIGH_REG_KERNEL, gpu_model="h100")
    codes = {f["code"] for f in r["findings"]}
    assert "high_register_pressure" not in codes

_TC_DIM_48_KERNEL = "__global__ void k(float* a) { __shared__ half tile[48][48]; a[threadIdx.x] = tile[0][threadIdx.x]; }"
# dim 48: % 16 == 0 (OK for generic WMMA), but % 64 != 0 (bad for Hopper wgmma)

def test_tc_unfriendly_dim_flags_on_hopper():
    r = inspect_cuda_source(_TC_DIM_48_KERNEL, gpu_model="h100")
    codes = {f["code"] for f in r["findings"]}
    assert "dimensions_not_tensor_core_friendly" in codes

def test_tc_unfriendly_dim_does_not_flag_generic_for_48():
    # 48 % 16 == 0, so no flag without arch override
    r = inspect_cuda_source(_TC_DIM_48_KERNEL)  # no gpu_model
    codes = {f["code"] for f in r["findings"]}
    assert "dimensions_not_tensor_core_friendly" not in codes

_TC_DIM_12_KERNEL = "__global__ void k(float* a) { __shared__ half tile[12][12]; a[threadIdx.x] = tile[0][threadIdx.x]; }"
# dim 12: % 16 != 0 — should flag on all arches

def test_tc_unfriendly_dim_12_flags_on_turing():
    r = inspect_cuda_source(_TC_DIM_12_KERNEL, gpu_model="t4")
    codes = {f["code"] for f in r["findings"]}
    assert "dimensions_not_tensor_core_friendly" in codes


# ── device_limits_for_gpu — sm_version strings ────────────────────────────────

def test_device_limits_sm80_like_a100():
    sm80 = device_limits_for_gpu("sm_80")
    a100 = device_limits_for_gpu("a100")
    assert sm80["max_threads_per_sm"] == a100["max_threads_per_sm"]

def test_device_limits_sm100_no_crash():
    limits = device_limits_for_gpu("sm_100")
    assert "max_threads_per_sm" in limits


# ── Public API exports ────────────────────────────────────────────────────────

def test_resolve_sm_version_exported():
    assert fn.resolve_sm_version("h100") == "sm_90"

def test_get_arch_profile_exported():
    assert fn.get_arch_profile("t4")["arch_family"] == "turing"

def test_load_arch_profile_overrides_exported(tmp_path):
    path = tmp_path / "arch.yaml"
    path.write_text("peak_fp32_tflops: 42.0\n", encoding="utf-8")
    assert fn.load_arch_profile_overrides(path)["peak_fp32_tflops"] == 42.0


# ── Smoke tests ───────────────────────────────────────────────────────────────

def test_inspect_cuda_source_with_h100():
    r = inspect_cuda_source(_SHARED_MEM_65KB, gpu_model="h100")
    assert r["schema_version"] == "cuda_static_v1"
    assert r["kernel_count"] == 1

def test_inspect_cuda_source_with_sm90():
    r = inspect_cuda_source(_SHARED_MEM_65KB, gpu_model="sm_90")
    assert r["kernel_count"] == 1
