"""GPU architecture scoring profiles for Fournex static analysis.

Separates scoring calibration (when to flag) from hardware limits (what the GPU can do).
`kernel_inspector.py` owns hardware facts; this module owns threshold tuning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# ── Per-sm scoring profiles ───────────────────────────────────────────────────

_PROFILES: dict[str, dict[str, Any]] = {
    "sm_75": {
        "arch_family": "turing",
        "display_name": "Turing (sm_75)",
        "register_pressure_var_threshold": 20,  # > N local vars triggers high_register_pressure
        "shared_memory_static_limit": 49152,    # 48 KB — flag static alloc above this
        "tensor_core_min_dim": 16,              # WMMA requires multiples of 16
        "bf16_supported": False,                # Turing has no BF16 tensor cores
        "fp8_supported": False,
        "occupancy_low_pct": 25.0,
        "warp_size": 32,
        # Roofline hardware specs — T4 reference; override via YAML arch profile for other SKUs
        "peak_fp32_tflops": 8.1,
        "peak_fp16_tflops": 65.0,
        "peak_memory_bw_gbps": 320.0,
    },
    "sm_80": {
        "arch_family": "ampere",
        "display_name": "Ampere A-class (sm_80)",
        "register_pressure_var_threshold": 20,
        "shared_memory_static_limit": 49152,    # 48 KB default; dynamic alloc can reach 164 KB
        "tensor_core_min_dim": 16,
        "bf16_supported": True,
        "fp8_supported": False,
        "occupancy_low_pct": 25.0,
        "warp_size": 32,
        # A100 80GB reference
        "peak_fp32_tflops": 19.5,
        "peak_fp16_tflops": 312.0,
        "peak_memory_bw_gbps": 2000.0,
    },
    "sm_86": {
        "arch_family": "ampere",
        "display_name": "Ampere GA10x (sm_86)",
        "register_pressure_var_threshold": 20,
        "shared_memory_static_limit": 49152,
        "tensor_core_min_dim": 16,
        "bf16_supported": True,
        "fp8_supported": False,
        "occupancy_low_pct": 25.0,
        "warp_size": 32,
        # RTX 3090 reference
        "peak_fp32_tflops": 35.6,
        "peak_fp16_tflops": 142.6,
        "peak_memory_bw_gbps": 936.0,
    },
    "sm_89": {
        "arch_family": "ada",
        "display_name": "Ada Lovelace (sm_89)",
        "register_pressure_var_threshold": 20,
        "shared_memory_static_limit": 49152,
        "tensor_core_min_dim": 16,
        "bf16_supported": True,
        "fp8_supported": True,
        "occupancy_low_pct": 25.0,
        "warp_size": 32,
        # RTX 4090 reference
        "peak_fp32_tflops": 82.6,
        "peak_fp16_tflops": 165.2,
        "peak_memory_bw_gbps": 1008.0,
    },
    "sm_90": {
        "arch_family": "hopper",
        "display_name": "Hopper (sm_90)",
        # Hopper SM supports 2048 threads; with 65536 regs/SM that's 32 regs/thread at full
        # occupancy. More register budget per thread before occupancy drops: raise threshold.
        "register_pressure_var_threshold": 32,
        # H100 has 228 KB shared mem per SM and 32 blocks per SM.
        # Even 96 KB per block gives 2 blocks → 50% occupancy at 1024 threads/block.
        # Flag static alloc above 96 KB as potentially limiting.
        "shared_memory_static_limit": 98304,    # 96 KB
        "tensor_core_min_dim": 64,              # wgmma requires M/N multiples of 64
        "bf16_supported": True,
        "fp8_supported": True,
        "occupancy_low_pct": 25.0,
        "warp_size": 32,
        # H100 SXM5 reference
        "peak_fp32_tflops": 67.0,
        "peak_fp16_tflops": 989.0,
        "peak_memory_bw_gbps": 3350.0,
    },
    "sm_100": {
        "arch_family": "blackwell",
        "display_name": "Blackwell (sm_100) — preliminary",
        "register_pressure_var_threshold": 32,
        "shared_memory_static_limit": 131072,   # 128 KB — preliminary estimate
        "tensor_core_min_dim": 64,
        "bf16_supported": True,
        "fp8_supported": True,
        "occupancy_low_pct": 25.0,
        "warp_size": 32,
        # B100 reference (preliminary)
        "peak_fp32_tflops": 60.0,
        "peak_fp16_tflops": 1800.0,
        "peak_memory_bw_gbps": 8000.0,
    },
    "sm_120": {
        "arch_family": "blackwell",
        "display_name": "Blackwell (sm_120) — preliminary",
        "register_pressure_var_threshold": 32,
        "shared_memory_static_limit": 131072,
        "tensor_core_min_dim": 64,
        "bf16_supported": True,
        "fp8_supported": True,
        "occupancy_low_pct": 25.0,
        "warp_size": 32,
        # RTX 5090 reference (preliminary)
        "peak_fp32_tflops": 104.8,
        "peak_fp16_tflops": 209.6,
        "peak_memory_bw_gbps": 1792.0,
    },
}

_DEFAULT_PROFILE: dict[str, Any] = _PROFILES["sm_80"]

# ── Product name → sm version ──────────────────────────────────────────────────

_PRODUCT_TO_SM: dict[str, str] = {
    # Turing
    "t4": "sm_75",
    "rtx2080": "sm_75",
    "rtx2080ti": "sm_75",
    "rtx2070": "sm_75",
    "rtx2060": "sm_75",
    # Ampere A-class (GA100)
    "a100": "sm_80",
    "a30": "sm_80",
    # Ampere GA10x consumer / workstation
    "a10": "sm_86",
    "a16": "sm_86",
    "rtx3090": "sm_86",
    "rtx3090ti": "sm_86",
    "rtx3080": "sm_86",
    "rtx3070": "sm_86",
    "rtx3060": "sm_86",
    # Ada Lovelace
    "l4": "sm_89",
    "l40": "sm_89",
    "l40s": "sm_89",
    "rtx4090": "sm_89",
    "rtx4080": "sm_89",
    "rtx4070": "sm_89",
    "rtx4060": "sm_89",
    # Hopper
    "h100": "sm_90",
    "h200": "sm_90",
    # Blackwell
    "b100": "sm_100",
    "b200": "sm_100",
    "rtx5090": "sm_120",
    "rtx5080": "sm_120",
    "rtx5070ti": "sm_120",
    "rtx5070": "sm_120",
    "rtx5060ti": "sm_120",
    "rtx5060": "sm_120",
}


# ── Public API ────────────────────────────────────────────────────────────────

def resolve_sm_version(gpu_model: str | None) -> str | None:
    """Normalize a GPU model string to a canonical sm version, e.g. "h100" → "sm_90"."""
    if not gpu_model:
        return None
    normalized = gpu_model.lower().strip().replace(" ", "").replace("-", "")
    if normalized.startswith("sm_"):
        return normalized if normalized in _PROFILES else None
    return _PRODUCT_TO_SM.get(normalized)


def detect_gpu_model(gpu_name: str | None) -> str | None:
    """Infer a canonical GPU model key from a raw device name string.

    Matches the full device name (e.g. "NVIDIA H100 80GB HBM3") against the
    known product keys in ``_PRODUCT_TO_SM`` by checking whether any key is a
    substring of the normalized device name. Returns the longest matching key so
    that "rtx3090ti" wins over "rtx3090" for an RTX 3090 Ti device.

    Returns ``None`` when no known product key matches.
    """
    if not gpu_name:
        return None
    normalized = gpu_name.lower().replace("nvidia ", "").replace(" ", "").replace("-", "")
    # Longest match wins (avoids "a100" matching "a10" prefix)
    best: str | None = None
    for key in _PRODUCT_TO_SM:
        if key in normalized:
            if best is None or len(key) > len(best):
                best = key
    return best


def load_arch_profile_overrides(path: str | Path | None) -> dict[str, Any]:
    """Load user arch profile overrides from a YAML file.

    Supported shapes:

    - top-level profile values, applied to the selected GPU:
      ``peak_fp32_tflops: 82.6``
    - keyed profiles:
      ``profiles: {h100: {peak_fp32_tflops: 60.0}}``
    - shorthand keyed profiles:
      ``h100: {peak_fp32_tflops: 60.0}``

    Note: top-level scalar overrides are applied last and unconditionally, so
    they win for *every* GPU. Don't combine them with keyed ``profiles`` unless
    you intend the scalar to override the keyed value for all hardware.
    """
    if path is None:
        return {}
    override_path = Path(path)
    if not override_path.exists():
        raise FileNotFoundError(f"arch profile override file not found: {override_path}")

    import yaml

    data = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("arch profile override YAML must contain a mapping")
    return data


def get_arch_profile(
    gpu_model: str | None,
    overrides: dict[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Return the scoring profile for a GPU model, or the default (Ampere) if unknown.

    ``overrides`` may be either a dict or a YAML path. Override keys can be the
    explicit GPU model ("h100") or the resolved SM version ("sm_90"). Top-level
    scalar fields are also accepted for one-off custom hardware specs.
    """
    sm = resolve_sm_version(gpu_model)
    base = _PROFILES.get(sm, _DEFAULT_PROFILE) if sm else _DEFAULT_PROFILE
    profile = dict(base)
    override_map = (
        load_arch_profile_overrides(overrides)
        if isinstance(overrides, (str, Path))
        else (overrides or {})
    )
    if override_map:
        profile.update(_select_profile_override(gpu_model, sm, override_map))
    return profile


def list_known_gpus() -> list[str]:
    """Return all recognized product names and sm version strings."""
    return sorted(set(_PRODUCT_TO_SM) | set(_PROFILES))


def _select_profile_override(
    gpu_model: str | None,
    sm: str | None,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    profiles = overrides.get("profiles", overrides)
    selected: dict[str, Any] = {}

    if isinstance(profiles, dict):
        candidate_keys = []
        if gpu_model:
            candidate_keys.append(_normalize_gpu_key(gpu_model))
        if sm:
            candidate_keys.append(_normalize_gpu_key(sm))

        normalized_profiles = {
            _normalize_gpu_key(key): value
            for key, value in profiles.items()
            if isinstance(value, dict)
        }
        for key in candidate_keys:
            selected.update(normalized_profiles.get(key, {}))

    top_level_scalars = {
        key: value
        for key, value in overrides.items()
        if key != "profiles" and not isinstance(value, dict)
    }
    selected.update(top_level_scalars)
    return selected


def _normalize_gpu_key(value: str) -> str:
    return value.lower().strip().replace(" ", "").replace("-", "")
