"""GPU architecture scoring profiles for Fournex static analysis.

Separates scoring calibration (when to flag) from hardware limits (what the GPU can do).
`kernel_inspector.py` owns hardware facts; this module owns threshold tuning.
"""
from __future__ import annotations

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


def get_arch_profile(gpu_model: str | None) -> dict[str, Any]:
    """Return the scoring profile for a GPU model, or the default (Ampere) if unknown."""
    sm = resolve_sm_version(gpu_model)
    return _PROFILES.get(sm, _DEFAULT_PROFILE) if sm else _DEFAULT_PROFILE


def list_known_gpus() -> list[str]:
    """Return all recognized product names and sm version strings."""
    return sorted(set(_PRODUCT_TO_SM) | set(_PROFILES))
