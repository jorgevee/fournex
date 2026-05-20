"""Signal extraction + rule matching engine for CUDA static antipattern rules.

Architecture mirrors recommendations/engine.py:
  1. extract_source_signals()  — flat signal dict from parsed kernel body
  2. extract_launch_signals()  — flat signal dict from launch config / occupancy
  3. load_rules(scope)         — cached list of rule dicts from YAML files
  4. match_rules(signals, rules) — AND-logic condition matching
  5. format_finding(rule, signals) — interpolate signals into message template

Rule YAML schema (scope: kernel | launch | occupancy):

  id: uncoalesced_access
  scope: kernel          # default
  category: memory
  severity: medium       # high | medium | low
  confidence: medium     # high | medium | low  (informational, not yet used in scoring)
  message: "Strided access detected; {strided_or_pitched} confirms non-adjacent loads."
  conditions:
    strided_or_pitched: true          # bool equality
    sync_count_gte: 3                 # numeric >=
    branch_count_gt: 6                # numeric >
    local_var_count_lte: 20           # numeric <=
    occupancy_pct_lt: 25.0            # numeric <
  recommendations:                   # linked rec IDs (informational)
    - rec_ncu_improve_coalescing
  ncu_signals:                       # NCU metrics that corroborate (Stage 2 use)
    sectors_per_request_gt: 4.0
  architecture_overrides: {}         # placeholder for Stage 2 per-arch tuning
"""
from __future__ import annotations

import functools
import pathlib
import re
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from ..cuda_static import CudaKernelSource

_RULES_DIR = pathlib.Path(__file__).parent


@functools.cache
def _load_all_rules() -> dict[str, list[dict[str, Any]]]:
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for yaml_file in sorted(_RULES_DIR.rglob("*.yaml")):
        entry = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        scope = entry.get("scope", "kernel")
        by_scope.setdefault(scope, []).append(entry)
    return by_scope


def load_rules(scope: str = "kernel") -> list[dict[str, Any]]:
    """Return all loaded rules for the given scope (cached after first load)."""
    return _load_all_rules().get(scope, [])


def _count_syncs_in_loops(body: str) -> int:
    """Count __syncthreads() calls that appear inside for/while loop bodies.

    Each loop's body is extracted by matching braces. Syncs in inner loops
    are counted once per enclosing loop level — the result reflects whether
    barriers are entangled with loop structure, not how many runtime calls occur.
    """
    count = 0
    for m in re.finditer(r"\b(?:for|while)\s*\(", body):
        # Skip past the loop condition (balanced parens)
        depth = 1
        i = m.end()
        while i < len(body) and depth > 0:
            if body[i] == "(":
                depth += 1
            elif body[i] == ")":
                depth -= 1
            i += 1
        # Skip whitespace to reach the opening brace
        while i < len(body) and body[i] in " \t\n\r":
            i += 1
        if i >= len(body) or body[i] != "{":
            continue  # single-statement loop body — ignore
        # Extract the braced body
        depth = 1
        j = i + 1
        while j < len(body) and depth > 0:
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
            j += 1
        count += body[i + 1 : j - 1].count("__syncthreads()")
    return count


def extract_source_signals(kernel: "CudaKernelSource") -> dict[str, Any]:
    """Compute a flat signal dict from a parsed CudaKernelSource."""
    body = kernel.body
    lowered = body.lower()
    styles = kernel.memory_access_styles

    for_count = len(re.findall(r"\bfor\s*\(", body))
    global_access_count = len(re.findall(r"\w+\s*\[[^\]]+\]", body))
    sync_count = body.count("__syncthreads()")
    sync_in_loop_count = _count_syncs_in_loops(body)
    branch_count = len(re.findall(r"\bif\s*\(", body))
    bounds_check_count = len(re.findall(r"\bif\s*\([^)]*(<|<=)[^)]*\)", body))
    local_var_count = len(re.findall(
        r"\b(float|double|int|long|unsigned|uint|char|half)\s+\w+\s*(=|;)", body))
    max_shared_bytes = max((item["bytes"] or 0 for item in kernel.shared_memory), default=0)

    tc_unfriendly = [
        dim for item in kernel.shared_memory
        for dim in item["dims"]
        if dim.isdigit() and int(dim) > 8 and int(dim) % 16 != 0
    ]
    tc_unfriendly_hopper = [
        dim for item in kernel.shared_memory
        for dim in item["dims"]
        if dim.isdigit() and int(dim) > 8 and int(dim) % 64 != 0
    ]

    return {
        # ── Memory access styles ────────────────────────────────────────────
        "strided_or_pitched": "strided_or_pitched" in styles,
        "likely_coalesced_1d": "likely_coalesced_1d" in styles,
        "vectorized": "vectorized" in styles,
        "shared_memory_tiling": "shared_memory_tiling" in styles,
        "read_only_or_constant_cache": "read_only_or_constant_cache" in styles,

        # ── Structural flags ────────────────────────────────────────────────
        "has_shared": "__shared__" in body,
        "has_sync": "__syncthreads()" in body,
        "has_loop": bool(re.search(r"\b(for|while)\s*\(", body)),
        "has_tc": bool(re.search(r"\bwmma::|mma::|nvcuda::wmma\b", body)),
        "has_fp16": bool(re.search(
            r"\bhalf\b|\b__half\b|\bbfloat16\b|\bhalf2\b|\b__nv_bfloat16\b", body)),
        "has_matmul": bool(re.search(
            r"\+=\s*\w+\s*\[[^\]]+\]\s*\*\s*\w+\s*\[[^\]]+\]", body)),
        "has_syncwarp": "__syncwarp" in body,
        "has_thread_indexing": any(f"threadIdx.{c}" in body for c in "xyz"),
        "has_bounds_guard": bool(re.search(r"if\s*\([^)]*(<|<=)\s*[^)]*\)", body)),
        "has_indexed_global": bool(re.search(r"\w+\s*\[[^\]]*(idx|i|tid)[^\]]*\]", body)),
        "uses_cooperative_groups": "cooperative_groups" in lowered,
        "has_scalar_params": bool(re.search(r"\bfloat\b|\bint\b|\bchar\b", kernel.params)),
        "bank_conflict_risk": any(item["bank_conflict_risk"] for item in kernel.shared_memory),
        "tc_unfriendly_dims": bool(tc_unfriendly),

        # ── Pre-computed complex patterns ───────────────────────────────────
        "conditional_syncthreads_pattern": bool(
            "__syncthreads()" in body
            and re.search(
                r"if\s*\([^)]*(threadIdx|idx|tid)[^)]*\)\s*\{[^{}]*__syncthreads\s*\(",
                body, re.DOTALL,
            )
        ),
        "warp_divergence_pattern": bool(re.search(
            r"if\s*\([^)]*\bthreadIdx\.x\b\s*[%&]"
            r"|\bthreadIdx\.x\b\s*%\s*\w+\s*==",
            body,
        )),

        # ── Counts ──────────────────────────────────────────────────────────
        "sync_count": sync_count,
        "sync_in_loop_count": sync_in_loop_count,
        "for_count": for_count,
        "global_access_count": global_access_count,
        "branch_count": branch_count,
        "bounds_check_count": bounds_check_count,
        "local_var_count": local_var_count,
        "max_shared_bytes": max_shared_bytes,

        # ── String values for message interpolation ─────────────────────────
        "tc_unfriendly_dims_str": ", ".join(dict.fromkeys(tc_unfriendly)) if tc_unfriendly else "",

        # ── Hopper+ specific: wgmma requires M/N tile alignment to 64 ───────
        "tc_unfriendly_dims_hopper": bool(tc_unfriendly_hopper),
        "tc_unfriendly_dims_hopper_str": (
            ", ".join(dict.fromkeys(tc_unfriendly_hopper)) if tc_unfriendly_hopper else ""
        ),
    }


def extract_launch_signals(block_size_hint: int | None, occupancy_pct: float = 100.0) -> dict[str, Any]:
    """Compute signals from a launch configuration and occupancy estimate."""
    known = block_size_hint is not None
    hint = block_size_hint if known else -1
    return {
        "block_size": hint,
        "block_size_known": known,
        "block_size_warp_aligned": (hint % 32 == 0) if known else True,
        "occupancy_pct": occupancy_pct,
    }


def match_rules(
    signals: dict[str, Any],
    rules: list[dict[str, Any]],
    *,
    sm_version: str | None = None,
) -> list[dict[str, Any]]:
    """Return every rule whose conditions all match signals (AND logic).

    When ``sm_version`` is provided, per-architecture condition overrides and
    message overrides from ``architecture_overrides`` are merged in before matching.
    Override condition values replace the base value for the same key; other base
    conditions remain unchanged.
    """
    result = []
    for rule in rules:
        conditions = rule.get("conditions", {})
        arch_message: str | None = None
        if sm_version:
            override = rule.get("architecture_overrides", {}).get(sm_version, {})
            if override.get("conditions"):
                # Full replacement: arch override defines exactly which conditions
                # apply for this architecture, enabling signal substitution (not
                # just threshold adjustment).
                conditions = override["conditions"]
            arch_message = override.get("message")
        if _conditions_match(conditions, signals):
            if arch_message:
                rule = {**rule, "message": arch_message}
            result.append(rule)
    return result


def format_finding(rule: dict[str, Any], signals: dict[str, Any]) -> dict[str, str]:
    """Build a finding dict from a matched rule, interpolating signals into message."""
    message = rule.get("message", "")
    try:
        message = message.format_map(signals)
    except (KeyError, ValueError):
        pass
    return {"severity": rule["severity"], "code": rule["id"], "message": message}


def _conditions_match(conditions: dict[str, Any], signals: dict[str, Any]) -> bool:
    for key, expected in conditions.items():
        if key.endswith("_gte"):
            if signals.get(key[:-4], 0) < expected:
                return False
        elif key.endswith("_gt"):
            if signals.get(key[:-3], 0) <= expected:
                return False
        elif key.endswith("_lte"):
            if signals.get(key[:-4], float("inf")) > expected:
                return False
        elif key.endswith("_lt"):
            if signals.get(key[:-3], float("inf")) >= expected:
                return False
        else:
            if signals.get(key) != expected:
                return False
    return True
