from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


# ── Instruction classification ────────────────────────────────────────────────

_INSTRUCTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bld\.global\b"),                              "global_loads"),
    (re.compile(r"\bst\.global\b"),                              "global_stores"),
    (re.compile(r"\bld\.shared\b"),                              "shared_loads"),
    (re.compile(r"\bst\.shared\b"),                              "shared_stores"),
    (re.compile(r"\bld\.local\b"),                               "local_loads"),
    (re.compile(r"\bst\.local\b"),                               "local_stores"),
    (re.compile(r"\bwmma\.\b"),                                  "tensor_ops"),
    (re.compile(r"\b(?:sin|cos|rsqrt|ex2|lg2)\.approx\b"),      "special_func"),
    (re.compile(r"\batom\.\b|\bred\.\b"),                        "atomic_ops"),
    (re.compile(r"\bbar\.\b|\bmembar\.\b"),                      "barrier_ops"),
    (re.compile(r"\b(?:bra|ret|exit|call)\b"),                   "control_flow"),
    (re.compile(r"\bcvta?\.\b"),                                 "conversions"),
    (re.compile(r"\bsetp\.\b|\bselp\.\b"),                       "comparisons"),
    (re.compile(r"\.f64\b"),                                     "fp64_ops"),
    (re.compile(r"\.(?:f16|bf16)\b"),                            "fp16_ops"),
    (re.compile(r"\.f32\b"),                                     "fp32_ops"),
    (re.compile(r"\.[subu]\d+\b|\.b\d+\b"),                      "int_ops"),
]

_ENTRY_RE = re.compile(
    r"\.(?:visible\s+|extern\s+)?\.entry\s+(?P<name>\w+)",
    re.MULTILINE,
)
_REG_RE = re.compile(r"\.reg\s+\.(?P<type>\w+)\s+%\w+<(?P<n>\d+)>")
_LOCAL_RE = re.compile(r"\.local\s+\.align\s+\d+\s+\.b8\s+\w+\[(?P<size>\d+)\]")
_LABEL_RE = re.compile(r"^\s*\$?[\w.]+:\s*$", re.MULTILINE)
_COND_BRANCH_RE = re.compile(r"@%\w+\s+bra\b")
_BRA_TARGET_RE = re.compile(r"\bbra(?:\.uni)?\s+(?P<label>\$?[\w.]+)")
_VERSION_RE = re.compile(r"\.version\s+([\d.]+)")
_TARGET_RE = re.compile(r"\.target\s+(\w+)")

# 64-bit register types consume 2 hardware registers each
_DOUBLE_WIDE = frozenset({"f64", "b64", "u64", "s64"})
# Predicate registers don't draw from the hardware register file
_PRED_TYPE = "pred"


@dataclass(slots=True)
class PtxKernelAnalysis:
    kernel_name: str
    # Register pressure
    register_count: int
    register_breakdown: dict[str, int]
    # Local memory / spills
    local_memory_bytes: int
    has_register_spills: bool
    spill_load_count: int
    spill_store_count: int
    # Instruction mix
    instruction_count: int
    instruction_mix: dict[str, int]
    # Memory convenience accessors (duplicated from instruction_mix for API clarity)
    global_load_count: int
    global_store_count: int
    shared_load_count: int
    shared_store_count: int
    # Control flow
    branch_count: int
    conditional_branch_count: int
    estimated_loop_count: int
    # Capability flags
    has_tensor_ops: bool
    has_special_function_ops: bool
    has_fp64: bool
    has_atomics: bool
    # Findings
    findings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_ptx_text(text: str) -> list[PtxKernelAnalysis]:
    results: list[PtxKernelAnalysis] = []
    entries = list(_ENTRY_RE.finditer(text))
    for i, match in enumerate(entries):
        name = match.group("name")
        block_start = text.find("{", match.end())
        if block_start == -1:
            continue
        block_end = _find_matching_brace(text, block_start)
        if block_end is None:
            continue
        body = text[block_start + 1:block_end]
        analysis = _analyze_kernel_body(name, body)
        results.append(analysis)
    return results


def analyze_ptx_text(text: str, *, filename: str = "<memory>") -> dict[str, Any]:
    kernels = parse_ptx_text(text)
    all_findings: list[dict[str, Any]] = []
    for k in kernels:
        for f in k.findings:
            all_findings.append({**f, "kernel_name": k.kernel_name})

    ptx_version = _first_match(_VERSION_RE, text)
    target = _first_match(_TARGET_RE, text)

    any_spills = any(k.has_register_spills for k in kernels)
    total_spill_loads = sum(k.spill_load_count for k in kernels)
    total_spill_stores = sum(k.spill_store_count for k in kernels)
    total_instructions = sum(k.instruction_count for k in kernels)
    avg_regs = (
        round(sum(k.register_count for k in kernels) / len(kernels), 1) if kernels else 0.0
    )

    # Dominant instruction category across all kernels
    combined_mix: dict[str, int] = {}
    for k in kernels:
        for cat, cnt in k.instruction_mix.items():
            combined_mix[cat] = combined_mix.get(cat, 0) + cnt
    dominant = max(combined_mix, key=combined_mix.__getitem__) if combined_mix else "unknown"

    return {
        "schema": "ptx_analysis_v1",
        "filename": filename,
        "ptx_version": ptx_version,
        "target": target,
        "kernel_count": len(kernels),
        "kernels": [k.to_dict() for k in kernels],
        "findings": sorted(all_findings, key=lambda f: _severity_rank(f["severity"])),
        "run_summary": {
            "total_instructions": total_instructions,
            "avg_register_count": avg_regs,
            "any_spills": any_spills,
            "total_spill_loads": total_spill_loads,
            "total_spill_stores": total_spill_stores,
            "dominant_instruction_category": dominant,
            "has_fp64": any(k.has_fp64 for k in kernels),
            "has_tensor_ops": any(k.has_tensor_ops for k in kernels),
        },
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _analyze_kernel_body(name: str, body: str) -> PtxKernelAnalysis:
    # Register declarations
    reg_breakdown: dict[str, int] = {}
    register_count = 0
    for m in _REG_RE.finditer(body):
        rtype = m.group("type")
        n = int(m.group("n"))
        if rtype == _PRED_TYPE:
            continue
        weight = 2 if rtype in _DOUBLE_WIDE else 1
        reg_breakdown[rtype] = reg_breakdown.get(rtype, 0) + n
        register_count += n * weight

    # Local memory (register spills)
    local_memory_bytes = sum(int(m.group("size")) for m in _LOCAL_RE.finditer(body))
    has_register_spills = local_memory_bytes > 0

    # Instruction mix — strip declarations and labels, count remaining lines
    mix: dict[str, int] = {}
    branch_count = 0
    conditional_branch_count = 0
    label_positions: dict[str, int] = {}
    branch_targets: list[tuple[int, str]] = []  # (line_no, target)

    lines = body.splitlines()
    instruction_line_no = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        # Label definition
        if re.match(r"^\$?[\w.]+:\s*$", line):
            label_name = line.rstrip(":").strip()
            label_positions[label_name] = instruction_line_no
            continue
        # PTX directive (declaration) — skip
        if line.startswith("."):
            continue

        instruction_line_no += 1
        # Classify
        category = "other"
        for pattern, cat in _INSTRUCTION_PATTERNS:
            if pattern.search(line):
                category = cat
                break
        mix[category] = mix.get(category, 0) + 1

        # Branch tracking
        if re.search(r"\bbra\b", line):
            branch_count += 1
            if _COND_BRANCH_RE.search(line):
                conditional_branch_count += 1
            bra_m = _BRA_TARGET_RE.search(line)
            if bra_m:
                branch_targets.append((instruction_line_no, bra_m.group("label")))

    instruction_count = instruction_line_no

    # Back-edge detection (loops): branch target label defined before branch line
    estimated_loop_count = len({
        target for line_no, target in branch_targets
        if label_positions.get(target, instruction_count + 1) < line_no
    })

    analysis = PtxKernelAnalysis(
        kernel_name=name,
        register_count=register_count,
        register_breakdown=reg_breakdown,
        local_memory_bytes=local_memory_bytes,
        has_register_spills=has_register_spills,
        spill_load_count=mix.get("local_loads", 0),
        spill_store_count=mix.get("local_stores", 0),
        instruction_count=instruction_count,
        instruction_mix=mix,
        global_load_count=mix.get("global_loads", 0),
        global_store_count=mix.get("global_stores", 0),
        shared_load_count=mix.get("shared_loads", 0),
        shared_store_count=mix.get("shared_stores", 0),
        branch_count=branch_count,
        conditional_branch_count=conditional_branch_count,
        estimated_loop_count=estimated_loop_count,
        has_tensor_ops=mix.get("tensor_ops", 0) > 0,
        has_special_function_ops=mix.get("special_func", 0) > 0,
        has_fp64=mix.get("fp64_ops", 0) > 0,
        has_atomics=mix.get("atomic_ops", 0) > 0,
        findings=[],
    )
    analysis.findings = _ptx_findings(analysis)
    return analysis


def _ptx_findings(a: PtxKernelAnalysis) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    if a.has_register_spills:
        findings.append(_finding(
            "high", "register_spills_detected",
            f"Kernel has {a.local_memory_bytes} bytes of local memory — registers are spilling to local (L1/DRAM) memory.",
            "Lower loop unroll factor, split the kernel into smaller passes, or use __launch_bounds__ to cap register allocation.",
        ))

    if a.register_count > 128:
        findings.append(_finding(
            "high", "very_high_register_count",
            f"Very high virtual register count ({a.register_count}); likely to spill at runtime.",
            "Use __launch_bounds__(MAX_THREADS, MIN_BLOCKS) to constrain register allocation. Check with Nsight Compute for actual registers_per_thread.",
        ))
    elif a.register_count > 64:
        findings.append(_finding(
            "medium", "high_register_count",
            f"Elevated virtual register count ({a.register_count}); may reduce occupancy.",
            "Monitor occupancy with Nsight Compute. Consider splitting work across smaller kernels if occupancy is low.",
        ))

    if a.has_fp64:
        findings.append(_finding(
            "medium", "fp64_detected",
            f"FP64 operations detected ({a.instruction_mix.get('fp64_ops', 0)} instructions). FP64 runs at 1/2 to 1/64× of FP32 throughput on most GPUs.",
            "Use FP32 unless double precision is required. Check if accumulations can stay in FP32.",
        ))

    if a.instruction_count > 0:
        global_ratio = (a.global_load_count + a.global_store_count) / a.instruction_count
        if global_ratio > 0.40:
            findings.append(_finding(
                "medium", "high_global_memory_ratio",
                f"{round(global_ratio * 100)}% of instructions are global memory ops — kernel is memory-bound.",
                "Add shared memory tiling to stage data and improve reuse. Check L2 cache hit rate in Nsight Compute.",
            ))

        branch_ratio = a.conditional_branch_count / a.instruction_count
        if branch_ratio > 0.15:
            findings.append(_finding(
                "medium", "high_branch_count",
                f"High conditional branch density ({a.conditional_branch_count} conditional branches, {round(branch_ratio * 100)}% of instructions) — warp divergence risk.",
                "Restructure to minimize thread-divergent code paths. Consider sorting inputs to reduce divergence.",
            ))

    if a.has_special_function_ops:
        count = a.instruction_mix.get("special_func", 0)
        findings.append(_finding(
            "low", "special_function_ops",
            f"{count} SFU instruction(s) (sin/cos/rsqrt/ex2/lg2) found. These have multi-cycle latency on the Special Function Unit.",
            "Batch or cache SFU results across threads where possible. Ensure SFU throughput is not the bottleneck.",
        ))

    if a.shared_load_count == 0 and a.global_load_count > 20:
        findings.append(_finding(
            "low", "no_shared_memory_usage",
            "Kernel reads from global memory without using shared memory.",
            "If the same data is accessed by multiple threads in a block, shared memory tiling may improve performance.",
        ))

    if a.has_tensor_ops:
        findings.append(_finding(
            "low", "tensor_ops_detected",
            f"WMMA tensor core ops detected ({a.instruction_mix.get('tensor_ops', 0)} instructions).",
            "Ensure inputs are in FP16/BF16 and matrix dimensions are aligned to multiples of 16 for optimal tensor core efficiency.",
        ))

    if a.has_atomics:
        findings.append(_finding(
            "low", "atomics_detected",
            f"Atomic operations detected ({a.instruction_mix.get('atomic_ops', 0)} instructions).",
            "Atomics serialize under high contention. Consider warp-level reduction (shuffles) followed by one atomic per warp.",
        ))

    return findings


def _find_matching_brace(text: str, open_index: int) -> int | None:
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _finding(severity: str, code: str, message: str, suggestion: str = "") -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message, "suggestion": suggestion}


def _severity_rank(severity: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(severity, 3)


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1) if m else None
