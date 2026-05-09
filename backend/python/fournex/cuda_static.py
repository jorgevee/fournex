from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .kernel_inspector import device_limits_for_gpu, estimate_occupancy


@dataclass(slots=True)
class CudaKernelSource:
    name: str
    filename: str
    signature: str
    params: str
    body: str
    line: int
    indexing_patterns: list[str] = field(default_factory=list)
    memory_access_styles: list[str] = field(default_factory=list)
    atomics: list[str] = field(default_factory=list)
    reductions: list[str] = field(default_factory=list)
    shared_memory: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("body", None)
        return payload


@dataclass(slots=True)
class CudaLaunchConfig:
    kernel_name: str
    filename: str
    line: int
    raw_config: str
    grid_expr: str | None = None
    block_expr: str | None = None
    shared_memory_expr: str | None = None
    stream_expr: str | None = None
    block_size_hint: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_cuda_source(source: str, *, filename: str = "<memory>", gpu_model: str | None = None) -> dict[str, Any]:
    kernels = parse_cuda_kernels(source, filename=filename)
    launches = parse_cuda_launches(source, filename=filename)
    return build_static_cuda_report(kernels, launches, gpu_model=gpu_model)


def inspect_cuda_files(paths: list[str | Path], *, gpu_model: str | None = None) -> dict[str, Any]:
    kernels: list[CudaKernelSource] = []
    launches: list[CudaLaunchConfig] = []
    for path_like in paths:
        path = Path(path_like)
        source = path.read_text(encoding="utf-8", errors="replace")
        kernels.extend(parse_cuda_kernels(source, filename=str(path)))
        launches.extend(parse_cuda_launches(source, filename=str(path)))
    return build_static_cuda_report(kernels, launches, gpu_model=gpu_model)


def build_static_cuda_report(
    kernels: list[CudaKernelSource],
    launches: list[CudaLaunchConfig],
    *,
    gpu_model: str | None = None,
) -> dict[str, Any]:
    launch_by_kernel: dict[str, list[CudaLaunchConfig]] = {}
    for launch in launches:
        launch_by_kernel.setdefault(launch.kernel_name, []).append(launch)

    kernel_payloads: list[dict[str, Any]] = []
    all_findings: list[dict[str, Any]] = []
    for kernel in kernels:
        _annotate_kernel(kernel)
        for finding in kernel.findings:
            all_findings.append({**finding, "kernel_name": kernel.name, "filename": kernel.filename, "line": kernel.line})
        kernel_payloads.append(kernel.to_dict())

    launch_payloads = [launch.to_dict() for launch in launches]
    advisor = _launch_advice(kernels, launch_by_kernel, gpu_model=gpu_model)
    return {
        "schema_version": "cuda_static_v1",
        "gpu_model": gpu_model,
        "kernel_count": len(kernels),
        "launch_count": len(launches),
        "kernels": kernel_payloads,
        "launches": launch_payloads,
        "findings": sorted(all_findings, key=lambda item: (_severity_rank(item["severity"]), item["line"])),
        "launch_advisor": advisor,
    }


def parse_cuda_kernels(source: str, *, filename: str = "<memory>") -> list[CudaKernelSource]:
    clean = _strip_comments(source)
    kernels: list[CudaKernelSource] = []
    pattern = re.compile(
        r"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?"
        r"(?P<prefix>(?:[\w:<>,~*&]+\s+)+?)"
        r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{}]*)\)\s*\{",
        re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(clean):
        open_brace = clean.find("{", match.end() - 1)
        close_brace = _find_matching_brace(clean, open_brace)
        if close_brace is None:
            continue
        body = clean[open_brace + 1:close_brace]
        signature = " ".join(clean[match.start():open_brace].split())
        kernel = CudaKernelSource(
            name=match.group("name"),
            filename=filename,
            signature=signature,
            params=" ".join(match.group("params").split()),
            body=body,
            line=_line_number(clean, match.start()),
        )
        _annotate_kernel(kernel)
        kernels.append(kernel)
    return kernels


def parse_cuda_launches(source: str, *, filename: str = "<memory>") -> list[CudaLaunchConfig]:
    clean = _strip_comments(source)
    launches: list[CudaLaunchConfig] = []
    pattern = re.compile(r"(?P<name>[A-Za-z_]\w*)\s*<<<(?P<config>.*?)>>>\s*\(", re.DOTALL)
    for match in pattern.finditer(clean):
        parts = _split_top_level(match.group("config"))
        block_expr = parts[1].strip() if len(parts) > 1 else None
        launches.append(
            CudaLaunchConfig(
                kernel_name=match.group("name"),
                filename=filename,
                line=_line_number(clean, match.start()),
                raw_config=" ".join(match.group("config").split()),
                grid_expr=parts[0].strip() if parts else None,
                block_expr=block_expr,
                shared_memory_expr=parts[2].strip() if len(parts) > 2 else None,
                stream_expr=parts[3].strip() if len(parts) > 3 else None,
                block_size_hint=_block_size_hint(block_expr or ""),
            )
        )
    return launches


def _annotate_kernel(kernel: CudaKernelSource) -> None:
    body = kernel.body
    lowered = body.lower()
    kernel.indexing_patterns = _indexing_patterns(body)
    kernel.memory_access_styles = _memory_access_styles(body)
    kernel.atomics = sorted(set(re.findall(r"\batomic[A-Za-z_0-9]*\s*\(", body)))
    kernel.reductions = _reduction_patterns(body)
    kernel.shared_memory = _shared_memory_allocations(body)
    kernel.findings = []

    if "__syncthreads()" in body and "__shared__" not in body and "cooperative_groups" not in lowered:
        kernel.findings.append(_finding("medium", "unnecessary_syncthreads", "Kernel uses __syncthreads() without visible shared memory or cooperative groups."))
    if "__syncthreads()" in body and re.search(r"if\s*\([^)]*(threadIdx|idx|tid)[^)]*\)\s*\{[^{}]*__syncthreads\s*\(", body, re.DOTALL):
        kernel.findings.append(_finding("high", "conditional_syncthreads", "__syncthreads() appears inside a thread-dependent branch."))
    if any(item["bytes"] and item["bytes"] >= 49152 for item in kernel.shared_memory):
        kernel.findings.append(_finding("medium", "large_static_shared_memory", "Static shared memory allocation may constrain occupancy."))
    if any(item["bank_conflict_risk"] for item in kernel.shared_memory):
        kernel.findings.append(_finding("medium", "possible_shared_memory_bank_conflict", "Shared memory tile dimensions are multiples of 32; consider padding one column."))
    if "__shared__" in body and "__syncthreads()" not in body:
        kernel.findings.append(_finding("medium", "shared_memory_without_barrier", "Shared memory is used without a visible __syncthreads() barrier."))
    if "threadIdx.x" in body and not re.search(r"if\s*\([^)]*(<|<=)\s*[^)]*\)", body) and re.search(r"\w+\s*\[[^\]]*(idx|i|tid)[^\]]*\]", body):
        kernel.findings.append(_finding("medium", "missing_obvious_bounds_guard", "Global indexing is used but no obvious upper-bound guard was found."))
    if "threadIdx.x" not in body and "threadIdx.y" not in body and "threadIdx.z" not in body:
        kernel.findings.append(_finding("low", "no_thread_indexing_detected", "No threadIdx usage detected; verify this is intended."))


def _indexing_patterns(body: str) -> list[str]:
    patterns: list[str] = []
    if re.search(r"blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x", body):
        patterns.append("1d_grid_stride_index")
    if re.search(r"blockIdx\.y\s*\*\s*blockDim\.y\s*\+\s*threadIdx\.y", body):
        patterns.append("2d_y_index")
    if re.search(r"blockIdx\.z\s*\*\s*blockDim\.z\s*\+\s*threadIdx\.z", body):
        patterns.append("3d_z_index")
    if re.search(r"for\s*\([^;]+;[^;]+;[^)]*\+=\s*blockDim\.x\s*\*\s*gridDim\.x", body):
        patterns.append("grid_stride_loop")
    if "threadIdx.x" in body and not patterns:
        patterns.append("thread_x_indexing")
    return patterns


def _memory_access_styles(body: str) -> list[str]:
    styles: list[str] = []
    if re.search(r"\w+\s*\[\s*(idx|i|tid)\s*\]", body):
        styles.append("likely_coalesced_1d")
    if re.search(r"\w+\s*\[[^\]]*\*\s*(stride|pitch|ld|width)[^\]]*\]", body):
        styles.append("strided_or_pitched")
    if "__shared__" in body:
        styles.append("shared_memory_tiling")
    if re.search(r"\b(float4|int4|uint4|half2|double2)\b", body):
        styles.append("vectorized")
    if "__ldg" in body or "__constant__" in body:
        styles.append("read_only_or_constant_cache")
    if not styles and re.search(r"\w+\s*\[[^\]]+\]", body):
        styles.append("indexed_global_or_local_memory")
    return styles


def _reduction_patterns(body: str) -> list[str]:
    patterns: list[str] = []
    if re.search(r"for\s*\([^)]*(>>=|/=\s*2|stride\s*>\s*0)", body):
        patterns.append("tree_reduction_loop")
    if "__shfl" in body:
        patterns.append("warp_shuffle_reduction")
    if re.search(r"\+=\s*\w+\s*\[[^\]]+\]", body):
        patterns.append("accumulation")
    return sorted(set(patterns))


def _shared_memory_allocations(body: str) -> list[dict[str, Any]]:
    allocations: list[dict[str, Any]] = []
    pattern = re.compile(r"__shared__\s+(?P<type>[\w:<>,]+)\s+(?P<name>\w+)\s*(?P<dims>(?:\[[^\]]+\])+)")
    for match in pattern.finditer(body):
        dims = [dim.strip() for dim in re.findall(r"\[([^\]]+)\]", match.group("dims"))]
        element_count = _static_product(dims)
        type_size = _cuda_type_size(match.group("type"))
        bytes_used = element_count * type_size if element_count is not None and type_size is not None else None
        allocations.append(
            {
                "name": match.group("name"),
                "type": match.group("type"),
                "dims": dims,
                "bytes": bytes_used,
                "bank_conflict_risk": _bank_conflict_risk(dims),
            }
        )
    if re.search(r"extern\s+__shared__\s+", body):
        allocations.append({"name": "extern_dynamic_shared", "type": "extern", "dims": [], "bytes": None, "bank_conflict_risk": False})
    return allocations


def _launch_advice(
    kernels: list[CudaKernelSource],
    launch_by_kernel: dict[str, list[CudaLaunchConfig]],
    *,
    gpu_model: str | None,
) -> list[dict[str, Any]]:
    limits = device_limits_for_gpu(gpu_model)
    advice: list[dict[str, Any]] = []
    for kernel in kernels:
        launches = launch_by_kernel.get(kernel.name, [])
        observed_blocks = [launch.block_size_hint for launch in launches if launch.block_size_hint]
        shared_bytes = max((item["bytes"] or 0 for item in kernel.shared_memory), default=0)
        candidates = []
        for block_size in (128, 256, 512):
            occupancy = estimate_occupancy(
                registers_per_thread=None,
                shared_memory_per_block_bytes=shared_bytes,
                threads_per_block=block_size,
                device_limits=limits,
            )
            candidates.append({"block_size": block_size, "occupancy_estimate": occupancy})
        notes = ["Safe recommended starting configurations; benchmark before treating as optimal."]
        if observed_blocks:
            notes.append(f"Observed launch block sizes: {sorted(set(observed_blocks))}.")
        if any(item["bank_conflict_risk"] for item in kernel.shared_memory):
            notes.append("Consider padding shared-memory tiles before tuning block size.")
        advice.append({"kernel_name": kernel.name, "candidate_block_sizes": candidates, "notes": notes})
    return advice


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//.*", "", source)


def _find_matching_brace(source: str, open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closing = set(pairs.values())
    for index, char in enumerate(text):
        if char in pairs:
            depth += 1
        elif char in closing and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _block_size_hint(expr: str) -> int | None:
    expr = expr.strip()
    if expr.isdigit():
        return int(expr)
    dim3_match = re.search(r"dim3\s*\(\s*(\d+)\s*,\s*(\d+)?\s*,?\s*(\d+)?", expr)
    if dim3_match:
        values = [int(value) if value else 1 for value in dim3_match.groups()]
        return values[0] * values[1] * values[2]
    return None


def _line_number(source: str, index: int) -> int:
    return source.count("\n", 0, index) + 1


def _finding(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def _severity_rank(severity: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(severity, 3)


def _static_product(dims: list[str]) -> int | None:
    product = 1
    for dim in dims:
        if not dim.isdigit():
            return None
        product *= int(dim)
    return product


def _cuda_type_size(type_name: str) -> int | None:
    base = type_name.replace("const", "").strip()
    sizes = {
        "char": 1,
        "unsigned char": 1,
        "short": 2,
        "half": 2,
        "__half": 2,
        "int": 4,
        "unsigned": 4,
        "float": 4,
        "double": 8,
        "long": 8,
        "long long": 8,
    }
    return sizes.get(base)


def _bank_conflict_risk(dims: list[str]) -> bool:
    if not dims:
        return False
    last = dims[-1]
    return last.isdigit() and int(last) % 32 == 0
