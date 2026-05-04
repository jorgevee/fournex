from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from .recommendations import generate_recommendations

CLASSIFIER_VERSION = "0.2.0"

DEFAULT_STEADY_STATE_SKIP_FIRST_N = 2

# Bottlenecks that are symptoms (not root causes). When one of these ranks first,
# look for an underlying stall-type bottleneck to surface instead.
_SYMPTOM_BOTTLENECKS = {"underutilized_gpu"}
_STALL_BOTTLENECK_PRIORITY = ["input_bound", "copy_bound", "sync_bound", "launch_bound"]
DEFAULT_STEADY_STATE_LAST_K: int | None = None


def summarize_run(events: list[dict[str, Any]]) -> dict[str, Any]:
    per_step = derive_step_metrics(events)
    return summarize_step_scope(events, per_step=per_step)


def summarize_step_scope(
    events: list[dict[str, Any]],
    *,
    step_ids: list[int] | None = None,
    per_step: list[dict[str, Any]] | None = None,
    scope_name: str | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_per_step = list(per_step) if per_step is not None else derive_step_metrics(events)
    selected_step_ids = sorted({int(step_id) for step_id in step_ids}) if step_ids is not None else None
    scoped_steps = _select_steps(resolved_per_step, selected_step_ids)
    run_summary = derive_run_summary(events, scoped_steps)
    bottlenecks = classify_bottlenecks(events, scoped_steps, run_summary)
    diagnosis = build_diagnosis_result(bottlenecks, run_summary, scoped_steps, environment)

    summary = {
        "event_count": len(events),
        "step_count": len(scoped_steps),
        "per_step": scoped_steps,
        "run_summary": run_summary,
        "bottlenecks": bottlenecks,
        "diagnosis": diagnosis,
        "scope": {
            "name": scope_name or ("selected_steps" if selected_step_ids is not None else "run"),
            "step_ids": selected_step_ids if selected_step_ids is not None else [step["step_id"] for step in scoped_steps],
        },
    }
    return summary


def summarize_steady_state(
    events: list[dict[str, Any]],
    *,
    skip_first_n: int = 0,
    last_k: int | None = None,
    per_step: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_per_step = list(per_step) if per_step is not None else derive_step_metrics(events)
    steady_state_step_ids = select_steady_state_step_ids(
        resolved_per_step,
        skip_first_n=skip_first_n,
        last_k=last_k,
    )
    return summarize_step_scope(
        events,
        step_ids=steady_state_step_ids,
        per_step=resolved_per_step,
        scope_name="steady_state",
    )


def summarize_run_with_steady_state(
    events: list[dict[str, Any]],
    *,
    skip_first_n: int | None = None,
    last_k: int | None = None,
) -> dict[str, Any]:
    resolved_skip_first_n, resolved_last_k, selector_policy = _resolve_steady_state_selector(
        skip_first_n=skip_first_n,
        last_k=last_k,
    )
    per_step = derive_step_metrics(events)
    run_summary = summarize_step_scope(events, per_step=per_step, scope_name="run")
    steady_state_summary = summarize_steady_state(
        events,
        skip_first_n=resolved_skip_first_n,
        last_k=resolved_last_k,
        per_step=per_step,
    )
    return {
        "event_count": len(events),
        "step_count": len(per_step),
        "selector": {
            "policy": selector_policy,
            "skip_first_n": resolved_skip_first_n,
            "last_k": resolved_last_k,
        },
        "run": run_summary,
        "steady_state": steady_state_summary,
        "scope_comparison": _scope_comparison(run_summary, steady_state_summary),
    }


def derive_step_metrics(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: dict[int, dict[str, Any]] = {}

    for event in events:
        step_id = event.get("step_id")
        if step_id is None:
            continue
        step = steps.setdefault(step_id, _empty_step_metrics(step_id))
        _accumulate_step_event(step, event)

    return [steps[step_id] for step_id in sorted(steps)]


def derive_run_summary(events: list[dict[str, Any]], per_step: list[dict[str, Any]]) -> dict[str, Any]:
    gpu_samples = [event for event in events if event.get("event_type") == "gpu_sample"]
    completed_steps = [step for step in per_step if step["status"] == "ok"]
    step_wall_times = [step["step_wall_time_ns"] for step in completed_steps if step["step_wall_time_ns"] > 0]

    avg_gpu_util = _average_numeric_payload(gpu_samples, "utilization_gpu_pct")
    avg_mem_util = _average_numeric_payload(gpu_samples, "utilization_mem_pct")
    peak_memory_ratio = _peak_memory_ratio(gpu_samples)

    total_step_time_ns = sum(step_wall_times)
    throughput_steps_per_sec = 0.0
    if total_step_time_ns > 0:
        throughput_steps_per_sec = len(completed_steps) / (total_step_time_ns / 1_000_000_000)

    return {
        "average_gpu_utilization_pct": avg_gpu_util,
        "average_memory_utilization_pct": avg_mem_util,
        "throughput_steps_per_sec": throughput_steps_per_sec,
        "memory_pressure_peak_ratio": peak_memory_ratio,
        "utilization_instability_pct": _utilization_instability(gpu_samples),
        "step_time_avg_ns": mean(step_wall_times) if step_wall_times else 0,
        "step_time_max_ns": max(step_wall_times) if step_wall_times else 0,
        "shape_volatility_ratio": _shape_volatility_ratio(per_step),
        "profiler_windows_exported": sum(step["profiler_windows_exported"] for step in per_step),
        "kernel_count_per_step": _average_positive(step["kernel_count"] for step in completed_steps),
        "median_cuda_kernel_duration_us": _median_positive(
            step["median_cuda_kernel_duration_us"] for step in completed_steps
        ),
        "small_kernel_fraction": _average_positive(step["small_kernel_fraction"] for step in completed_steps),
        "dominant_stall_type": _dominant_stall_type(per_step),
    }


def classify_bottlenecks(
    events: list[dict[str, Any]],
    per_step: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    completed_steps = [step for step in per_step if step["status"] == "ok" and step["step_wall_time_ns"] > 0]
    if not completed_steps:
        return []

    classifications: list[dict[str, Any]] = []

    input_ratio = mean(
        _bounded_ratio(step["dataloader_wait_time_ns"], step["step_wall_time_ns"])
        for step in completed_steps
    )
    if input_ratio >= 0.2:
        classifications.append(
            _classification(
                "input_bound",
                input_ratio,
                {
                    "avg_dataloader_fraction": round(input_ratio, 4),
                    "dominant_stall_type": run_summary["dominant_stall_type"],
                },
                worst_steps=_top_steps(
                    completed_steps,
                    lambda s: _bounded_ratio(s["dataloader_wait_time_ns"], s["step_wall_time_ns"]),
                ),
            )
        )

    copy_ratio = mean(
        _bounded_ratio(step["h2d_copy_time_ns"], step["step_wall_time_ns"])
        for step in completed_steps
    )
    if copy_ratio >= 0.15:
        classifications.append(
            _classification(
                "copy_bound",
                copy_ratio,
                {
                    "avg_h2d_fraction": round(copy_ratio, 4),
                    "steps_with_h2d": sum(1 for step in completed_steps if step["h2d_copy_time_ns"] > 0),
                },
                worst_steps=_top_steps(
                    completed_steps,
                    lambda s: _bounded_ratio(s["h2d_copy_time_ns"], s["step_wall_time_ns"]),
                ),
            )
        )

    sync_ratio = mean(
        _bounded_ratio(step["sync_wait_time_ns"], step["step_wall_time_ns"])
        for step in completed_steps
    )
    if sync_ratio >= 0.1:
        classifications.append(
            _classification(
                "sync_bound",
                sync_ratio,
                {
                    "avg_sync_fraction": round(sync_ratio, 4),
                    "steps_with_sync_wait": sum(1 for step in completed_steps if step["sync_wait_time_ns"] > 0),
                },
                worst_steps=_top_steps(
                    completed_steps,
                    lambda s: _bounded_ratio(s["sync_wait_time_ns"], s["step_wall_time_ns"]),
                ),
            )
        )

    if run_summary["average_gpu_utilization_pct"] > 0 and run_summary["average_gpu_utilization_pct"] < 35:
        classifications.append(
            _classification(
                "underutilized_gpu",
                1.0 - (run_summary["average_gpu_utilization_pct"] / 100.0),
                {
                    "average_gpu_utilization_pct": round(run_summary["average_gpu_utilization_pct"], 2),
                    "utilization_instability_pct": round(run_summary["utilization_instability_pct"], 2),
                },
                worst_steps=_top_steps(
                    completed_steps,
                    lambda s: 1.0 - s["gpu_active_fraction_proxy"],
                ),
            )
        )

    if run_summary["memory_pressure_peak_ratio"] >= 0.9:
        classifications.append(
            _classification(
                "memory_pressure",
                run_summary["memory_pressure_peak_ratio"],
                {
                    "memory_pressure_peak_ratio": round(run_summary["memory_pressure_peak_ratio"], 4),
                    "average_memory_utilization_pct": round(run_summary["average_memory_utilization_pct"], 2),
                },
            )
        )

    if run_summary["shape_volatility_ratio"] >= 0.3:
        classifications.append(
            _classification(
                "shape_instability",
                run_summary["shape_volatility_ratio"],
                {
                    "shape_volatility_ratio": round(run_summary["shape_volatility_ratio"], 4),
                    "changed_steps": [step["step_id"] for step in per_step if step["shape_changed"]],
                },
                worst_steps=[
                    {"step_id": step["step_id"], "value": 1.0}
                    for step in per_step
                    if step["shape_changed"]
                ],
            )
        )

    if (
        run_summary["profiler_windows_exported"] > 0
        and run_summary["average_gpu_utilization_pct"] < 50
        and copy_ratio < 0.1
        and input_ratio < 0.1
        and sync_ratio < 0.1
    ):
        avg_gpu = run_summary["average_gpu_utilization_pct"]
        launch_bound_score = round(min(0.5, max(0.3, (50.0 - avg_gpu) / 100.0 + 0.3)), 4)
        classifications.append(
            _classification(
                "launch_bound",
                launch_bound_score,
                {
                    "profiler_windows_exported": run_summary["profiler_windows_exported"],
                    "average_gpu_utilization_pct": round(avg_gpu, 2),
                    "kernel_count_per_step": round(run_summary.get("kernel_count_per_step", 0.0), 2),
                    "median_cuda_kernel_duration_us": round(
                        run_summary.get("median_cuda_kernel_duration_us", 0.0), 3
                    ),
                    "small_kernel_fraction": round(run_summary.get("small_kernel_fraction", 0.0), 4),
                    "shapes_stable": run_summary.get("shape_volatility_ratio", 0.0) < 0.3,
                    "note": "Profiler windows were captured but dominant stalls were not input, copy, or sync heavy.",
                },
                worst_steps=_top_steps(
                    completed_steps,
                    lambda s: 1.0 - s["gpu_active_fraction_proxy"],
                ),
            )
        )

    if run_summary["average_gpu_utilization_pct"] <= 0.0 and run_summary["dominant_stall_type"] == "unknown":
        classifications.append(
            _classification(
                "insufficient_telemetry",
                1.0,
                {
                    "step_count": len(completed_steps),
                    "note": "No timing breakdowns or GPU utilization data were recorded.",
                },
            )
        )

    classifications.sort(key=lambda item: item["score"], reverse=True)
    return classifications


def _classification(
    label: str,
    score: float,
    evidence: dict[str, Any],
    worst_steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "score": round(score, 4),
        "evidence": evidence,
        "worst_steps": worst_steps if worst_steps is not None else [],
    }


def build_diagnosis_result(
    bottlenecks: list[dict[str, Any]],
    run_summary: dict[str, Any],
    per_step: list[dict[str, Any]] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not bottlenecks:
        return {
            "primary_bottleneck": None,
            "user_facing_bottleneck": None,
            "secondary_bottlenecks": [],
            "confidence": {
                "level": "low",
                "score": 0.0,
                "reason": "No bottleneck cleared the current ruleset thresholds.",
            },
            "evidence": {},
            "why": [],
            "why_not_others": [],
            "recommendations": [],
            "recommendation_bundles": [],
            "dominant_stall_type": run_summary.get("dominant_stall_type", "unknown"),
            "classifier_version": CLASSIFIER_VERSION,
        }

    primary = bottlenecks[0]
    secondary = [item["label"] for item in bottlenecks[1:3]]
    confidence_score = _confidence_score(primary, bottlenecks, run_summary)

    if confidence_score >= 0.75:
        confidence_level = "high"
    elif confidence_score >= 0.4:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    rec_output = generate_recommendations(bottlenecks, run_summary, per_step, environment)

    # If primary is a symptom, find the root-cause stall bottleneck to surface to users.
    primary_label = primary["label"]
    bottleneck_labels = {b["label"] for b in bottlenecks}
    if primary_label in _SYMPTOM_BOTTLENECKS:
        user_facing = next(
            (label for label in _STALL_BOTTLENECK_PRIORITY if label in bottleneck_labels),
            primary_label,
        )
    else:
        user_facing = primary_label

    return {
        "primary_bottleneck": primary_label,
        "user_facing_bottleneck": user_facing,
        "secondary_bottlenecks": secondary,
        "confidence": {
            "level": confidence_level,
            "score": confidence_score,
            "reason": _confidence_reason(primary, bottlenecks, run_summary),
        },
        "evidence": primary["evidence"],
        "why": _explanation_bullets(primary, run_summary),
        "why_not_others": _contradiction_bullets(primary, bottlenecks[1:3], run_summary),
        "recommendations": rec_output["recommendations"],
        "recommendation_bundles": rec_output["bundles"],
        "dominant_stall_type": run_summary.get("dominant_stall_type", "unknown"),
        "classifier_version": CLASSIFIER_VERSION,
    }


def _select_steps(per_step: list[dict[str, Any]], step_ids: list[int] | None) -> list[dict[str, Any]]:
    if step_ids is None:
        return list(per_step)
    selected = set(step_ids)
    return [step for step in per_step if step["step_id"] in selected]


def select_steady_state_step_ids(
    per_step: list[dict[str, Any]],
    *,
    skip_first_n: int = 0,
    last_k: int | None = None,
) -> list[int]:
    completed_step_ids = [
        step["step_id"]
        for step in per_step
        if step["status"] == "ok" and step["step_wall_time_ns"] > 0
    ]

    skip_count = max(0, skip_first_n)
    selected = completed_step_ids[skip_count:]

    if last_k is not None:
        keep_count = max(0, last_k)
        if keep_count == 0:
            selected = []
        else:
            selected = selected[-keep_count:]

    return selected


def _resolve_steady_state_selector(
    *,
    skip_first_n: int | None,
    last_k: int | None,
) -> tuple[int, int | None, str]:
    if skip_first_n is None and last_k is None:
        return DEFAULT_STEADY_STATE_SKIP_FIRST_N, DEFAULT_STEADY_STATE_LAST_K, "default"

    resolved_skip_first_n = max(0, skip_first_n or 0)
    resolved_last_k = None if last_k is None else max(0, last_k)
    return resolved_skip_first_n, resolved_last_k, "explicit"


def _scope_comparison(run_summary: dict[str, Any], steady_state_summary: dict[str, Any]) -> dict[str, Any]:
    run_primary = run_summary["diagnosis"]["primary_bottleneck"]
    steady_state_primary = steady_state_summary["diagnosis"]["primary_bottleneck"]
    return {
        "diagnosis_changed": run_primary != steady_state_primary,
        "run_primary_bottleneck": run_primary,
        "steady_state_primary_bottleneck": steady_state_primary,
    }


def _empty_step_metrics(step_id: int) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "status": "unknown",
        "step_kind": None,
        "step_wall_time_ns": 0,
        "dataloader_wait_time_ns": 0,
        "h2d_copy_time_ns": 0,
        "forward_time_ns": 0,
        "backward_time_ns": 0,
        "optimizer_time_ns": 0,
        "sync_wait_time_ns": 0,
        "gpu_active_fraction_proxy": 0.0,
        "shape_signature": None,
        "shape_changed": False,
        "batch_size": 0,
        "sequence_length": None,
        "profiler_windows_exported": 0,
        "kernel_count": 0,
        "median_cuda_kernel_duration_us": 0.0,
        "small_kernel_fraction": 0.0,
        "loss": None,
    }


def _confidence_reason(
    primary: dict[str, Any],
    bottlenecks: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> str:
    second_score = bottlenecks[1]["score"] if len(bottlenecks) > 1 else 0.0
    score_gap = round(primary["score"] - second_score, 4)
    stall_type = run_summary.get("dominant_stall_type", "unknown")
    if len(bottlenecks) == 1:
        return f"{primary['label']} is the only bottleneck above threshold."
    if stall_type == primary["label"]:
        return f"{primary['label']} leads the ranking and matches the dominant stall summary with a score gap of {score_gap}."
    return f"{primary['label']} leads the ranking with a score gap of {score_gap}, but other signals remain present."


def _confidence_score(
    primary: dict[str, Any],
    bottlenecks: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> float:
    second_score = bottlenecks[1]["score"] if len(bottlenecks) > 1 else 0.0
    score_gap = primary["score"] - second_score
    stall_type = run_summary.get("dominant_stall_type", "unknown")

    score = float(primary["score"])

    if len(bottlenecks) == 1:
        score += 0.15

    if score_gap >= 0.25:
        score += 0.15
    elif score_gap >= 0.1:
        score += 0.1
    elif score_gap > 0:
        score += 0.05

    if stall_type == primary["label"]:
        score += 0.1
    elif stall_type not in {"unknown", "compute_bound"}:
        score -= 0.05

    return round(min(1.0, max(0.0, score)), 4)


def _explanation_bullets(primary: dict[str, Any], run_summary: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    label = primary["label"]
    evidence = primary["evidence"]

    if label == "input_bound":
        bullets.append(f"Average DataLoader wait fraction is {evidence.get('avg_dataloader_fraction', 0):.3f}.")
    elif label == "copy_bound":
        bullets.append(f"Average H2D copy fraction is {evidence.get('avg_h2d_fraction', 0):.3f}.")
    elif label == "sync_bound":
        bullets.append(f"Average sync wait fraction is {evidence.get('avg_sync_fraction', 0):.3f}.")
    elif label == "underutilized_gpu":
        bullets.append(
            f"Average GPU utilization is {evidence.get('average_gpu_utilization_pct', run_summary.get('average_gpu_utilization_pct', 0)):.1f}%."
        )
    elif label == "memory_pressure":
        bullets.append(
            f"Peak memory pressure ratio reached {evidence.get('memory_pressure_peak_ratio', run_summary.get('memory_pressure_peak_ratio', 0)):.3f}."
        )
    elif label == "shape_instability":
        bullets.append(f"Shape volatility ratio is {evidence.get('shape_volatility_ratio', 0):.3f}.")
    elif label == "launch_bound":
        kernel_count = evidence.get("kernel_count_per_step", 0)
        median_kernel = evidence.get("median_cuda_kernel_duration_us", 0)
        if kernel_count:
            bullets.append(
                f"Profiler saw about {kernel_count:.1f} CUDA kernels per step with median duration {median_kernel:.3f} us."
            )
        bullets.append("GPU utilization sampling stayed low, which is expected for bursty tiny-kernel workloads.")
        if evidence.get("shapes_stable"):
            bullets.append("Shapes were stable, so compile or CUDA graph mitigations are viable.")
    elif label == "insufficient_telemetry":
        bullets.append(f"No timing breakdowns were observed across {evidence.get('step_count', 0)} completed steps.")
        bullets.append("GPU utilization data is absent — no utilization samples were recorded.")

    stall_type = run_summary.get("dominant_stall_type", "unknown")
    if stall_type != "unknown":
        bullets.append(f"Run summary dominant stall type is {stall_type}.")

    return bullets[:3]


def _contradiction_bullets(
    primary: dict[str, Any],
    secondary_candidates: list[dict[str, Any]],
    run_summary: dict[str, Any],
) -> list[str]:
    bullets: list[str] = []
    for item in secondary_candidates:
        bullets.append(f"{item['label']} also triggered with score {item['score']:.3f}.")
    if run_summary.get("dominant_stall_type") not in {"unknown", primary["label"]}:
        bullets.append(
            f"Run summary stall type is {run_summary.get('dominant_stall_type')}, which does not exactly match the top-ranked diagnosis."
        )
    return bullets[:3]


def _recommendations_for_label(label: str) -> list[str]:
    mapping = {
        "input_bound": [
            "Increase DataLoader workers.",
            "Enable pinned memory.",
            "Prefetch batches or move CPU transforms off the critical path.",
        ],
        "copy_bound": [
            "Reduce host-to-device transfer volume.",
            "Enable pinned memory for input batches.",
            "Improve overlap between copies and compute.",
        ],
        "sync_bound": [
            "Reduce explicit synchronization points.",
            "Check for host waits on CUDA events or device syncs.",
            "Increase overlap between CPU and GPU work.",
        ],
        "underutilized_gpu": [
            "Increase batch size if memory allows.",
            "Look for small kernels or fragmented execution.",
            "Use profiler windows to confirm launch overhead or input stalls.",
        ],
        "memory_pressure": [
            "Reduce batch size or sequence length.",
            "Check for allocator churn or fragmentation.",
            "Lower activation or optimizer memory footprint.",
        ],
        "shape_instability": [
            "Reduce shape variability across steps.",
            "Bucket or pad inputs to stabilize execution shape.",
            "Review dynamic-shape overhead in the input pipeline.",
        ],
        "launch_bound": [
            "Look for many short kernels in profiler output.",
            "Fuse small operations where possible.",
            "Reduce Python dispatch overhead or consider graph capture.",
        ],
        "insufficient_telemetry": [
            "Verify that telemetry collection is properly configured.",
            "Check that event hooks are attached before training begins.",
            "Ensure the profiler or tracing layer is active during the run.",
        ],
    }
    return mapping.get(label, [])


def _accumulate_step_event(step: dict[str, Any], event: dict[str, Any]) -> None:
    event_type = event.get("event_type")
    payload = event.get("payload", {})
    duration_ns = event.get("duration_ns") or 0

    if event_type == "step_start":
        step["step_kind"] = payload.get("step_kind")
        return

    if event_type == "step_end":
        step["status"] = payload.get("status", "unknown")
        step["step_kind"] = payload.get("step_kind", step["step_kind"])
        step["step_wall_time_ns"] = duration_ns
        if "loss" in payload:
            step["loss"] = payload.get("loss")
        _update_gpu_active_fraction(step)
        return

    if event_type == "dataloader_span" and payload.get("stage") == "next":
        step["dataloader_wait_time_ns"] += duration_ns
        return

    if event_type == "memcpy_span" and payload.get("copy_kind") == "h2d":
        step["h2d_copy_time_ns"] += duration_ns
        return

    if event_type == "phase_span":
        phase_name = payload.get("phase_name")
        if phase_name == "forward":
            step["forward_time_ns"] = max(step["forward_time_ns"], duration_ns)
        elif phase_name == "backward":
            step["backward_time_ns"] = max(step["backward_time_ns"], duration_ns)
        elif phase_name == "optimizer":
            step["optimizer_time_ns"] = max(step["optimizer_time_ns"], duration_ns)
        return

    if event_type == "sync_wait":
        step["sync_wait_time_ns"] += duration_ns
        return

    if event_type == "shape_snapshot":
        step["batch_size"] = payload.get("batch_size", 0)
        step["sequence_length"] = payload.get("sequence_length")
        step["shape_signature"] = _shape_signature(payload.get("shapes", {}))
        return

    if event_type == "profiler_window" and payload.get("window_state") == "exported":
        step["profiler_windows_exported"] += 1
        step["kernel_count"] += int(_to_float(payload.get("kernel_count")) or 0)
        median_us = _to_float(payload.get("median_cuda_kernel_duration_us"))
        if median_us is not None and median_us > 0:
            step["median_cuda_kernel_duration_us"] = median_us
        small_fraction = _to_float(payload.get("small_kernel_fraction"))
        if small_fraction is not None:
            step["small_kernel_fraction"] = max(0.0, min(1.0, small_fraction))


def _update_gpu_active_fraction(step: dict[str, Any]) -> None:
    step_wall_time_ns = step["step_wall_time_ns"]
    if step_wall_time_ns <= 0:
        step["gpu_active_fraction_proxy"] = 0.0
        return

    active_ns = step["forward_time_ns"] + step["backward_time_ns"] + step["optimizer_time_ns"]
    step["gpu_active_fraction_proxy"] = round(_bounded_ratio(active_ns, step_wall_time_ns), 4)


def _shape_signature(shapes: dict[str, Any]) -> str:
    items = []
    for key in sorted(shapes):
        items.append(f"{key}:{shapes[key]}")
    return "|".join(items)


def _average_numeric_payload(events: list[dict[str, Any]], field: str) -> float:
    values: list[float] = []
    for event in events:
        raw = event.get("payload", {}).get(field)
        number = _to_float(raw)
        if number is not None:
            values.append(number)
    return mean(values) if values else 0.0


def _average_positive(values: Any) -> float:
    positives = [float(value) for value in values if value and float(value) > 0]
    return mean(positives) if positives else 0.0


def _median_positive(values: Any) -> float:
    positives = sorted(float(value) for value in values if value and float(value) > 0)
    if not positives:
        return 0.0
    midpoint = len(positives) // 2
    if len(positives) % 2:
        return positives[midpoint]
    return (positives[midpoint - 1] + positives[midpoint]) / 2.0


def _peak_memory_ratio(gpu_samples: list[dict[str, Any]]) -> float:
    ratios: list[float] = []
    for event in gpu_samples:
        payload = event.get("payload", {})
        used = _to_float(payload.get("memory_used_bytes"))
        total = _to_float(payload.get("memory_total_bytes"))
        if used is not None and total not in (None, 0):
            ratios.append(used / total)
    return max(ratios) if ratios else 0.0


def _utilization_instability(gpu_samples: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for event in gpu_samples:
        number = _to_float(event.get("payload", {}).get("utilization_gpu_pct"))
        if number is not None:
            values.append(number)
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def _shape_volatility_ratio(per_step: list[dict[str, Any]]) -> float:
    signatures = [step["shape_signature"] for step in per_step if step["shape_signature"]]
    if len(signatures) < 2:
        for step in per_step:
            step["shape_changed"] = False
        return 0.0

    changes = 0
    previous = signatures[0]
    seen = iter(signatures)
    next(seen)
    for signature in seen:
        changed = signature != previous
        changes += int(changed)
        previous = signature

    previous = None
    for step in per_step:
        signature = step["shape_signature"]
        if not signature:
            step["shape_changed"] = False
            continue
        step["shape_changed"] = previous is not None and signature != previous
        previous = signature

    return changes / max(len(signatures) - 1, 1)


def _dominant_stall_type(per_step: list[dict[str, Any]]) -> str:
    stall_totals = defaultdict(int)
    compute_total = 0

    for step in per_step:
        stall_totals["input_bound"] += step["dataloader_wait_time_ns"]
        stall_totals["copy_bound"] += step["h2d_copy_time_ns"]
        stall_totals["sync_bound"] += step["sync_wait_time_ns"]
        compute_total += step["forward_time_ns"] + step["backward_time_ns"] + step["optimizer_time_ns"]

    if not stall_totals and compute_total <= 0:
        return "unknown"

    dominant_stall = max(stall_totals.items(), key=lambda item: item[1]) if stall_totals else ("unknown", 0)
    if dominant_stall[1] > 0:
        return dominant_stall[0]

    return "compute_bound" if compute_total > 0 else "unknown"


def _top_steps(
    steps: list[dict[str, Any]],
    key_fn: Any,
    n: int = 3,
) -> list[dict[str, Any]]:
    ranked = sorted(steps, key=key_fn, reverse=True)
    result = []
    for step in ranked[:n]:
        value = key_fn(step)
        if value > 0:
            result.append({"step_id": step["step_id"], "value": round(value, 4)})
    return result


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _bounded_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return max(0.0, min(float(numerator) / float(denominator), 1.0))
