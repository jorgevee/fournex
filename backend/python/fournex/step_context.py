from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from .cuda_timers import time_phase
from .profiler import profiler_step_end, profiler_step_start
from .sdk import build_runtime_event, emit_event
from .shapes import describe_batch


@contextmanager
def step_context(
    step: int,
    batch: Any = None,
    *,
    step_kind: str = "train",
    mode: str | None = None,
    model: Any = None,
    model_name: str | None = None,
    precision_mode: str | None = None,
    is_training: bool | None = None,
) -> Iterator[dict[str, Any]]:
    start_ns = time.perf_counter_ns()
    step_span_id = f"step-{step}-{uuid.uuid4().hex[:8]}"
    resolved_model_name = _resolve_model_name(model_name, model)
    resolved_is_training = _resolve_is_training(is_training, model, step_kind, mode)
    profiler_step_start(step)

    emit_event(
        build_runtime_event(
            event_type="step_start",
            step_id=step,
            span_id=step_span_id,
            payload={
                "step_kind": step_kind,
                **({"mode": mode} if mode else {}),
            },
        )
    )

    if batch is not None:
        batch_description = describe_batch(batch)
        emit_event(
            build_runtime_event(
                event_type="shape_snapshot",
                step_id=step,
                parent_span_id=step_span_id,
                payload={
                    **batch_description,
                    **({"model_name": resolved_model_name} if resolved_model_name else {}),
                    **({"precision_mode": precision_mode} if precision_mode else {}),
                    **(
                        {"is_training": resolved_is_training}
                        if resolved_is_training is not None
                        else {}
                    ),
                },
            )
        )

    context = {
        "step": step,
        "batch": batch,
        "step_kind": step_kind,
        "mode": mode,
        "span_id": step_span_id,
        "model_name": resolved_model_name,
        "precision_mode": precision_mode,
        "is_training": resolved_is_training,
    }

    try:
        yield context
    except Exception:
        profiler_step_end(step)
        emit_event(
            build_runtime_event(
                event_type="step_end",
                step_id=step,
                span_id=step_span_id,
                duration_ns=time.perf_counter_ns() - start_ns,
                level="error",
                payload={
                    "step_kind": step_kind,
                    "status": "error",
                    **({"mode": mode} if mode else {}),
                },
            )
        )
        raise
    else:
        profiler_step_end(step)
        emit_event(
            build_runtime_event(
                event_type="step_end",
                step_id=step,
                span_id=step_span_id,
                duration_ns=time.perf_counter_ns() - start_ns,
                payload={
                    "step_kind": step_kind,
                    "status": "ok",
                    **({"mode": mode} if mode else {}),
                },
            )
        )


@contextmanager
def phase(
    name: str,
    *,
    step: int | None = None,
    parent_span_id: str | None = None,
    device: str | None = None,
) -> Iterator[dict[str, Any]]:
    with time_phase(name, step=step, parent_span_id=parent_span_id, device=device) as ctx:
        yield ctx


def _resolve_model_name(model_name: str | None, model: Any) -> str | None:
    if model_name:
        return model_name
    if model is None:
        return None
    return type(model).__name__


def _resolve_is_training(
    is_training: bool | None,
    model: Any,
    step_kind: str,
    mode: str | None,
) -> bool | None:
    if is_training is not None:
        return is_training
    if model is not None and hasattr(model, "training"):
        return bool(getattr(model, "training"))
    if mode == "train" or step_kind == "train":
        return True
    if mode in {"eval", "inference"} or step_kind in {"eval", "inference"}:
        return False
    return None
