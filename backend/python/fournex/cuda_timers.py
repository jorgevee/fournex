from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from .sdk import begin_span, build_runtime_event, end_span

try:
    import torch
except ImportError:
    torch = None


class _TimerBackend:
    def __init__(self, device: str | None):
        self._device = device
        self._cpu_start_ns = time.perf_counter_ns()
        self._cpu_end_ns: int | None = None
        self._start_event = None
        self._end_event = None

        if torch is not None and _cuda_events_supported(device):
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()

    def stop(self) -> int:
        self._cpu_end_ns = time.perf_counter_ns()

        if self._start_event is not None and self._end_event is not None:
            self._end_event.record()
            self._end_event.synchronize()
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
            return int(elapsed_ms * 1_000_000)

        return self._cpu_end_ns - self._cpu_start_ns


@contextmanager
def time_phase(
    name: str,
    *,
    step: int | None = None,
    parent_span_id: str | None = None,
    device: str | None = None,
) -> Iterator[dict[str, Any]]:
    timer = _TimerBackend(device)
    span_id = f"phase-{name}-{uuid.uuid4().hex[:8]}"

    begin_span(
        build_runtime_event(
            event_type="phase_span",
            step_id=step,
            span_id=span_id,
            parent_span_id=parent_span_id,
            duration_ns=0,
            payload={
                "phase_name": name,
                **({"device": device} if device else {}),
            },
        )
    )

    try:
        yield {
            "name": name,
            "step": step,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "device": device,
        }
    finally:
        end_span(
            build_runtime_event(
                event_type="phase_span",
                step_id=step,
                span_id=span_id,
                parent_span_id=parent_span_id,
                duration_ns=timer.stop(),
                payload={
                    "phase_name": name,
                    **({"device": device} if device else {}),
                },
            )
        )


@contextmanager
def time_memcpy(
    *,
    copy_kind: str = "h2d",
    step: int | None = None,
    parent_span_id: str | None = None,
    src_device: str | None = None,
    dst_device: str | None = None,
    num_bytes: int | None = None,
    non_blocking: bool | None = None,
    device: str | None = None,
) -> Iterator[dict[str, Any]]:
    timer = _TimerBackend(device or dst_device)
    span_id = f"memcpy-{copy_kind}-{uuid.uuid4().hex[:8]}"

    begin_span(
        build_runtime_event(
            event_type="memcpy_span",
            step_id=step,
            span_id=span_id,
            parent_span_id=parent_span_id,
            duration_ns=0,
            payload=_memcpy_payload(
                copy_kind=copy_kind,
                src_device=src_device,
                dst_device=dst_device,
                num_bytes=num_bytes,
                non_blocking=non_blocking,
            ),
        )
    )

    try:
        yield {
            "copy_kind": copy_kind,
            "step": step,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "src_device": src_device,
            "dst_device": dst_device,
            "bytes": num_bytes,
            "non_blocking": non_blocking,
        }
    finally:
        end_span(
            build_runtime_event(
                event_type="memcpy_span",
                step_id=step,
                span_id=span_id,
                parent_span_id=parent_span_id,
                duration_ns=timer.stop(),
                payload=_memcpy_payload(
                    copy_kind=copy_kind,
                    src_device=src_device,
                    dst_device=dst_device,
                    num_bytes=num_bytes,
                    non_blocking=non_blocking,
                ),
            )
        )


def time_region(
    name: str,
    *,
    step: int | None = None,
    parent_span_id: str | None = None,
    device: str | None = None,
) -> Iterator[dict[str, Any]]:
    return time_phase(name, step=step, parent_span_id=parent_span_id, device=device)


def _memcpy_payload(
    *,
    copy_kind: str,
    src_device: str | None,
    dst_device: str | None,
    num_bytes: int | None,
    non_blocking: bool | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "copy_kind": copy_kind,
    }
    if src_device:
        payload["src_device"] = src_device
    if dst_device:
        payload["dst_device"] = dst_device
    if num_bytes is not None:
        payload["bytes"] = num_bytes
    if non_blocking is not None:
        payload["non_blocking"] = non_blocking
    return payload


def _cuda_events_supported(device: str | None) -> bool:
    if torch is None or not hasattr(torch, "cuda"):
        return False
    if not torch.cuda.is_available():
        return False
    if device is not None and not str(device).startswith("cuda"):
        return False
    return True
