from __future__ import annotations

import time
import uuid
from collections.abc import Iterable, Iterator
from typing import Any

from .sdk import build_runtime_event, emit_event
from .shapes import extract_shapes, infer_batch_size


class InstrumentedDataLoader:
    def __init__(self, loader: Iterable[Any], *, loader_name: str | None = None):
        self._loader = loader
        self._loader_name = loader_name or type(loader).__name__
        self._metadata_emitted = False

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self._loader)
        if not self._metadata_emitted:
            self._emit_loader_metadata()
            self._metadata_emitted = True
        return _InstrumentedDataLoaderIterator(iterator, self._loader, self._loader_name)

    def _emit_loader_metadata(self) -> None:
        emit_event(
            build_runtime_event(
                event_type="dataloader_span",
                duration_ns=0,
                payload=_build_loader_payload(
                    self._loader,
                    stage="transfer_ready",
                    batch_size=_get_config_batch_size(self._loader),
                ),
            )
        )


class _InstrumentedDataLoaderIterator:
    def __init__(self, iterator: Iterator[Any], loader: Iterable[Any], loader_name: str):
        self._iterator = iterator
        self._loader = loader
        self._loader_name = loader_name
        self._batch_index = 0

    def __iter__(self) -> "_InstrumentedDataLoaderIterator":
        return self

    def __next__(self) -> Any:
        start_ns = time.perf_counter_ns()
        try:
            batch = next(self._iterator)
        except StopIteration:
            raise
        duration_ns = time.perf_counter_ns() - start_ns

        shapes = extract_shapes(batch)
        batch_size = infer_batch_size(shapes) or _get_config_batch_size(self._loader)
        span_id = f"dataloader-{self._batch_index}-{uuid.uuid4().hex[:8]}"

        emit_event(
            build_runtime_event(
                event_type="dataloader_span",
                step_id=self._batch_index,
                span_id=span_id,
                duration_ns=duration_ns,
                payload=_build_loader_payload(
                    self._loader,
                    stage="next",
                    batch_size=batch_size,
                ),
            )
        )

        self._batch_index += 1
        return batch


def instrument_dataloader(loader: Iterable[Any], *, loader_name: str | None = None) -> InstrumentedDataLoader:
    return InstrumentedDataLoader(loader, loader_name=loader_name)


def _build_loader_payload(
    loader: Iterable[Any],
    *,
    stage: str,
    batch_size: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage,
    }

    num_workers = getattr(loader, "num_workers", None)
    if isinstance(num_workers, int):
        payload["num_workers"] = num_workers

    prefetch_factor = getattr(loader, "prefetch_factor", None)
    if isinstance(prefetch_factor, int):
        payload["prefetch_factor"] = prefetch_factor

    pinned_memory = getattr(loader, "pin_memory", None)
    if isinstance(pinned_memory, bool):
        payload["pinned_memory"] = pinned_memory

    if isinstance(batch_size, int):
        payload["batch_size"] = batch_size

    return payload


def _get_config_batch_size(loader: Iterable[Any]) -> int | None:
    batch_size = getattr(loader, "batch_size", None)
    if isinstance(batch_size, int):
        return batch_size
    return None
