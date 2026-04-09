from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def extract_shapes(batch: Any) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    _walk_batch(batch, shapes, {}, "batch")
    return shapes


def extract_dtypes(batch: Any) -> dict[str, str]:
    dtypes: dict[str, str] = {}
    _walk_batch(batch, {}, dtypes, "batch")
    return dtypes


def describe_batch(batch: Any) -> dict[str, Any]:
    shapes = extract_shapes(batch)
    dtypes = extract_dtypes(batch)

    description: dict[str, Any] = {
        "batch_size": infer_batch_size(shapes),
        "shapes": shapes,
    }

    sequence_length = infer_sequence_length(shapes)
    if sequence_length is not None:
        description["sequence_length"] = sequence_length

    if dtypes:
        description["dtypes"] = dtypes

    return description


def infer_batch_size(shapes: dict[str, list[int]]) -> int:
    for shape in shapes.values():
        if shape:
            return shape[0]
    return 0


def infer_sequence_length(shapes: dict[str, list[int]]) -> int | None:
    for shape in shapes.values():
        if len(shape) >= 2:
            return shape[1]
    return None


def _walk_batch(
    value: Any,
    shapes: dict[str, list[int]],
    dtypes: dict[str, str],
    prefix: str,
) -> None:
    shape = _shape_of(value)
    if shape is not None:
        shapes[prefix] = shape
        dtype = _dtype_of(value)
        if dtype is not None:
            dtypes[prefix] = dtype
        return

    if isinstance(value, Mapping):
        for key, child in value.items():
            _walk_batch(child, shapes, dtypes, f"{prefix}.{key}")
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _walk_batch(child, shapes, dtypes, f"{prefix}[{index}]")


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None

    try:
        return [int(dim) for dim in shape]
    except TypeError:
        return None


def _dtype_of(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype)
