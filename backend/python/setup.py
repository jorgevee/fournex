from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup


def build_extensions() -> list[Extension]:
    if os.environ.get("BUILD_NATIVE") != "1":
        return []

    try:
        import pybind11
    except ImportError as exc:
        raise RuntimeError(
            "BUILD_NATIVE=1 requires pybind11 to be installed in the active environment. "
            "Install it first, then rerun with --no-build-isolation."
        ) from exc

    root = Path(__file__).resolve().parent.parent
    native_include = root / "native" / "include"
    native_src = root / "native" / "src"

    sources = [
        native_src / "bindings.cpp",
        native_src / "clock.cpp",
        native_src / "event_buffer.cpp",
        native_src / "nvml_sampler.cpp",
        native_src / "telemetry_engine.cpp",
        native_src / "writer.cpp",
    ]

    return [
        Extension(
            "fournex._fournex_native",
            sources=[str(source) for source in sources],
            include_dirs=[str(native_include), pybind11.get_include()],
            language="c++",
            extra_compile_args=["/std:c++17"] if os.name == "nt" else ["-std=c++17"],
        )
    ]


setup(ext_modules=build_extensions())
