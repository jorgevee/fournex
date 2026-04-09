from __future__ import annotations

try:
    from . import _autopilot_telemetry_native as native
except ImportError:
    native = None


HAS_NATIVE = native is not None
