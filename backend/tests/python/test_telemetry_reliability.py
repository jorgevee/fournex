"""Tests for telemetry reliability hardening: streaming durability, the native
persistence guard, and nvidia-smi sampler resilience."""
import json
import sys
from pathlib import Path
from threading import Event

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex import cli, sdk, storage


@pytest.fixture(autouse=True)
def _reset_sdk_state():
    sdk._close_trace_stream()
    sdk.clear_local_events()
    yield
    sdk._close_trace_stream()
    sdk.clear_local_events()


def _emit(n: int):
    for i in range(n):
        sdk.emit_event(sdk.build_runtime_event(event_type="step_start", step_id=i))


# ── Streaming durability ──────────────────────────────────────────────────────

def test_events_stream_to_disk_before_persist(tmp_path):
    trace = tmp_path / "raw" / "trace.jsonl"
    sdk.init(raw_trace_path=str(trace), stream_trace=True, run_id="stream-test")
    _emit(3)
    sdk.flush()  # force the buffered handle out without a clean persist

    # Simulate a crash: persist never runs, yet the trace is already on disk.
    assert trace.exists()
    lines = [l for l in trace.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    assert all(json.loads(l)["event_type"] == "step_start" for l in lines)


def test_streaming_off_by_default_for_library_use(tmp_path, monkeypatch):
    monkeypatch.delenv("FRX_AUTO_PERSIST", raising=False)
    monkeypatch.delenv("FRX_STREAM_TRACE", raising=False)
    trace = tmp_path / "raw" / "trace.jsonl"
    sdk.init(raw_trace_path=str(trace), run_id="no-stream")
    _emit(2)
    sdk.flush()
    assert not trace.exists()  # no surprise files when nobody asked to persist
    assert len(sdk.get_local_events()) == 2


def test_clean_persist_rewrites_canonical_trace(tmp_path):
    trace = tmp_path / "raw" / "trace.jsonl"
    summary = tmp_path / "derived" / "summary.json"
    sdk.init(raw_trace_path=str(trace), derived_summary_path=str(summary),
             stream_trace=True, run_id="clean")
    _emit(4)
    sdk._close_trace_stream()  # release the streaming handle (as auto-persist does)
    out = storage.persist_run_artifacts()
    lines = [l for l in Path(out["raw_trace_path"]).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 4
    assert Path(out["derived_summary_path"]).exists()


# ── Native persistence guard ──────────────────────────────────────────────────

def test_persist_refuses_empty_under_native(monkeypatch):
    monkeypatch.setattr(storage, "HAS_NATIVE", True)
    with pytest.raises(RuntimeError, match="native telemetry backend"):
        storage.persist_run_artifacts()


def test_persist_with_explicit_events_works_under_native(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "HAS_NATIVE", True)
    events = [sdk.build_runtime_event(event_type="step_start", step_id=0)]
    out = storage.persist_local_trace(events, output_path=str(tmp_path / "t.jsonl"))
    assert Path(out).read_text(encoding="utf-8").strip()


# ── nvidia-smi sampler resilience ─────────────────────────────────────────────

def test_sampler_survives_transient_failures(tmp_path, monkeypatch):
    calls = {"n": 0}
    stop = Event()

    class _FakeProc:
        stdout = "0, GPU, 50, 10, 1000, 8000, 4, 16"

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("transient nvidia-smi hiccup")
        if calls["n"] >= 3:
            stop.set()  # let the loop exit after one success
        return _FakeProc()

    monkeypatch.setattr(cli.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    out = tmp_path / "gpu_metrics.csv"
    warnings: list[str] = []
    cli._sample_gpu_metrics(out, interval_ms=100, stop=stop, warnings=warnings)

    # It kept sampling past the failures (did not return on the first error)...
    assert calls["n"] >= 3
    # ...recorded a "continuing" warning rather than dying...
    assert any("continuing" in w for w in warnings)
    # ...and still wrote a data row after recovery.
    rows = [r for r in out.read_text(encoding="utf-8").splitlines() if r.strip()]
    assert len(rows) >= 2  # header + at least one sample
