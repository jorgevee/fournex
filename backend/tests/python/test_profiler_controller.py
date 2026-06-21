"""Tests for the SDK sampled profiler: kernel-count capture + collect wiring.

These cover the fix that makes `frx collect` feed real CUDA kernel counts to the
launch_bound / framework_abstraction_tax classifiers through the normal SDK step
loop (previously the controller was a metadata-only stub that emitted no counts).
The real torch.profiler capture itself is validated on GPU; here we cover the
pure kernel-counting reducer, the emitted event schema, and the env gating.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex import sdk
from fournex import profiler as prof
from fournex.profiler import (
    ProfilerSchedule,
    SampledProfilerController,
    summarize_chrome_trace_kernels,
)


# ── pure kernel-count reducer ─────────────────────────────────────────────────

def test_summarize_chrome_trace_kernels_counts_and_per_step():
    events = {"traceEvents": [
        {"cat": "kernel", "dur": 5.0},
        {"cat": "kernel", "dur": 15.0},
        {"cat": "kernel", "dur": 3.0},
        {"cat": "cpu_op", "dur": 100.0},   # not a kernel
        {"cat": "kernel", "dur": 0},       # zero-duration ignored
    ]}
    s = summarize_chrome_trace_kernels(events, n_steps=2)
    assert s["kernel_count_total"] == 3
    assert s["kernel_count"] == 1.5                       # 3 kernels / 2 steps
    assert s["median_cuda_kernel_duration_us"] == 5.0     # median of [3,5,15]
    assert s["small_kernel_fraction"] == round(2 / 3, 4)  # 3us and 5us are < 10us


def test_summarize_chrome_trace_kernels_accepts_bare_list_and_empty():
    assert summarize_chrome_trace_kernels([], n_steps=5)["kernel_count"] == 0
    bare = [{"cat": "kernel", "dur": 4.0}]
    assert summarize_chrome_trace_kernels(bare, n_steps=1)["kernel_count_total"] == 1


# ── controller emits the kernel-count schema ──────────────────────────────────

def test_exported_window_carries_kernel_count_fields():
    sdk.clear_local_events()
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = SampledProfilerController(
            ProfilerSchedule(wait=0, warmup=0, record=1, repeat=1),
            output_dir=tmp,
            enabled=True,
        )
        ctrl.on_step_start(0)
        ctrl.on_step_end(0)

    exported = [
        e for e in sdk.get_local_events()
        if e.get("event_type") == "profiler_window"
        and e.get("payload", {}).get("window_state") == "exported"
    ]
    assert exported, "controller did not emit an exported profiler_window"
    payload = exported[0]["payload"]
    # The reducer reads exactly these keys to drive launch_bound / framework tax.
    for key in ("kernel_count", "median_cuda_kernel_duration_us", "small_kernel_fraction"):
        assert key in payload, f"exported window missing {key}"
        assert isinstance(payload[key], (int, float))


# ── init env-gating (only auto-enable under `frx collect`) ─────────────────────

def _init_clean(monkeypatch, tmp, **env):
    prof._controller = None
    monkeypatch.setenv("FRX_STREAM_TRACE", "0")
    monkeypatch.delenv("FRX_AUTO_PERSIST", raising=False)
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    sdk.init(job_name="t", output_dir=tmp)


def test_init_enables_profiler_under_collect_env(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _init_clean(monkeypatch, tmp, FRX_PROFILER_ENABLED="1", FRX_PROFILER=None)
    assert prof.get_profiler_controller() is not None


def test_init_profiler_opt_out(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _init_clean(monkeypatch, tmp, FRX_PROFILER_ENABLED="1", FRX_PROFILER="0")
    assert prof.get_profiler_controller() is None


def test_init_no_profiler_without_collect_env(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _init_clean(monkeypatch, tmp, FRX_PROFILER_ENABLED=None, FRX_PROFILER=None)
    assert prof.get_profiler_controller() is None
