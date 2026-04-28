import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from autopilot_telemetry.autopilot.tuners import generate_all_candidates


def _ids(candidates):
    return [candidate.config_id for candidate in candidates]


def test_input_bound_focuses_dataloader_candidates():
    candidates = generate_all_candidates(
        environment={"cpu_count": 8},
        max_total=4,
        bottleneck_diagnosis="input_pipeline_bound",
    )

    assert _ids(candidates) == ["dl_nw0_pin1", "dl_nw2_pin1", "dl_nw2_pin0", "dl_nw4_pin1"]
    assert all(candidate.actions[0].type == "dataloader" for candidate in candidates)


def test_copy_bound_focuses_pinned_memory_candidates():
    candidates = generate_all_candidates(
        environment={"cpu_count": 8},
        max_total=3,
        bottleneck_diagnosis={"primary_bottleneck": "copy_bound"},
    )

    assert _ids(candidates) == ["dl_nw0_pin1", "dl_nw2_pin1", "dl_nw4_pin1"]
    assert all(candidate.env_vars["FRX_PIN_MEMORY"] == "true" for candidate in candidates)


def test_launch_bound_generates_runtime_candidates_when_validated_actions_allowed():
    candidates = generate_all_candidates(
        environment={"static_shapes": True},
        max_total=5,
        safe_only=False,
        bottleneck_diagnosis={"diagnosis": {"primary_bottleneck": "launch_bound"}},
    )

    assert _ids(candidates) == [
        "compile_default",
        "compile_reduce_overhead",
        "cuda_graphs_try_static",
    ]


def test_launch_bound_skips_cuda_graphs_for_dynamic_shapes():
    candidates = generate_all_candidates(
        environment={"shapes_dynamic": True},
        max_total=5,
        safe_only=False,
        bottleneck_diagnosis="small_kernel_overhead",
    )

    assert _ids(candidates) == ["compile_default", "compile_reduce_overhead"]


def test_memory_pressure_includes_allocator_then_precision_when_allowed():
    candidates = generate_all_candidates(
        environment={
            "cuda_available": True,
            "gpu_name": "NVIDIA A100",
        },
        max_total=4,
        safe_only=False,
        bottleneck_diagnosis="memory_bound",
    )

    assert _ids(candidates) == [
        "allocator_max_split_128",
        "allocator_expandable_segments",
        "amp_bf16",
        "amp_fp16",
    ]


def test_steady_state_summary_diagnosis_is_supported():
    candidates = generate_all_candidates(
        environment={"cpu_count": 4},
        max_total=2,
        bottleneck_diagnosis={
            "steady_state": {
                "diagnosis": {
                    "primary_bottleneck": "copy_bound",
                }
            }
        },
    )

    assert _ids(candidates) == ["dl_nw0_pin1", "dl_nw2_pin1"]
