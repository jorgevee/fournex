import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import autopilot_telemetry as at

from common_ir_golden_cases import (
    DATA_PIPELINE_GOLDEN,
    DISTRIBUTED_GOLDEN,
    NVML_GOLDEN,
    PYTORCH_PROFILER_GOLDEN,
)


def test_pytorch_profiler_golden_mapper() -> None:
    trace = at.PytorchProfilerTrace.from_json_payload(PYTORCH_PROFILER_GOLDEN)
    events, metrics = at.map_pytorch_profiler_to_ir(trace, run_id="run_prof_001")
    assert len(events) == 1
    assert len(metrics) == 1
    at.validate_event_record(events[0])
    at.validate_metric_record(metrics[0])
    assert events[0].event_family == "kernel"
    assert metrics[0].metric_name == "gpu_utilization"


def test_nvml_golden_mapper() -> None:
    sample = at.NvmlSampleRecord.from_dict(NVML_GOLDEN)
    metrics, annotations = at.map_nvml_sample_to_ir(sample, run_id="run_nvml_001", step_id="step_3")
    assert len(metrics) >= 4
    assert len(annotations) == 2
    for metric in metrics:
        at.validate_metric_record(metric)
    for annotation in annotations:
        at.validate_annotation_record(annotation)
    assert annotations[0].attrs["raw_sample"]["device_index"] == 0


def test_distributed_golden_mapper() -> None:
    record = at.DistributedCommRecord.from_dict(DISTRIBUTED_GOLDEN)
    event = at.map_distributed_record_to_ir(record, run_id="run_dist_001")
    at.validate_event_record(event)
    assert event.event_family == "distributed"
    assert event.attrs["collective_op"] == "all_reduce"


def test_data_pipeline_golden_mapper() -> None:
    record = at.DataPipelineRecord.from_dict(DATA_PIPELINE_GOLDEN)
    event = at.map_data_pipeline_record_to_ir(record, run_id="run_data_001")
    at.validate_event_record(event)
    assert event.event_family == "data_pipeline"
    assert event.event_type == "dataloader_wait"


def test_run_semantic_warnings_for_missing_step_reference() -> None:
    run = at.RunRecord(
        run_id="run_1",
        job=at.JobInfo(job_id="job_1", workload_class="training", status="completed"),
        workload=at.WorkloadInfo(),
        metrics=[
            at.MetricRecord(
                metric_id="metric_1",
                run_id="run_1",
                metric_name="gpu_utilization",
                metric_unit="percent",
                value=90.0,
                ts_ns=100,
                source="nvml",
                step_id="step_missing",
            )
        ],
    )
    warnings = at.semantic_warnings_for_run(run)
    assert warnings
    assert "missing step_id" in warnings[0]
