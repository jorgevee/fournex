# Common IR Examples

## Canonical V1 Run Example

```json
{
  "schema_version": "1.0.0",
  "run_id": "run_demo_001",
  "job": {
    "job_id": "bert_train_demo",
    "status": "completed",
    "workload_class": "training"
  },
  "workload": {
    "model_family": "transformer",
    "model_name": "bert-base",
    "precision_mode": "bf16",
    "batch_size": 32,
    "sequence_length": 512
  },
  "events": [
    {
      "event_id": "evt_step_1",
      "event_family": "cpu",
      "event_type": "training_step",
      "ts_start_ns": 1000,
      "ts_end_ns": 5000,
      "duration_ns": 4000,
      "clock_domain": "monotonic",
      "host_id": "host_0",
      "process_id": 4242,
      "thread_id": 88,
      "device_id": null,
      "step_id": "step_1",
      "span_id": "span_step_1",
      "parent_span_id": null,
      "correlation_id": "corr_step_1",
      "source": "python_sdk",
      "attrs": {
        "step_kind": "train"
      }
    },
    {
      "event_id": "evt_kernel_1",
      "event_family": "kernel",
      "event_type": "cuda_kernel",
      "ts_start_ns": 2000,
      "ts_end_ns": 2600,
      "duration_ns": 600,
      "clock_domain": "monotonic",
      "host_id": "host_0",
      "process_id": 4242,
      "thread_id": 88,
      "device_id": "gpu0",
      "step_id": "step_1",
      "span_id": "span_kernel_1",
      "parent_span_id": "span_step_1",
      "correlation_id": "corr_kernel_1",
      "source": "pytorch_profiler",
      "attrs": {
        "kernel_name_raw": "volta_sgemm_128x64",
        "kernel_class_canonical": "gemm"
      }
    }
  ],
  "metrics": [
    {
      "metric_id": "metric_gpu_util_1",
      "metric_name": "gpu_utilization",
      "metric_unit": "percent",
      "value": 81.2,
      "ts_ns": 2500,
      "clock_domain": "monotonic",
      "host_id": "host_0",
      "device_id": "gpu0",
      "step_id": "step_1",
      "source": "nvml",
      "attrs": {}
    }
  ],
  "annotations": [
    {
      "annotation_id": "ann_1",
      "annotation_type": "bottleneck",
      "target_id": "step_1",
      "label": "input_bound",
      "score": 0.82,
      "source": "rule_engine_v1",
      "attrs": {
        "evidence": "dataloader_wait_fraction=0.31"
      }
    }
  ]
}
```
