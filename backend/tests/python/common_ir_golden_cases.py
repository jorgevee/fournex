PYTORCH_PROFILER_GOLDEN = {
    "traceEvents": [
        {
            "name": "volta_sgemm_128x64",
            "cat": "cuda_kernel",
            "ph": "X",
            "ts": 100,
            "dur": 25,
            "pid": 123,
            "tid": 9,
            "args": {"device": 0, "step": 3, "correlation": "corr_1"},
        },
        {
            "name": "GPU Utilization",
            "cat": "metric",
            "ph": "C",
            "ts": 110,
            "args": {"device": 0, "value": 82.5, "unit": "percent", "step": 3},
        },
    ]
}

NVML_GOLDEN = {
    "timestamp_ns": 1_000_000,
    "device_index": 0,
    "utilization_gpu_pct": 72.5,
    "utilization_mem_pct": 88.0,
    "memory_used_bytes": 95,
    "memory_total_bytes": 100,
    "temperature_c": 87.0,
    "power_w": 240.0,
}

DISTRIBUTED_GOLDEN = {
    "collective_type": "ncclAllReduceRingLLKernel",
    "backend": "nccl",
    "rank": 1,
    "world_size": 8,
    "ts_start_ns": 1000,
    "ts_end_ns": 2500,
    "tensor_bytes": 1048576,
    "communicator_id": "comm_1",
    "group_name": "dp_group",
    "stream_id": 7,
    "overlap_with_compute": 0.42,
    "wait_time_ns": 200,
    "active_time_ns": 1300,
    "host_id": "host_1",
    "process_id": 4321,
    "thread_id": 17,
    "device_id": "gpu1",
    "step_id": "step_12",
    "correlation_id": "corr_dist_1",
}

DATA_PIPELINE_GOLDEN = {
    "stage": "next",
    "ts_start_ns": 1000,
    "ts_end_ns": 5000,
    "batch_size": 32,
    "num_workers": 4,
    "prefetch_factor": 2,
    "pinned_memory": True,
    "step_id": "step_5",
    "span_id": "dl_span_1",
    "correlation_id": "corr_dl_1",
}
