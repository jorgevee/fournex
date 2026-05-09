# INPUT_BOUND
# Intended primary: input_bound (avg DataLoader fraction ~32.5%, above 0.20 threshold).
# GPU utilization is 40-42% — above the underutilized_gpu threshold of 35%.
# No sync, copy, or shape signals. Only DataLoader wait drives the diagnosis.
INPUT_BOUND_EVENTS = [
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 40, "utilization_mem_pct": 35, "memory_used_bytes": 40, "memory_total_bytes": 100}},
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 42, "utilization_mem_pct": 37, "memory_used_bytes": 42, "memory_total_bytes": 100}},
    {"event_type": "step_start", "step_id": 1, "payload": {"step_kind": "train"}},
    {"event_type": "dataloader_span", "step_id": 1, "duration_ns": 35, "payload": {"stage": "next"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 20, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 20, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 10, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 1, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    {"event_type": "step_start", "step_id": 2, "payload": {"step_kind": "train"}},
    {"event_type": "dataloader_span", "step_id": 2, "duration_ns": 30, "payload": {"stage": "next"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 25, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 20, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 10, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 2, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
]

# COPY_BOUND
# Intended primary: copy_bound (avg H2D fraction ~19%, above 0.15 threshold).
# GPU utilization is 55-58% — well above the underutilized_gpu threshold.
# Two steps both have H2D copies; no sync or DataLoader stalls.
COPY_BOUND_EVENTS = [
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 55, "utilization_mem_pct": 40, "memory_used_bytes": 45, "memory_total_bytes": 100}},
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 58, "utilization_mem_pct": 42, "memory_used_bytes": 46, "memory_total_bytes": 100}},
    {"event_type": "step_start", "step_id": 1, "payload": {"step_kind": "train"}},
    {"event_type": "memcpy_span", "step_id": 1, "duration_ns": 20, "payload": {"copy_kind": "h2d"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 30, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 25, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 10, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 1, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    {"event_type": "step_start", "step_id": 2, "payload": {"step_kind": "train"}},
    {"event_type": "memcpy_span", "step_id": 2, "duration_ns": 18, "payload": {"copy_kind": "h2d"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 30, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 25, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 10, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 2, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
]

# SYNC_BOUND
# Intended primary: sync_bound (avg sync fraction ~13.5%, above 0.10 threshold).
# GPU utilization is 48-50% — above the underutilized_gpu threshold.
# Both steps have sync_wait events; no DataLoader or H2D stalls.
SYNC_BOUND_EVENTS = [
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 48, "utilization_mem_pct": 38, "memory_used_bytes": 40, "memory_total_bytes": 100}},
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 50, "utilization_mem_pct": 39, "memory_used_bytes": 41, "memory_total_bytes": 100}},
    {"event_type": "step_start", "step_id": 1, "payload": {"step_kind": "train"}},
    {"event_type": "sync_wait", "step_id": 1, "duration_ns": 15, "payload": {"wait_kind": "device_sync"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 35, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 25, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 10, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 1, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    {"event_type": "step_start", "step_id": 2, "payload": {"step_kind": "train"}},
    {"event_type": "sync_wait", "step_id": 2, "duration_ns": 12, "payload": {"wait_kind": "event_wait"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 35, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 25, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 10, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 2, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
]

# LAUNCH_BOUND
# Intended primary: underutilized_gpu (avg GPU 30%, below 35% threshold; score ~0.70).
# Secondary: launch_bound — profiler windows were exported (2 total) and no DataLoader/copy/sync
# stalls explain the low utilization, suggesting kernel launch overhead.
# Caveats: underutilized_gpu always outscores launch_bound here; launch_bound is definitively secondary.
LAUNCH_BOUND_EVENTS = [
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 28, "utilization_mem_pct": 22, "memory_used_bytes": 30, "memory_total_bytes": 100}},
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 32, "utilization_mem_pct": 24, "memory_used_bytes": 31, "memory_total_bytes": 100}},
    {"event_type": "step_start", "step_id": 1, "payload": {"step_kind": "train"}},
    {"event_type": "profiler_window", "step_id": 1, "payload": {"window_state": "exported"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 20, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 15, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 5, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 1, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    {"event_type": "step_start", "step_id": 2, "payload": {"step_kind": "train"}},
    {"event_type": "profiler_window", "step_id": 2, "payload": {"window_state": "exported"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 18, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 17, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 5, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 2, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
]

# LAUNCH_BOUND_TINY_KERNEL
# Intended primary: underutilized_gpu, with launch_bound as the actionable secondary.
# Profiler windows include many short kernels so evals can assert that kernel-level
# evidence survives summarization.
LAUNCH_BOUND_TINY_KERNEL_EVENTS = [
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 20, "utilization_mem_pct": 20, "memory_used_bytes": 30, "memory_total_bytes": 100}},
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 24, "utilization_mem_pct": 22, "memory_used_bytes": 31, "memory_total_bytes": 100}},
    {"event_type": "step_start", "step_id": 1, "payload": {"step_kind": "train"}},
    {"event_type": "profiler_window", "step_id": 1, "payload": {"window_state": "exported", "kernel_count": 180, "median_cuda_kernel_duration_us": 8.0, "small_kernel_fraction": 0.82}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 18, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 14, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 4, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 1, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    {"event_type": "step_start", "step_id": 2, "payload": {"step_kind": "train"}},
    {"event_type": "profiler_window", "step_id": 2, "payload": {"window_state": "exported", "kernel_count": 220, "median_cuda_kernel_duration_us": 7.0, "small_kernel_fraction": 0.86}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 16, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 15, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 4, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 2, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
]

# MEMORY_PRESSURE
# Intended primary: memory_pressure (peak ratio = 0.95, above 0.90 threshold).
# GPU utilization is 70-72% — healthy, no underutilization. No input/copy/sync stalls.
# Caveat: this case only has 2 steps; a real trace would need more samples to be confident.
MEMORY_PRESSURE_EVENTS = [
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 72, "utilization_mem_pct": 91, "memory_used_bytes": 93, "memory_total_bytes": 100}},
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 70, "utilization_mem_pct": 89, "memory_used_bytes": 95, "memory_total_bytes": 100}},
    {"event_type": "step_start", "step_id": 1, "payload": {"step_kind": "train"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 35, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 35, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 10, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 1, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    {"event_type": "step_start", "step_id": 2, "payload": {"step_kind": "train"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 35, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 35, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 10, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 2, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
]

# SHAPE_INSTABILITY
# Intended primary: shape_instability (volatility ratio = 1.0 — every step changes shape).
# Three steps alternate between seq_len=128 and seq_len=256; all transitions trigger a shape change.
# GPU utilization is 68-70% — healthy. No input/copy/sync stalls.
# Caveat: the small step count makes the volatility ratio maximally sensitive; real traces
# with stable interleavings may produce lower ratios that still exceed the 0.30 threshold.
SHAPE_INSTABILITY_EVENTS = [
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 68, "utilization_mem_pct": 50, "memory_used_bytes": 52, "memory_total_bytes": 100}},
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 70, "utilization_mem_pct": 52, "memory_used_bytes": 54, "memory_total_bytes": 100}},
    {"event_type": "step_start", "step_id": 1, "payload": {"step_kind": "train"}},
    {"event_type": "shape_snapshot", "step_id": 1, "payload": {"batch_size": 16, "sequence_length": 128, "shapes": {"input_ids": [16, 128], "attention_mask": [16, 128]}}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 30, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 30, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 10, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 1, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    {"event_type": "step_start", "step_id": 2, "payload": {"step_kind": "train"}},
    {"event_type": "shape_snapshot", "step_id": 2, "payload": {"batch_size": 16, "sequence_length": 256, "shapes": {"input_ids": [16, 256], "attention_mask": [16, 256]}}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 30, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 30, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 10, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 2, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    {"event_type": "step_start", "step_id": 3, "payload": {"step_kind": "train"}},
    {"event_type": "shape_snapshot", "step_id": 3, "payload": {"batch_size": 16, "sequence_length": 128, "shapes": {"input_ids": [16, 128], "attention_mask": [16, 128]}}},
    {"event_type": "phase_span", "step_id": 3, "duration_ns": 30, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 3, "duration_ns": 30, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 3, "duration_ns": 10, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 3, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
]

# MIXED_SIGNAL
# Intended: multiple bottlenecks trigger simultaneously with scores close enough to indicate ambiguity.
# underutilized_gpu (score ~0.66) is primary; input_bound (0.26) and copy_bound (0.18) are secondary.
# GPU utilization is 33-35% — just below the 35% threshold. All three stall types exceed their
# individual thresholds. This exercises contradiction handling and multi-label confidence scoring.
MIXED_SIGNAL_EVENTS = [
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 33, "utilization_mem_pct": 37, "memory_used_bytes": 45, "memory_total_bytes": 100}},
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 35, "utilization_mem_pct": 38, "memory_used_bytes": 46, "memory_total_bytes": 100}},
    {"event_type": "step_start", "step_id": 1, "payload": {"step_kind": "train"}},
    {"event_type": "dataloader_span", "step_id": 1, "duration_ns": 28, "payload": {"stage": "next"}},
    {"event_type": "memcpy_span", "step_id": 1, "duration_ns": 19, "payload": {"copy_kind": "h2d"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 20, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 18, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 5, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 1, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    {"event_type": "step_start", "step_id": 2, "payload": {"step_kind": "train"}},
    {"event_type": "dataloader_span", "step_id": 2, "duration_ns": 24, "payload": {"stage": "next"}},
    {"event_type": "memcpy_span", "step_id": 2, "duration_ns": 17, "payload": {"copy_kind": "h2d"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 21, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 18, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 5, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 2, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
]

# SPARSE_TELEMETRY
# Intended: insufficient_telemetry — no phase spans, no GPU samples, no stall events of any kind.
# Only step_start / step_end boundaries are present. The classifier cannot infer a root cause
# and should explicitly flag missing instrumentation rather than returning a silent None diagnosis.
SPARSE_TELEMETRY_EVENTS = [
    {"event_type": "step_start", "step_id": 1, "payload": {"step_kind": "train"}},
    {"event_type": "step_end", "step_id": 1, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    {"event_type": "step_start", "step_id": 2, "payload": {"step_kind": "train"}},
    {"event_type": "step_end", "step_id": 2, "duration_ns": 120, "payload": {"step_kind": "train", "status": "ok"}},
]

# UNDERUTILIZED_GPU (standalone)
# Intended primary: underutilized_gpu only, with high confidence.
# GPU utilization is ~23% — well below the 35% threshold. No profiler windows, so launch_bound
# does not co-fire. No DataLoader, copy, sync, or shape signals. This case isolates
# underutilized_gpu from the LAUNCH_BOUND case, which always pairs it with launch_bound as secondary.
UNDERUTILIZED_GPU_EVENTS = [
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 22, "utilization_mem_pct": 20, "memory_used_bytes": 30, "memory_total_bytes": 100}},
    {"event_type": "gpu_sample", "payload": {"utilization_gpu_pct": 24, "utilization_mem_pct": 21, "memory_used_bytes": 31, "memory_total_bytes": 100}},
    {"event_type": "step_start", "step_id": 1, "payload": {"step_kind": "train"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 20, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 15, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 1, "duration_ns": 5, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 1, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
    {"event_type": "step_start", "step_id": 2, "payload": {"step_kind": "train"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 22, "payload": {"phase_name": "forward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 14, "payload": {"phase_name": "backward"}},
    {"event_type": "phase_span", "step_id": 2, "duration_ns": 5, "payload": {"phase_name": "optimizer"}},
    {"event_type": "step_end", "step_id": 2, "duration_ns": 100, "payload": {"step_kind": "train", "status": "ok"}},
]
