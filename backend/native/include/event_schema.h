#pragma once

#include <array>
#include <string_view>

namespace autopilot::telemetry {

inline constexpr std::string_view kSchemaVersion = "0.1.0";

enum class EventSource {
    kPythonSdk,
    kNativeEngine,
};

enum class EventLevel {
    kDebug,
    kInfo,
    kWarning,
    kError,
};

inline constexpr std::array<std::string_view, 11> kEventTypes = {
    "gpu_sample",
    "step_start",
    "step_end",
    "phase_span",
    "dataloader_span",
    "memcpy_span",
    "shape_snapshot",
    "sync_wait",
    "profiler_window",
    "system_info",
    "warning_annotation",
};

}  // namespace autopilot::telemetry
