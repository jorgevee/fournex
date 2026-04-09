#include <optional>
#include <string>
#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "event_schema.h"
#include "telemetry_engine.h"

namespace py = pybind11;

namespace autopilot::telemetry {
namespace {

TelemetryEngine& EngineInstance() {
    static TelemetryEngine engine;
    return engine;
}

std::unordered_map<std::string, std::string> NormalizePayload(const py::dict& payload) {
    std::unordered_map<std::string, std::string> normalized;
    for (const auto& item : payload) {
        normalized.emplace(py::cast<std::string>(item.first),
                           py::str(item.second).cast<std::string>());
    }
    return normalized;
}

std::optional<int> OptionalInt(const py::object& value) {
    if (value.is_none()) {
        return std::nullopt;
    }
    return value.cast<int>();
}

std::optional<std::int64_t> OptionalInt64(const py::object& value) {
    if (value.is_none()) {
        return std::nullopt;
    }
    return value.cast<std::int64_t>();
}

std::optional<std::string> OptionalString(const py::object& value) {
    if (value.is_none()) {
        return std::nullopt;
    }
    return value.cast<std::string>();
}

TelemetryEvent EventFromDict(const py::dict& event) {
    return TelemetryEvent{
        py::cast<std::string>(event["schema_version"]),
        py::cast<std::string>(event["event_id"]),
        py::cast<std::int64_t>(event["timestamp_ns"]),
        py::cast<int>(event["pid"]),
        py::cast<int>(event["tid"]),
        py::cast<std::string>(event["job_id"]),
        py::cast<std::string>(event["run_id"]),
        py::cast<std::string>(event["event_type"]),
        py::cast<std::string>(event["event_source"]),
        OptionalInt(event.contains("gpu_id") ? event["gpu_id"] : py::none()),
        OptionalInt(event.contains("step_id") ? event["step_id"] : py::none()),
        OptionalString(event.contains("span_id") ? event["span_id"] : py::none()),
        OptionalString(event.contains("parent_span_id") ? event["parent_span_id"] : py::none()),
        OptionalInt64(event.contains("duration_ns") ? event["duration_ns"] : py::none()),
        py::cast<std::string>(event["level"]),
        NormalizePayload(py::cast<py::dict>(event["payload"])),
    };
}

void InitEngine(const std::string& job_name, const std::string& output_path,
                int sample_interval_ms, const std::string& run_id,
                bool enable_cupti, bool cupti_debug_mode) {
    TelemetryEngineOptions options;
    options.job_id = job_name;
    options.output_path = output_path;
    options.sample_interval_ms = sample_interval_ms;
    options.run_id = run_id;
    options.enable_cupti = enable_cupti;
    options.cupti_debug_mode = cupti_debug_mode;
    EngineInstance().Initialize(options);
}

void EmitEvent(const py::dict& event) {
    EngineInstance().Emit(EventFromDict(event));
}

void BeginSpan(const py::dict& event) {
    EngineInstance().Emit(EventFromDict(event));
}

void EndSpan(const py::dict& event) {
    EngineInstance().Emit(EventFromDict(event));
}

}  // namespace
}  // namespace autopilot::telemetry

PYBIND11_MODULE(_autopilot_telemetry_native, module) {
    module.doc() = "Native telemetry engine bindings";
    module.attr("SCHEMA_VERSION") = autopilot::telemetry::kSchemaVersion;
    module.def("init", &autopilot::telemetry::InitEngine, py::arg("job_name"),
               py::arg("output_path"), py::arg("sample_interval_ms"),
               py::arg("run_id"), py::arg("enable_cupti") = false,
               py::arg("cupti_debug_mode") = false);
    module.def("shutdown",
               [] { autopilot::telemetry::EngineInstance().Shutdown(); });
    module.def("flush", [] { autopilot::telemetry::EngineInstance().Flush(); });
    module.def("emit_event", &autopilot::telemetry::EmitEvent, py::arg("event"));
    module.def("begin_span", &autopilot::telemetry::BeginSpan, py::arg("event"));
    module.def("end_span", &autopilot::telemetry::EndSpan, py::arg("event"));
}
