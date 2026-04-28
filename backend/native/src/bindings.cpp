#include <optional>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

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

// Fields that are identical for every event in a run — cached once at init().
// Eliminates 5 py::cast<> calls per emit that would otherwise re-derive the
// same strings from Python on every event.
struct RunStatics {
    std::string schema_version;
    std::string job_id;
    std::string run_id;
    std::string event_source;
    int pid = 0;
};

RunStatics& Statics() {
    static RunStatics s;
    return s;
}

PayloadMap NormalizePayload(const py::dict& payload,
                            std::pmr::polymorphic_allocator<> alloc) {
    PayloadMap normalized(alloc);
    normalized.reserve(static_cast<std::size_t>(py::len(payload)));
    for (const auto& item : payload) {
        normalized.emplace_back(py::cast<std::string>(item.first),
                                py::str(item.second).cast<std::string>());
    }
    return normalized;
}

std::optional<int> OptionalInt(const py::object& value) {
    if (value.is_none()) return std::nullopt;
    return value.cast<int>();
}

std::optional<std::int64_t> OptionalInt64(const py::object& value) {
    if (value.is_none()) return std::nullopt;
    return value.cast<std::int64_t>();
}

std::optional<std::string> OptionalString(const py::object& value) {
    if (value.is_none()) return std::nullopt;
    return value.cast<std::string>();
}

// Converts a Python event dict to a C++ TelemetryEvent.
// Static run-scoped fields (schema_version, job_id, run_id, event_source, pid)
// are copied from RunStatics — no py::cast for those fields.
// PayloadMap is constructed with the engine's pool allocator so it moves O(1)
// into the ring buffer slot when allocators compare equal.
TelemetryEvent EventFromDict(const py::dict& event) {
    const auto& s = Statics();
    auto alloc = EngineInstance().GetAllocator();
    TelemetryEvent ev(alloc);
    ev.schema_version  = s.schema_version;
    ev.event_id        = py::cast<std::string>(event["event_id"]);
    ev.timestamp_ns    = py::cast<std::int64_t>(event["timestamp_ns"]);
    ev.pid             = s.pid;
    ev.tid             = py::cast<int>(event["tid"]);
    ev.job_id          = s.job_id;
    ev.run_id          = s.run_id;
    ev.event_type      = py::cast<std::string>(event["event_type"]);
    ev.event_source    = s.event_source;
    ev.gpu_id          = OptionalInt(event.contains("gpu_id") ? event["gpu_id"] : py::none());
    ev.step_id         = OptionalInt(event.contains("step_id") ? event["step_id"] : py::none());
    ev.span_id         = OptionalString(event.contains("span_id") ? event["span_id"] : py::none());
    ev.parent_span_id  = OptionalString(event.contains("parent_span_id") ? event["parent_span_id"] : py::none());
    ev.duration_ns     = OptionalInt64(event.contains("duration_ns") ? event["duration_ns"] : py::none());
    ev.level           = py::cast<std::string>(event["level"]);
    ev.payload         = NormalizePayload(py::cast<py::dict>(event["payload"]), alloc);
    return ev;
}

void InitEngine(const std::string& job_name, const std::string& output_path,
                int sample_interval_ms, const std::string& run_id,
                bool enable_cupti, bool cupti_debug_mode) {
    // Cache run-scoped static fields so EventFromDict doesn't py::cast them
    // repeatedly — they are constant for the lifetime of this engine instance.
    auto& s = Statics();
    s.schema_version = std::string(kSchemaVersion);
    s.job_id = job_name;
    s.run_id = run_id;
    s.event_source = "python_sdk";
    s.pid = static_cast<int>(
#ifdef _WIN32
        static_cast<int>(GetCurrentProcessId())
#else
        static_cast<int>(getpid())
#endif
    );

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
