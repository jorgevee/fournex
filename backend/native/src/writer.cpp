#include "writer.h"

#include <fstream>
#include <sstream>

namespace autopilot::telemetry {
namespace {

std::string EscapeJson(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value) {
        switch (ch) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped += ch;
                break;
        }
    }
    return escaped;
}

void WriteOptionalInt(std::ostringstream& stream, const char* key,
                      const std::optional<int>& value) {
    stream << ",\"" << key << "\":";
    if (value.has_value()) {
        stream << *value;
    } else {
        stream << "null";
    }
}

void WriteOptionalInt64(std::ostringstream& stream, const char* key,
                        const std::optional<std::int64_t>& value) {
    stream << ",\"" << key << "\":";
    if (value.has_value()) {
        stream << *value;
    } else {
        stream << "null";
    }
}

void WriteOptionalString(std::ostringstream& stream, const char* key,
                         const std::optional<std::string>& value) {
    stream << ",\"" << key << "\":";
    if (value.has_value()) {
        stream << "\"" << EscapeJson(*value) << "\"";
    } else {
        stream << "null";
    }
}

std::string SerializeEvent(const TelemetryEvent& event) {
    std::ostringstream stream;
    stream << "{";
    stream << "\"schema_version\":\"" << EscapeJson(event.schema_version) << "\"";
    stream << ",\"event_id\":\"" << EscapeJson(event.event_id) << "\"";
    stream << ",\"timestamp_ns\":" << event.timestamp_ns;
    stream << ",\"pid\":" << event.pid;
    stream << ",\"tid\":" << event.tid;
    stream << ",\"job_id\":\"" << EscapeJson(event.job_id) << "\"";
    stream << ",\"run_id\":\"" << EscapeJson(event.run_id) << "\"";
    stream << ",\"event_type\":\"" << EscapeJson(event.event_type) << "\"";
    stream << ",\"event_source\":\"" << EscapeJson(event.event_source) << "\"";
    WriteOptionalInt(stream, "gpu_id", event.gpu_id);
    WriteOptionalInt(stream, "step_id", event.step_id);
    WriteOptionalString(stream, "span_id", event.span_id);
    WriteOptionalString(stream, "parent_span_id", event.parent_span_id);
    WriteOptionalInt64(stream, "duration_ns", event.duration_ns);
    stream << ",\"level\":\"" << EscapeJson(event.level) << "\"";
    stream << ",\"payload\":{";

    bool first = true;
    for (const auto& [key, value] : event.payload) {
        if (!first) {
            stream << ",";
        }
        first = false;
        stream << "\"" << EscapeJson(key) << "\":\"" << EscapeJson(value) << "\"";
    }

    stream << "}}";
    return stream.str();
}

}  // namespace

TraceWriter::TraceWriter(std::filesystem::path output_path)
    : output_path_(std::move(output_path)) {}

void TraceWriter::WriteBatch(const std::vector<TelemetryEvent>& events) {
    if (events.empty()) {
        return;
    }

    const auto parent = output_path_.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(output_path_, std::ios::app);
    for (const auto& event : events) {
        out << SerializeEvent(event) << '\n';
    }
}

void TraceWriter::Flush() {}

const std::filesystem::path& TraceWriter::output_path() const {
    return output_path_;
}

}  // namespace autopilot::telemetry
