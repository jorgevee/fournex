#include "writer.h"

#include <charconv>
#include <filesystem>

namespace autopilot::telemetry {
namespace {

// Appends s to buf with JSON escaping applied in-place — no temporary string.
void AppendEscaped(std::string& buf, std::string_view s) {
    for (const char ch : s) {
        switch (ch) {
            case '\\': buf += "\\\\"; break;
            case '"':  buf += "\\\""; break;
            case '\n': buf += "\\n";  break;
            case '\r': buf += "\\r";  break;
            case '\t': buf += "\\t";  break;
            default:   buf += ch;     break;
        }
    }
}

void AppendQuotedEscaped(std::string& buf, std::string_view s) {
    buf += '"';
    AppendEscaped(buf, s);
    buf += '"';
}

void AppendOptionalInt(std::string& buf, const char* key, const std::optional<int>& v) {
    buf += ",\"";
    buf += key;
    buf += "\":";
    if (v.has_value()) {
        char digits[12];
        const auto [end, _] = std::to_chars(digits, digits + sizeof(digits), *v);
        buf.append(digits, end);
    } else {
        buf += "null";
    }
}

void AppendOptionalInt64(std::string& buf, const char* key,
                         const std::optional<std::int64_t>& v) {
    buf += ",\"";
    buf += key;
    buf += "\":";
    if (v.has_value()) {
        char digits[20];
        const auto [end, _] = std::to_chars(digits, digits + sizeof(digits), *v);
        buf.append(digits, end);
    } else {
        buf += "null";
    }
}

void AppendOptionalString(std::string& buf, const char* key,
                          const std::optional<std::string>& v) {
    buf += ",\"";
    buf += key;
    buf += "\":";
    if (v.has_value()) {
        AppendQuotedEscaped(buf, *v);
    } else {
        buf += "null";
    }
}

// Serializes event into buf without any temporary string allocations.
void SerializeEventInto(const TelemetryEvent& event, std::string& buf) {
    buf.clear();
    buf.reserve(256);

    buf += "{\"schema_version\":";
    AppendQuotedEscaped(buf, event.schema_version);

    buf += ",\"event_id\":";
    AppendQuotedEscaped(buf, event.event_id);

    buf += ",\"timestamp_ns\":";
    {
        char digits[20];
        const auto [end, _] = std::to_chars(digits, digits + sizeof(digits), event.timestamp_ns);
        buf.append(digits, end);
    }

    buf += ",\"pid\":";
    {
        char digits[12];
        const auto [end, _] = std::to_chars(digits, digits + sizeof(digits), event.pid);
        buf.append(digits, end);
    }

    buf += ",\"tid\":";
    {
        char digits[12];
        const auto [end, _] = std::to_chars(digits, digits + sizeof(digits), event.tid);
        buf.append(digits, end);
    }

    buf += ",\"job_id\":";
    AppendQuotedEscaped(buf, event.job_id);

    buf += ",\"run_id\":";
    AppendQuotedEscaped(buf, event.run_id);

    buf += ",\"event_type\":";
    AppendQuotedEscaped(buf, event.event_type);

    buf += ",\"event_source\":";
    AppendQuotedEscaped(buf, event.event_source);

    AppendOptionalInt(buf, "gpu_id", event.gpu_id);
    AppendOptionalInt(buf, "step_id", event.step_id);
    AppendOptionalString(buf, "span_id", event.span_id);
    AppendOptionalString(buf, "parent_span_id", event.parent_span_id);
    AppendOptionalInt64(buf, "duration_ns", event.duration_ns);

    buf += ",\"level\":";
    AppendQuotedEscaped(buf, event.level);

    buf += ",\"payload\":{";
    bool first = true;
    for (const auto& [key, value] : event.payload) {
        if (!first) buf += ',';
        first = false;
        AppendQuotedEscaped(buf, key);
        buf += ':';
        AppendQuotedEscaped(buf, value);
    }
    buf += "}}";
}

}  // namespace

TraceWriter::TraceWriter(std::filesystem::path output_path)
    : output_path_(std::move(output_path)) {
    const auto parent = output_path_.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    out_.open(output_path_, std::ios::app);
    // Install the 64 KB I/O buffer — amortises write syscalls across the batch.
    out_.rdbuf()->pubsetbuf(io_buf_.data(), static_cast<std::streamsize>(kIoBufSize));
    line_buf_.reserve(512);
}

void TraceWriter::WriteBatch(const std::pmr::vector<TelemetryEvent>& events) {
    if (events.empty() || !out_.is_open()) return;
    for (const auto& event : events) {
        SerializeEventInto(event, line_buf_);
        out_ << line_buf_ << '\n';
    }
    out_.flush();
}

void TraceWriter::Flush() {
    if (out_.is_open()) out_.flush();
}

const std::filesystem::path& TraceWriter::output_path() const {
    return output_path_;
}

}  // namespace autopilot::telemetry
