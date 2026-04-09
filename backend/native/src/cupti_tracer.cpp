#include "cupti_tracer.h"

#include <utility>

#include "clock.h"
#include "event_schema.h"

namespace autopilot::telemetry {
namespace {

constexpr bool kCuptiCompiledIn = false;

TelemetryEvent MakeWarningEvent(const TelemetryEngineOptions& options, ClockSync* clock,
                                const std::string& code, const std::string& message) {
    return TelemetryEvent{
        std::string(kSchemaVersion),
        clock->MakeEventId("cupti-warning"),
        clock->NowNs(),
        0,
        0,
        options.job_id,
        options.run_id,
        "warning_annotation",
        "native_engine",
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        "warning",
        {
            {"code", code},
            {"message", message},
        },
    };
}

}  // namespace

CuptiTracer::CuptiTracer(const TelemetryEngineOptions& options, ClockSync* clock, EventBuffer* buffer)
    : options_(options), clock_(clock), buffer_(buffer) {}

void CuptiTracer::Start() {
    if (started_ || !options_.enable_cupti) {
        return;
    }
    started_ = true;

    if (!available()) {
        EmitAvailabilityEvent("cupti_unavailable",
                              "CUPTI debug mode was requested but this build does not include CUPTI.");
        return;
    }

    EmitAvailabilityEvent("cupti_requested",
                          options_.cupti_debug_mode
                              ? "CUPTI debug mode was requested and would start in debug mode."
                              : "CUPTI collection was requested.");
}

void CuptiTracer::Stop() {
    if (!started_) {
        return;
    }
    started_ = false;
}

bool CuptiTracer::available() const {
    return kCuptiCompiledIn;
}

bool CuptiTracer::enabled() const {
    return started_ && options_.enable_cupti;
}

void CuptiTracer::EmitAvailabilityEvent(const std::string& code, const std::string& message) const {
    if (clock_ == nullptr || buffer_ == nullptr) {
        return;
    }
    buffer_->Push(MakeWarningEvent(options_, clock_, code, message));
}

}  // namespace autopilot::telemetry
