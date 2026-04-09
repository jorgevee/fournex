#include "telemetry_engine.h"

#include <stdexcept>
#include <utility>

#include "clock.h"
#include "cupti_tracer.h"
#include "event_schema.h"
#include "nvml_sampler.h"
#include "writer.h"

namespace autopilot::telemetry {

TelemetryEngine::TelemetryEngine() = default;

TelemetryEngine::~TelemetryEngine() {
    Shutdown();
}

void TelemetryEngine::Initialize(const TelemetryEngineOptions& options) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (initialized_) {
        return;
    }

    options_ = options;
    buffer_ = std::make_unique<EventBuffer>();
    writer_ = std::make_unique<TraceWriter>(options_.output_path);
    clock_ = std::make_unique<ClockSync>();
    sampler_ = std::make_unique<NVMLSampler>(
        options_.job_id, options_.run_id, options_.sample_interval_ms, clock_.get(),
        [this](TelemetryEvent event) { Emit(std::move(event)); });
    cupti_tracer_ = std::make_unique<CuptiTracer>(options_, clock_.get(), buffer_.get());

    initialized_ = true;

    Emit(TelemetryEvent{
        std::string(kSchemaVersion),
        clock_->MakeEventId("system"),
        clock_->NowNs(),
        0,
        0,
        options_.job_id,
        options_.run_id,
        "system_info",
        "native_engine",
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        "info",
        {
            {"hostname", "unknown"},
            {"platform", "windows"},
            {"gpu_count", "0"},
            {"cupti_enabled", options_.enable_cupti ? "true" : "false"},
            {"cupti_debug_mode", options_.cupti_debug_mode ? "true" : "false"},
        },
    });

    sampler_->Start();
    if (cupti_tracer_) {
        cupti_tracer_->Start();
    }
}

void TelemetryEngine::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!initialized_) {
        return;
    }

    if (sampler_) {
        sampler_->Stop();
    }
    if (cupti_tracer_) {
        cupti_tracer_->Stop();
    }
    Flush();

    cupti_tracer_.reset();
    sampler_.reset();
    writer_.reset();
    buffer_.reset();
    clock_.reset();
    initialized_ = false;
}

void TelemetryEngine::Emit(TelemetryEvent event) {
    if (!initialized_) {
        throw std::runtime_error("telemetry engine is not initialized");
    }
    buffer_->Push(std::move(event));
}

void TelemetryEngine::Flush() {
    if (!initialized_ || !writer_ || !buffer_) {
        return;
    }
    writer_->WriteBatch(buffer_->Drain());
    writer_->Flush();
}

bool TelemetryEngine::IsInitialized() const {
    return initialized_;
}

}  // namespace autopilot::telemetry
