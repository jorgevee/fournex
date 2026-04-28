#include "telemetry_engine.h"

#include <memory_resource>
#include <stdexcept>
#include <utility>

#include "clock.h"
#include "cupti_tracer.h"
#include "event_buffer.h"
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
        buffer_->GetAllocator(),
        [this](TelemetryEvent event) { Emit(std::move(event)); });
    cupti_tracer_ = std::make_unique<CuptiTracer>(options_, clock_.get(), buffer_.get());

    initialized_ = true;

    {
        auto alloc = buffer_->GetAllocator();
        PayloadMap payload(alloc);
        payload.emplace_back("hostname", "unknown");
        payload.emplace_back("platform", "windows");
        payload.emplace_back("gpu_count", "0");
        payload.emplace_back("cupti_enabled", options_.enable_cupti ? "true" : "false");
        payload.emplace_back("cupti_debug_mode", options_.cupti_debug_mode ? "true" : "false");
        TelemetryEvent ev(alloc);
        ev.schema_version = kSchemaVersion;
        ev.event_id       = clock_->MakeEventId("system");
        ev.timestamp_ns   = clock_->NowNs();
        ev.job_id         = options_.job_id;
        ev.run_id         = options_.run_id;
        ev.event_type     = "system_info";
        ev.event_source   = "native_engine";
        ev.level          = "info";
        ev.payload        = std::move(payload);
        Emit(std::move(ev));
    }

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
    // TryPush returns false when the ring buffer is full (65536 slots).
    // We drop silently rather than block — preserving training loop latency is
    // more important than capturing every event under extreme burst conditions.
    buffer_->Push(std::move(event));
}

void TelemetryEngine::Flush() {
    if (!initialized_ || !writer_ || !buffer_) {
        return;
    }
    // Stack-backed arena for the drain batch: all PayloadMap rebinds during
    // uses-allocator construction land here and are bulk-freed when mono
    // destructs — zero individual deallocations for the flush path.
    alignas(std::max_align_t) std::byte arena[64 * 1024];
    std::pmr::monotonic_buffer_resource mono(arena, sizeof(arena));
    writer_->WriteBatch(buffer_->Drain(&mono));
    writer_->Flush();
}

std::pmr::polymorphic_allocator<> TelemetryEngine::GetAllocator() {
    return buffer_ ? buffer_->GetAllocator()
                   : std::pmr::polymorphic_allocator<>{};
}

bool TelemetryEngine::IsInitialized() const {
    return initialized_;
}

}  // namespace autopilot::telemetry
