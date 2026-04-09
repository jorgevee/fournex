#include "nvml_sampler.h"

#include <utility>

#include "clock.h"
#include "event_schema.h"

namespace autopilot::telemetry {

NVMLSampler::NVMLSampler(std::string job_id, std::string run_id, int sample_interval_ms,
                         ClockSync* clock, EventSink sink)
    : job_id_(std::move(job_id)),
      run_id_(std::move(run_id)),
      sample_interval_(sample_interval_ms),
      clock_(clock),
      sink_(std::move(sink)) {}

NVMLSampler::~NVMLSampler() {
    Stop();
}

void NVMLSampler::Start() {
    if (running_.exchange(true)) {
        return;
    }
    worker_ = std::thread([this] { Run(); });
}

void NVMLSampler::Stop() {
    if (!running_.exchange(false)) {
        return;
    }
    if (worker_.joinable()) {
        worker_.join();
    }
}

void NVMLSampler::Run() {
    while (running_) {
        sink_(MakeSampleEvent());
        std::this_thread::sleep_for(sample_interval_);
    }
}

TelemetryEvent NVMLSampler::MakeSampleEvent() {
    const auto sample_number = sample_index_.fetch_add(1);
    return TelemetryEvent{
        std::string(kSchemaVersion),
        clock_->MakeEventId("gpu-sample-" + std::to_string(sample_number)),
        clock_->NowNs(),
        0,
        0,
        job_id_,
        run_id_,
        "gpu_sample",
        "native_engine",
        0,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        "info",
        {
            {"utilization_gpu_pct", "0"},
            {"utilization_mem_pct", "0"},
            {"memory_used_bytes", "0"},
            {"memory_total_bytes", "0"},
        },
    };
}

}  // namespace autopilot::telemetry
