#include "nvml_sampler.h"

#include <algorithm>
#include <charconv>
#include <memory_resource>
#include <utility>

#include "clock.h"
#include "event_schema.h"

namespace autopilot::telemetry {

NVMLSampler::NVMLSampler(std::string job_id, std::string run_id, int sample_interval_ms,
                         ClockSync* clock, std::pmr::polymorphic_allocator<> alloc,
                         EventSink sink)
    : job_id_(std::move(job_id)),
      run_id_(std::move(run_id)),
      sample_interval_(sample_interval_ms),
      clock_(clock),
      alloc_(alloc),
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
    const auto sample_number = sample_index_.fetch_add(1, std::memory_order_relaxed);
    // Build the event-id prefix once using to_chars — avoids two temporary
    // strings that the old "prefix + to_string" pattern created.
    char prefix_buf[32];
    auto* p = prefix_buf;
    const std::string_view base = "gpu-sample-";
    std::copy(base.begin(), base.end(), p);
    p += base.size();
    p = std::to_chars(p, prefix_buf + sizeof(prefix_buf), sample_number).ptr;
    const std::string_view prefix(prefix_buf, static_cast<std::size_t>(p - prefix_buf));
    PayloadMap payload(alloc_);
    payload.emplace_back("utilization_gpu_pct", "0");
    payload.emplace_back("utilization_mem_pct", "0");
    payload.emplace_back("memory_used_bytes", "0");
    payload.emplace_back("memory_total_bytes", "0");
    TelemetryEvent ev(alloc_);
    ev.schema_version = kSchemaVersion;
    ev.event_id       = clock_->MakeEventId(prefix);
    ev.timestamp_ns   = clock_->NowNs();
    ev.gpu_id         = 0;
    ev.job_id         = job_id_;
    ev.run_id         = run_id_;
    ev.event_type     = "gpu_sample";
    ev.event_source   = "native_engine";
    ev.level          = "info";
    ev.payload        = std::move(payload);
    return ev;
}

}  // namespace autopilot::telemetry
