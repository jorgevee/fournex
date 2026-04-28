#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory_resource>
#include <string>
#include <thread>

#include "telemetry_engine.h"

namespace autopilot::telemetry {

class ClockSync;

class NVMLSampler {
  public:
    using EventSink = std::function<void(TelemetryEvent)>;

    NVMLSampler(std::string job_id, std::string run_id, int sample_interval_ms,
                ClockSync* clock, std::pmr::polymorphic_allocator<> alloc,
                EventSink sink);
    ~NVMLSampler();

    void Start();
    void Stop();

  private:
    void Run();
    TelemetryEvent MakeSampleEvent();

    std::string job_id_;
    std::string run_id_;
    std::chrono::milliseconds sample_interval_;
    ClockSync* clock_;
    std::pmr::polymorphic_allocator<> alloc_;
    EventSink sink_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::atomic<std::uint64_t> sample_index_{0};
};

}  // namespace autopilot::telemetry
