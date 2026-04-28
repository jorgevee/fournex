#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace autopilot::telemetry {

// Payload storage using pmr containers so EventBuffer's synchronized_pool_resource
// backs all payload string allocations — avoids per-event heap round-trips.
using PayloadMap = std::pmr::vector<std::pair<std::pmr::string, std::pmr::string>>;

struct TelemetryEvent {
    // Declares allocator support so pmr::vector<TelemetryEvent> propagates its
    // allocator into PayloadMap via uses-allocator construction.
    using allocator_type = std::pmr::polymorphic_allocator<>;

    // run-scoped fields stay std::string (copied from RunStatics once at init).
    std::string schema_version;
    std::string event_id;
    std::int64_t timestamp_ns = 0;
    int pid = 0;
    int tid = 0;
    std::string job_id;
    std::string run_id;
    std::string event_type;
    std::string event_source;
    std::optional<int> gpu_id;
    std::optional<int> step_id;
    std::optional<std::string> span_id;
    std::optional<std::string> parent_span_id;
    std::optional<std::int64_t> duration_ns;
    std::string level;
    PayloadMap payload;

    TelemetryEvent() = default;

    // Allocator-extended default ctor — used when pmr::vector<TelemetryEvent>
    // default-constructs elements with the container's allocator.
    explicit TelemetryEvent(allocator_type alloc) : payload(alloc) {}

    TelemetryEvent(const TelemetryEvent&) = default;
    TelemetryEvent& operator=(const TelemetryEvent&) = default;
    TelemetryEvent(TelemetryEvent&&) = default;
    TelemetryEvent& operator=(TelemetryEvent&&) = default;

    // Allocator-extended move ctor — rebinds PayloadMap to the target allocator
    // (O(1) move if allocators compare equal, element-wise otherwise).
    TelemetryEvent(TelemetryEvent&& other, allocator_type alloc)
        : schema_version(std::move(other.schema_version)),
          event_id(std::move(other.event_id)),
          timestamp_ns(other.timestamp_ns),
          pid(other.pid),
          tid(other.tid),
          job_id(std::move(other.job_id)),
          run_id(std::move(other.run_id)),
          event_type(std::move(other.event_type)),
          event_source(std::move(other.event_source)),
          gpu_id(other.gpu_id),
          step_id(other.step_id),
          span_id(std::move(other.span_id)),
          parent_span_id(std::move(other.parent_span_id)),
          duration_ns(other.duration_ns),
          level(std::move(other.level)),
          payload(std::move(other.payload), alloc) {}

    TelemetryEvent(const TelemetryEvent& other, allocator_type alloc)
        : schema_version(other.schema_version),
          event_id(other.event_id),
          timestamp_ns(other.timestamp_ns),
          pid(other.pid),
          tid(other.tid),
          job_id(other.job_id),
          run_id(other.run_id),
          event_type(other.event_type),
          event_source(other.event_source),
          gpu_id(other.gpu_id),
          step_id(other.step_id),
          span_id(other.span_id),
          parent_span_id(other.parent_span_id),
          duration_ns(other.duration_ns),
          level(other.level),
          payload(other.payload, alloc) {}
};

// EventBuffer is defined in event_buffer.h (includes the lock-free ring buffer).
// Forward-declared here so telemetry_engine.h stays self-contained for headers
// that only need TelemetryEvent and TelemetryEngine.
class EventBuffer;

struct TelemetryEngineOptions {
    std::string output_path = "trace.jsonl";
    std::string job_id = "unknown-job";
    std::string run_id = "unknown-run";
    int sample_interval_ms = 1000;
    bool enable_cupti = false;
    bool cupti_debug_mode = false;
};

class TraceWriter;
class NVMLSampler;
class ClockSync;
class CuptiTracer;

class TelemetryEngine {
  public:
    TelemetryEngine();
    ~TelemetryEngine();

    void Initialize(const TelemetryEngineOptions& options);
    void Shutdown();
    void Emit(TelemetryEvent event);
    void Flush();

    [[nodiscard]] bool IsInitialized() const;
    // Returns the pool allocator owned by EventBuffer — use this to construct
    // TelemetryEvents whose PayloadMaps will move O(1) into the ring buffer.
    [[nodiscard]] std::pmr::polymorphic_allocator<> GetAllocator();

  private:
    std::mutex mutex_;
    TelemetryEngineOptions options_;
    std::unique_ptr<EventBuffer> buffer_;
    std::unique_ptr<TraceWriter> writer_;
    std::unique_ptr<NVMLSampler> sampler_;
    std::unique_ptr<ClockSync> clock_;
    std::unique_ptr<CuptiTracer> cupti_tracer_;
    std::atomic<bool> initialized_{false};
};

}  // namespace autopilot::telemetry
