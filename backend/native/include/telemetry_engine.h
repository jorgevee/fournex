#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace autopilot::telemetry {

struct TelemetryEvent {
    std::string schema_version;
    std::string event_id;
    std::int64_t timestamp_ns;
    int pid;
    int tid;
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
    std::unordered_map<std::string, std::string> payload;
};

class EventBuffer {
  public:
    void Push(TelemetryEvent event);
    std::vector<TelemetryEvent> Drain();
    std::size_t Size() const;

  private:
    mutable std::mutex mutex_;
    std::vector<TelemetryEvent> events_;
};

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
