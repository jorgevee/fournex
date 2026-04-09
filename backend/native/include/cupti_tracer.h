#pragma once

#include <string>

#include "telemetry_engine.h"

namespace autopilot::telemetry {

class ClockSync;

class CuptiTracer {
  public:
    CuptiTracer(const TelemetryEngineOptions& options, ClockSync* clock, EventBuffer* buffer);

    void Start();
    void Stop();

    [[nodiscard]] bool available() const;
    [[nodiscard]] bool enabled() const;

  private:
    void EmitAvailabilityEvent(const std::string& code, const std::string& message) const;

    TelemetryEngineOptions options_;
    ClockSync* clock_;
    EventBuffer* buffer_;
    bool started_ = false;
};

}  // namespace autopilot::telemetry
