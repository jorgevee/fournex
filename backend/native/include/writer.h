#pragma once

#include <filesystem>
#include <vector>

#include "telemetry_engine.h"

namespace autopilot::telemetry {

class TraceWriter {
  public:
    explicit TraceWriter(std::filesystem::path output_path);

    void WriteBatch(const std::vector<TelemetryEvent>& events);
    void Flush();

    [[nodiscard]] const std::filesystem::path& output_path() const;

  private:
    std::filesystem::path output_path_;
};

}  // namespace autopilot::telemetry
