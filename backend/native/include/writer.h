#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <memory_resource>
#include <string>
#include <vector>

#include "telemetry_engine.h"

namespace autopilot::telemetry {

class TraceWriter {
  public:
    explicit TraceWriter(std::filesystem::path output_path);

    void WriteBatch(const std::pmr::vector<TelemetryEvent>& events);
    void Flush();

    [[nodiscard]] const std::filesystem::path& output_path() const;

  private:
    std::filesystem::path output_path_;
    std::ofstream out_;
    // 64 KB I/O buffer — one syscall per batch instead of one per event.
    static constexpr std::size_t kIoBufSize = 64 * 1024;
    std::array<char, kIoBufSize> io_buf_;
    // Reused across WriteBatch calls; grows once and stays allocated.
    std::string line_buf_;
};

}  // namespace autopilot::telemetry
