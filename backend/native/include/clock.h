#pragma once

#include <cstdint>
#include <string>

namespace autopilot::telemetry {

class ClockSync {
  public:
    std::int64_t NowNs() const;
    std::string MakeEventId(const std::string& prefix);
};

}  // namespace autopilot::telemetry
