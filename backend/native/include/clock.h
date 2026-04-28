#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace autopilot::telemetry {

class ClockSync {
  public:
    std::int64_t NowNs() const;
    std::string MakeEventId(std::string_view prefix);
};

}  // namespace autopilot::telemetry
