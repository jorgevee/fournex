#include "clock.h"

#include <atomic>
#include <chrono>

namespace autopilot::telemetry {
namespace {
std::atomic<std::uint64_t> g_event_counter{0};
}

std::int64_t ClockSync::NowNs() const {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

std::string ClockSync::MakeEventId(const std::string& prefix) {
    const auto event_number = g_event_counter.fetch_add(1);
    return prefix + "-" + std::to_string(event_number);
}

}  // namespace autopilot::telemetry
