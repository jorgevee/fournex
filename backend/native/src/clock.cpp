#include "clock.h"

#include <atomic>
#include <charconv>
#include <chrono>

namespace autopilot::telemetry {
namespace {
std::atomic<std::uint64_t> g_event_counter{0};
}

std::int64_t ClockSync::NowNs() const {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

std::string ClockSync::MakeEventId(std::string_view prefix) {
    const auto n = g_event_counter.fetch_add(1, std::memory_order_relaxed);
    char digits[20];
    const auto [end, _] = std::to_chars(digits, digits + sizeof(digits), n);
    std::string id;
    id.reserve(prefix.size() + 1 + static_cast<std::size_t>(end - digits));
    id.append(prefix);
    id += '-';
    id.append(digits, end);
    return id;
}

}  // namespace autopilot::telemetry
