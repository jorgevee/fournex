#include "telemetry_engine.h"

#include <utility>

namespace autopilot::telemetry {

void EventBuffer::Push(TelemetryEvent event) {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.push_back(std::move(event));
}

std::vector<TelemetryEvent> EventBuffer::Drain() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<TelemetryEvent> drained;
    drained.swap(events_);
    return drained;
}

std::size_t EventBuffer::Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_.size();
}

}  // namespace autopilot::telemetry
