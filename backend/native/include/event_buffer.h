#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "telemetry_engine.h"

namespace autopilot::telemetry {

// Lock-free MPMC ring buffer for TelemetryEvent.
//
// Design:
//   - Power-of-two capacity; index masking replaces modulo.
//   - Each slot carries a sequence number. Producers CAS the sequence from
//     (index) to (index+1) to claim a slot; consumers CAS from (index+1) to
//     (index + Capacity) to reclaim it.
//   - Aligns head_ and tail_ to separate cache lines to avoid false sharing
//     between producer and consumer threads.
//   - On push failure (full buffer) the event is silently dropped to avoid
//     stalling the training loop. The caller can detect this via Size().
//
// Thread safety: Push() is safe to call from multiple producer threads
// simultaneously. Drain() must be called from a single consumer at a time
// (the flush path in TelemetryEngine is already serialised by mutex_).
template <std::size_t Capacity>
class RingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

    struct alignas(64) Slot {
        std::atomic<std::size_t> sequence{0};
        TelemetryEvent event;
    };

  public:
    RingBuffer() {
        for (std::size_t i = 0; i < Capacity; ++i) {
            slots_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    // Returns false if the buffer is full (event dropped).
    bool TryPush(TelemetryEvent event) {
        std::size_t pos = head_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = slots_[pos & (Capacity - 1)];
            const std::size_t seq = slot.sequence.load(std::memory_order_acquire);
            const std::intptr_t diff = static_cast<std::intptr_t>(seq) -
                                       static_cast<std::intptr_t>(pos);
            if (diff == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1,
                                                std::memory_order_relaxed)) {
                    slot.event = std::move(event);
                    slot.sequence.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;  // buffer full
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }
    }

    // Drains all available events into out. Single-consumer only.
    void DrainInto(std::pmr::vector<TelemetryEvent>& out) {
        std::size_t pos = tail_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& slot = slots_[pos & (Capacity - 1)];
            const std::size_t seq = slot.sequence.load(std::memory_order_acquire);
            const std::intptr_t diff = static_cast<std::intptr_t>(seq) -
                                       static_cast<std::intptr_t>(pos + 1);
            if (diff == 0) {
                tail_.store(pos + 1, std::memory_order_relaxed);
                out.push_back(std::move(slot.event));
                slot.sequence.store(pos + Capacity, std::memory_order_release);
                ++pos;
            } else {
                break;  // nothing more to consume
            }
        }
    }

    std::size_t ApproxSize() const {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        return (h >= t) ? (h - t) : 0;
    }

  private:
    Slot slots_[Capacity];
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

// EventBuffer wraps the ring buffer and exposes Push/Drain/Size.
// Owns a synchronized_pool_resource so PayloadMap allocations from multiple
// producer threads (Python emit + NVMLSampler) go through a single thread-safe
// pool rather than the process-global heap. The pool is declared before ring_
// so it outlives all ring buffer slots.
class EventBuffer {
    static constexpr std::size_t kCapacity = 65536;

  public:
    explicit EventBuffer(std::size_t /*initial_capacity_hint*/ = 4096) {}

    // Returns the pool allocator — pass to TelemetryEvent constructors so
    // PayloadMap strings are pool-backed and move O(1) into ring buffer slots.
    std::pmr::polymorphic_allocator<> GetAllocator() { return &pool_; }

    // Lock-free push from any thread. Returns false and drops the event if
    // the ring buffer is full (backpressure signal).
    bool Push(TelemetryEvent event) {
        return ring_.TryPush(std::move(event));
    }

    // Drain all pending events into a pmr::vector backed by mr (default: heap).
    // Pass a monotonic_buffer_resource from the flush scope to batch-free all
    // drained event allocations at once when the resource is destroyed.
    std::pmr::vector<TelemetryEvent> Drain(
        std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        std::pmr::vector<TelemetryEvent> out(mr);
        out.reserve(ring_.ApproxSize());
        ring_.DrainInto(out);
        return out;
    }

    std::size_t Size() const { return ring_.ApproxSize(); }

  private:
    // pool_ must be declared before ring_ — ring buffer slots hold TelemetryEvents
    // whose PayloadMaps may reference this pool. Destruction order: ring_ first,
    // then pool_ (reverse of declaration) which is the correct teardown sequence.
    std::pmr::synchronized_pool_resource pool_;
    RingBuffer<kCapacity> ring_;
};

}  // namespace autopilot::telemetry
