/*
 *  Per-kernel cycle counter using __rdtsc / __rdtscp.
 *
 * Compile with -DCYCLE_COUNT to activate. Without it define the
 * CYCLE_SCOPE macro is a no op so build is not affected.
 *
 * To use inside a kernel:
 *     #include "_cycle_counter.hpp"
 *     void some_kernel(...) {
 *         CYCLE_SCOPE("equi_linear");
 *         // ... existing kernel body ...
 *     }
 *
 * To read the accumulator from Python use:
 *     module.cycle_counter_get()  -> dict {name: (cycles, calls)}
 *     module.cycle_counter_reset()
 */
#pragma once

#ifdef CYCLE_COUNT

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#if defined(_MSC_VER)
#  include <intrin.h>
#else
#  include <x86intrin.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace cycle_count {

struct Entry {
    std::atomic<uint64_t> cycles{0};
    std::atomic<uint64_t> calls{0};
};

inline std::map<std::string, Entry>& registry() {
    static std::map<std::string, Entry> r;
    return r;
}

inline std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

inline Entry& get_entry(const char* name) {
    auto& reg = registry();
    auto it = reg.find(name);
    if (it != reg.end()) return it->second;
    std::lock_guard<std::mutex> lock(registry_mutex());
    return reg[name];
}

inline uint64_t rdtsc_start() {
    _mm_lfence();
    uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}

inline uint64_t rdtsc_end() {
    unsigned aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

class Scope {
public:
    explicit Scope(const char* name)
      : entry_(get_entry(name)), start_(rdtsc_start()) {}

    ~Scope() {
        uint64_t end = rdtsc_end();
        entry_.cycles.fetch_add(end - start_, std::memory_order_relaxed);
        entry_.calls.fetch_add(1, std::memory_order_relaxed);
    }

private:
    Entry& entry_;
    uint64_t start_;
};

}  // namespace cycle_count

#define CYCLE_SCOPE(name) ::cycle_count::Scope _cycle_scope_obj_(name)

#define BIND_CYCLE_COUNTER(m)                                                 \
    (m).def("cycle_counter_get", []() {                                       \
        std::map<std::string, std::pair<uint64_t, uint64_t>> out;             \
        for (auto& kv : ::cycle_count::registry()) {                            \
            out[kv.first] = {                                                 \
                kv.second.cycles.load(std::memory_order_relaxed),             \
                kv.second.calls.load(std::memory_order_relaxed)};             \
        }                                                                     \
        return out;                                                           \
    });                                                                       \
    (m).def("cycle_counter_reset", []() {                                     \
        for (auto& kv : ::cycle_count::registry()) {                            \
            kv.second.cycles.store(0, std::memory_order_relaxed);             \
            kv.second.calls.store(0, std::memory_order_relaxed);              \
        }                                                                     \
    });                                                                       \
    (m).def("cycle_counter_enabled", []() { return true; })

#else  /* !CYCLE_COUNT */

#define CYCLE_SCOPE(name) ((void)0)
#define BIND_CYCLE_COUNTER(m)                                                 \
    (m).def("cycle_counter_enabled", []() { return false; })

#endif  /* CYCLE_COUNT */
