cmake_minimum_required(VERSION 3.20)

project(autopilot_telemetry_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(autopilot_telemetry_native
    src/bindings.cpp
    src/clock.cpp
    src/cupti_tracer.cpp
    src/event_buffer.cpp
    src/nvml_sampler.cpp
    src/telemetry_engine.cpp
    src/writer.cpp
)

target_include_directories(autopilot_telemetry_native
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
