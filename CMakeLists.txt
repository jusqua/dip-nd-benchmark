# WARNING: HALF-BAKED CMAKE CONFIG FILES AHEAD!
cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(benchmark LANGUAGES CXX)

option(DIP_ND_BENCHMARK_ENABLE_VISIONGL "Enable VisionGL benchmarks" ON)
option(DIP_ND_BENCHMARK_ENABLE_CUDA "Enable CUDA benchmarks" ON)
option(DIP_ND_BENCHMARK_ENABLE_SYCL "Enable SYCL benchmarks" ON)

set(SHARED_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/_shared/src/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/_shared/include/_shared/utils.hpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/_shared/include)
add_compile_options(-Wall -Wextra -std=c++20 -O3)

if(DIP_ND_BENCHMARK_ENABLE_CUDA STREQUAL ON AND (CMAKE_CXX_COMPILER MATCHES "acpp" OR CMAKE_CUDA_COMPILER MATCHES "nvcc"))
    add_subdirectory(src/cuda cuda)
endif()

if(DIP_ND_BENCHMARK_ENABLE_SYCL STREQUAL ON AND (CMAKE_CXX_COMPILER MATCHES "acpp" OR CMAKE_CXX_COMPILER MATCHES "icpx"))
    add_subdirectory(src/sycl sycl)
endif()

if(DIP_ND_BENCHMARK_ENABLE_VISIONGL STREQUAL ON)
    add_subdirectory(src/visiongl visiongl)
endif()
