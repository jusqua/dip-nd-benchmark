cmake_minimum_required(VERSION 3.25)

if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

project(benchmark LANGUAGES CXX)
set(SOURCES main.cpp main.hpp utils.hpp)

find_package(AdaptiveCpp CONFIG REQUIRED)
find_package(visiongl CONFIG REQUIRED)

set(SYCL_SOURCES ${SOURCES} sycl.cpp)
add_executable(benchmark_sycl ${SYCL_SOURCES})
set_property(TARGET benchmark_sycl PROPERTY CXX_STANDARD 26)
target_link_libraries(benchmark_sycl PRIVATE ${AdaptiveCpp_LIBS} visiongl::visiongl)
add_sycl_to_target(TARGET benchmark_sycl SOURCES ${SYCL_SOURCES})

set(VISIONGL_SOURCES ${SOURCES} visiongl.cpp)
add_executable(benchmark_visiongl ${VISIONGL_SOURCES})
set_property(TARGET benchmark_visiongl PROPERTY CXX_STANDARD 26)
target_link_libraries(benchmark_visiongl PRIVATE visiongl::visiongl)
