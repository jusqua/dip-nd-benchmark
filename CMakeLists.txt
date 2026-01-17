cmake_minimum_required(VERSION 3.25)

if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

project(benchmark LANGUAGES CXX)

if(NOT DIP_ND_BENCHMARK_ENABLE_CUDA)
	if(CMAKE_CXX_COMPILER MATCHES "acpp")
		message(STATUS "Compiler capabilities detected, enabling CUDA benchmarks")
	else()
		message(STATUS "Compiler not capable of compiling CUDA benchmarks")
	endif()
	# TODO: Handle CUDA benchmarks with other compilers
endif()

if(NOT DIP_ND_BENCHMARK_ENABLE_SYCL)
	if(CMAKE_CXX_COMPILER MATCHES "acpp")
		message(STATUS "Compiler capabilities detected, enabling SYCL benchmarks")
	else()
		message(STATUS "Compiler not capable of compiling SYCL benchmarks")
	endif()
	# TODO: Handle SYCL benchmarks with Intel DPCPP
endif()

if(NOT DIP_ND_BENCHMARK_ENABLE_VISIONGL)
	message(STATUS "Enabling VisionGL benchmarks")
endif()

if(DIP_ND_BENCHMARK_ENABLE_CUDA STREQUAL ON)
	add_subdirectory(src/cuda)
endif()

if(DIP_ND_BENCHMARK_ENABLE_SYCL STREQUAL ON)
	add_subdirectory(src/sycl)
endif()

if(DIP_ND_BENCHMARK_ENABLE_VISIONGL STREQUAL ON)
	add_subdirectory(src/visiongl)
endif()
