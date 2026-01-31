set(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark.cpp)

if(CMAKE_CXX_COMPILER MATCHES "acpp")
    find_package(visiongl CONFIG REQUIRED)
    find_package(AdaptiveCpp CONFIG REQUIRED PATHS)
    add_executable(benchmark_acpp_sycl ${SOURCE} ${SHARED_SOURCES})
    target_link_libraries(benchmark_acpp_sycl PRIVATE ${AdaptiveCpp_LIBS} visiongl::visiongl)
    add_sycl_to_target(TARGET benchmark_acpp_sycl SOURCES ${SOURCE})
endif()
