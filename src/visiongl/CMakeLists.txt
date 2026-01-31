find_package(visiongl CONFIG REQUIRED)
add_executable(benchmark_visiongl ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark.cpp ${SHARED_SOURCES})
target_link_libraries(benchmark_visiongl PRIVATE visiongl::visiongl)
