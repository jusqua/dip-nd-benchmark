set(SOURCES "../_shared/main.cpp" "../_shared/utils.hpp" "benchmark.cpp")

find_package(visiongl CONFIG REQUIRED)
add_executable(benchmark_visiongl ${SOURCES})
set_property(TARGET benchmark_visiongl PROPERTY CXX_STANDARD 26)
target_link_libraries(benchmark_visiongl PRIVATE visiongl::visiongl)
