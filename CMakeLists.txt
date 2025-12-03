set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_minimum_required(VERSION 3.25)

project(benchmark LANGUAGES CXX)
set(SOURCES main.cpp main.hpp sycl.cpp utils.hpp)

find_package(AdaptiveCpp CONFIG REQUIRED)
find_package(visiongl CONFIG REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 26)
target_link_libraries(${PROJECT_NAME} PRIVATE ${AdaptiveCpp_LIBS} visiongl::visiongl)
add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${SOURCES})
