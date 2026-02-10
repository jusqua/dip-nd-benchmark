cmake_minimum_required(VERSION 3.25)

project(benchmark LANGUAGES CXX)

set(SHARED_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../shared/include)
set(SHARED_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../shared/src/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../shared/include/utils.hpp)
set(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark.cpp)

if(CMAKE_CXX_COMPILER MATCHES "acpp")
    find_package(visiongl CONFIG REQUIRED)
    find_package(AdaptiveCpp CONFIG REQUIRED PATHS)
    add_executable(${PROJECT_NAME} ${SOURCE} ${SHARED_SOURCES})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${AdaptiveCpp_LIBS} visiongl::visiongl)
    set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)
    target_include_directories(${PROJECT_NAME} PRIVATE ${SHARED_INCLUDE})
    add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${SOURCE})
endif()
