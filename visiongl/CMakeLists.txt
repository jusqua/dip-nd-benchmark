cmake_minimum_required(VERSION 3.25)

project(benchmark LANGUAGES CXX)

option(FORCE_BUFFER "Force buffer operations" ON)

set(SHARED_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../shared/include)
set(SHARED_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared/include/utils.hpp
)

set(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark.cpp)

find_package(visiongl CONFIG REQUIRED)
add_executable(${PROJECT_NAME} ${SOURCE} ${SHARED_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE visiongl::visiongl)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)
target_include_directories(${PROJECT_NAME} PRIVATE ${SHARED_INCLUDE})
if(FORCE_BUFFER)
    target_compile_definitions(${PROJECT_NAME} PUBLIC FORCE_BUFFER=true)
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC FORCE_BUFFER=false)
endif()
