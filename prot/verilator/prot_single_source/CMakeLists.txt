cmake_minimum_required(VERSION 3.22)

project(verilator1)
find_package(verilator HINTS $ENV{VERILATOR_ROOT})
set(BASE "${CMAKE_SOURCE_DIR}/../../../src")

add_executable(controller
    ${VERILATOR_ROOT}/include/verilated.cpp
    ${CMAKE_SOURCE_DIR}/gpu_runtime.cpp
    ${CMAKE_SOURCE_DIR}/my_gpu_test_client.cpp
)
verilate(controller
    SOURCES
        ${CMAKE_SOURCE_DIR}/controller.sv
    TOP_MODULE controller
    PREFIX controller
)
