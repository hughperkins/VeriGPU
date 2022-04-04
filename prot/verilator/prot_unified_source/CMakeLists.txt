cmake_minimum_required(VERSION 3.22)

project(verilator1)
find_package(verilator HINTS $ENV{VERILATOR_ROOT})
set(BASE "${CMAKE_SOURCE_DIR}/../../../src")

add_executable(controller ${CMAKE_SOURCE_DIR}/verilator_driver.cpp ${VERILATOR_ROOT}/include/verilated.cpp)
verilate(controller
    SOURCES
        ${CMAKE_SOURCE_DIR}/controller.sv
    TOP_MODULE controller
    PREFIX controller
)
