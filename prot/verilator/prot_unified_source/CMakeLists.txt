cmake_minimum_required(VERSION 3.22)

project(verilator1)
find_package(verilator HINTS $ENV{VERILATOR_ROOT})
add_executable(prot_unified_source ${CMAKE_SOURCE_DIR}/verilator_driver.cpp ${VERILATOR_ROOT}/include/verilated.cpp)
set(BASE "${CMAKE_SOURCE_DIR}/../../../src")
verilate(prot_unified_source SOURCES
    ${CMAKE_SOURCE_DIR}/controller.sv
    TOP_MODULE prot_unified_source
    PREFIX prot_unified_source
)
