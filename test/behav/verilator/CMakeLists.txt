cmake_minimum_required(VERSION 3.16.3)

project(verilator1)
find_package(verilator HINTS $ENV{VERILATOR_ROOT})
# set(BASE ${CMAKE_SOURCE_DIR}/..)
set(BASE ${CMAKE_SOURCE_DIR}/../../..)
set(SRC ${BASE}/src)
set(BEHAV ${CMAKE_SOURCE_DIR}/..)

add_executable(single_core_mounted ${CMAKE_SOURE_DIR}verilator_driver.cpp ${VERILATOR_ROOT}/include/verilated.cpp)
verilate(single_core_mounted SOURCES ${SRC}/assert.sv ${SRC}/const.sv ${SRC}/op_const.sv
    ${SRC}/int/chunked_add_task.sv ${SRC}/int/chunked_sub_task.sv
    ${SRC}/float/float_params.sv ${SRC}/float/float_mul_pipeline.sv
    ${SRC}/float/float_add_pipeline.sv
    ${SRC}/int/mul_pipeline_32bit.sv
    ${SRC}/generated/mul_pipeline_cycle_24bit_2bpc.sv
    ${SRC}/generated/mul_pipeline_cycle_32bit_2bpc.sv
    ${SRC}/int/int_div_regfile.sv
    ${SRC}/core.sv
    ${SRC}/mem_large.sv ${SRC}/global_mem_controller.sv
    ${BEHAV}/single_core_mounted.sv
    TOP_MODULE single_core_mounted
    PREFIX single_core_mounted
    VERILATOR_ARGS +1800-2017ext+sv --x-initial unique --x-assign unique
)
option(BUILD_NETLIST "build netlist" OFF)
if(BUILD_NETLIST)
add_executable(single_core_netlist ../verilator_driver.cpp ${VERILATOR_ROOT}/include/verilated.cpp)
verilate(single_core_netlist SOURCES ${SRC}/assert.sv ${SRC}/const.sv ${SRC}/op_const.sv
    ${BASE}/tech/osu018/osu018_stdcells-verilator.sv
    ${BASE}/build/netlist/6.v
    ${SRC}/mem_large.sv ${SRC}/global_mem_controller.sv
    ${BEHAV}/single_core_mounted.sv
    TOP_MODULE single_core_mounted
    PREFIX single_core_netlist
    VERILATOR_ARGS +1800-2017ext+sv --x-initial unique --x-assign unique
)
endif()
