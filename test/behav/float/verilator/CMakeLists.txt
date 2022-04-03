
cmake_minimum_required(VERSION 3.16.3)

project(float_mul_pipeline_test_vtor)
find_package(verilator HINTS $ENV{VERILATOR_ROOT})

set(TST ${CMAKE_SOURCE_DIR}/../../../../test)
set(SRC ${CMAKE_SOURCE_DIR}/../../../../src)

add_executable(float_mul_pipeline_test_vtor
    ${CMAKE_SOURCE_DIR}/float_mul_pipeline_test_vtor.cpp
    ${CMAKE_SOURCE_DIR}/verilator_test_runner.cpp
    ${VERILATOR_ROOT}/include/verilated.cpp
)
verilate(float_mul_pipeline_test_vtor SOURCES ${SRC}/assert.sv ${SRC}/const.sv ${SRC}/op_const.sv
    ${SRC}/int/chunked_add_task.sv ${SRC}/int/chunked_sub_task.sv
    ${SRC}/float/float_params.sv ${SRC}/float/float_mul_pipeline.sv
    ${SRC}/generated/mul_pipeline_cycle_24bit_2bpc.sv
    ${TST}/behav/float/verilator/float_mul_pipeline_test_vtor.sv
    ${TST}/lib/float_test_funcs.sv
    TOP_MODULE float_mul_pipeline_test_vtor
    PREFIX float_mul_pipeline_test_vtor
    VERILATOR_ARGS +1800-2017ext+sv --x-initial unique --x-assign unique --assert
)

add_executable(float_add_pipeline_test_vtor
    ${CMAKE_SOURCE_DIR}/float_add_pipeline_test_vtor.cpp
    ${CMAKE_SOURCE_DIR}/verilator_test_runner.cpp
    ${VERILATOR_ROOT}/include/verilated.cpp
)
verilate(float_add_pipeline_test_vtor SOURCES ${SRC}/assert.sv ${SRC}/const.sv ${SRC}/op_const.sv
    ${SRC}/int/chunked_add_task.sv ${SRC}/int/chunked_sub_task.sv
    ${SRC}/float/float_params.sv ${SRC}/float/float_add_pipeline.sv
    ${TST}/behav/float/verilator/float_add_pipeline_test_vtor.sv
    ${TST}/lib/float_test_funcs.sv
    TOP_MODULE float_add_pipeline_test_vtor
    PREFIX float_add_pipeline_test_vtor
    VERILATOR_ARGS +1800-2017ext+sv --x-initial unique --x-assign unique --assert
)
