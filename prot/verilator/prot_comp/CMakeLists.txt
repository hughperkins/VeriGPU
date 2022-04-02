cmake_minimum_required(VERSION 3.22)

project(verilator1)
find_package(verilator HINTS $ENV{VERILATOR_ROOT})
add_executable(verilator1 ../verilator_driver.cpp ${VERILATOR_ROOT}/include/verilated.cpp)
set(BASE "../../src")
verilate(verilator1 SOURCES ${BASE}/assert.sv ${BASE}/const.sv ${BASE}/op_const.sv
    ${BASE}/int/chunked_add_task.sv ${BASE}/int/chunked_sub_task.sv
    ${BASE}/float/float_params.sv ${BASE}/float/float_mul_pipeline.sv
    ${BASE}/float/float_add_pipeline.sv
    ${BASE}/int/mul_pipeline_32bit.sv
    ${BASE}/generated/mul_pipeline_cycle_24bit_2bpc.sv
    ${BASE}/generated/mul_pipeline_cycle_32bit_2bpc.sv
    ${BASE}/int/int_div_regfile.sv
    ${BASE}/proc.sv
    ${BASE}/mem_delayed_large.sv ${BASE}/mem_delayed.sv
    ${BASE}/comp.sv
    TOP_MODULE comp
    PREFIX comp
)
# target_link_libraries(verilator1 ${VERILATOR_ROOT}/include/verilated.cpp)
