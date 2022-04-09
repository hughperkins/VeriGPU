#!/bin/bash

# check compiles ok with verilator
# - acts as a type of lint

verilator -sv --cc src/op_const.sv src/assert.sv src/const.sv \
    src/float/float_params.sv src/float/float_add_pipeline.sv \
    src/int/chunked_add_task.sv src/int/chunked_sub_task.sv \
    src/generated/mul_pipeline_cycle_24bit_2bpc.sv src/float/float_mul_pipeline.sv \
    src/int/mul_pipeline_32bit.sv src/generated/mul_pipeline_cycle_32bit_2bpc.sv \
    src/int/int_div_regfile.sv \
    src/core.sv
