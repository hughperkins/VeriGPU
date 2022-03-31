#!/bin/bash

verilator -sv --cc src/op_const.sv src/assert.sv src/const.sv \
    src/float/float_params.sv src/float/float_add_pipeline.sv \
    src/int/chunked_add_task.sv src/int/chunked_sub_task.sv \
    src/generated/mul_pipeline_cycle_24bit_2bpc.sv src/float/float_mul_pipeline.sv \
    src/int/mul_pipeline_32bit.sv src/generated/mul_pipeline_cycle_32bit_2bpc.sv \
    src/int/int_div_regfile.sv \
    src/proc.sv \
    src/mem_delayed_small.sv src/mem_delayed.sv \
    src/comp.sv
