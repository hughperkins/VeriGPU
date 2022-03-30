#!/bin/bash
# calculates delay propagation per cycle for proc.sv, and any child modules

set -ex
set -o pipefail

python verigpu/timing.py --in-verilog src/assert_ignore.sv src/const.sv src/op_const.sv \
    src/float/float_params.sv src/float/float_add_pipeline.sv \
    src/int/chunked_add_task.sv src/int/chunked_sub_task.sv \
    src/generated/mul_pipeline_cycle_24bit_2bpc.sv src/float/float_mul_pipeline.sv \
    src/generated/mul_pipeline_cycle_32bit_2bpc.sv src/int/mul_pipeline_32bit.sv \
    src/assert_ignore.sv src/int/int_div_regfile.sv \
    src/proc.sv --top-module proc
