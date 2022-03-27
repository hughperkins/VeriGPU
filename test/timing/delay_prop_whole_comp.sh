#!/bin/bash

# calculates delay propagation per cycle for comp.sv, and any child modules
# this includes memory
# so, SLOOWWWW!

set -ex
set -o pipefail

python verigpu/timing.py --in-verilog src/const.sv src/op_const.sv \
    src/assert_ignore.sv \
    src/float/float_params.sv src/float/float_add_pipeline.sv \
    src/generated/mul_pipeline_cycle_32bit_2bpc.sv src/int/mul_pipeline_32bit.sv \
    src/int/int_div_regfile.sv \
    src/proc.sv src/mem_delayed_small.sv src/mem_delayed.sv src/comp.sv \
    --top-module comp
