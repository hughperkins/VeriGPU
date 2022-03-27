#!/bin/bash
# calculates delay propagation per cycle for proc.sv, and any child modules

set -ex
set -o pipefail

python toy_proc/timing.py --in-verilog src/assert_ignore.sv src/const.sv src/op_const.sv \
    src/float/float_params.sv src/float/float_add_pipeline.sv \
    src/assert_ignore.sv src/int/int_div_regfile.sv \
    src/reg_file.sv src/proc.sv --top-module proc
