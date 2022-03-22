#!/bin/bash

# check compiles ok with verilator
# - acts as a type of lint

verilator -sv --cc src/op_const.sv src/assert.sv src/const.sv \
    src/float_params.sv src/float_add_pipeline.sv \
    src/int_div_regfile.sv \
    src/proc.sv
