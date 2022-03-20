#!/bin/bash

# calculates delay propagation per cycle for comp.sv, and any child modules
# this includes memory
# so, SLOOWWWW!

# (Doesn't work yet; need to fix something in timing.py to handle this more
# complex netlist)

python toy_proc/timing.py --in-verilog src/const.sv src/op_const.sv \
    src/assert_ignore.sv \
    src/int_div_regfile.sv \
    src/proc.sv src/mem_delayed_small.sv src/mem_delayed.sv src/comp.sv \
    --top-module comp
