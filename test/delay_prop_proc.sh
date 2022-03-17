#!/bin/bash
# calculates delay propagation per cycle for proc.sv, and any child modules

python toy_proc/timing.py --in-verilog src/const.sv src/op_const.sv src/int_div_regfile.sv src/reg_file.sv src/proc.sv --top-module proc
