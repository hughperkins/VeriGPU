#!/bin/bash

# run gate-level simulation on src/int_div_regfile.sv, as part of entire proc, comp etc

set -ex
set -o pipefail

prog=prog22

python toy_proc/assembler.py --in-asm examples/${prog}.asm --out-hex build/build.hex
cat src/comp_driver.sv | sed -e "s/{PROG}/build/g" > build/comp_driver.sv

# first output gate-level netlists for int_div_regfile.sv
python toy_proc/run_yosys.py --in-verilog src/const.sv src/int_div_regfile.sv --top-module int_div_regfile >/dev/null

# now try running with proc, comp etc
iverilog -g2012 src/int_div_regfile.sv src/const.sv \
    src/op_const.sv src/proc.sv src/mem_delayed.sv src/comp.sv src/comp_driver.sv

./a.out
