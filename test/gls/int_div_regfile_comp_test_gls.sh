#!/bin/bash

# run gate-level simulation on src/int_div_regfile.sv, as part of entire proc, comp etc

set -ex
set -o pipefail

prog=test_divu_modu

python verigpu/assembler.py --in-asm examples/${prog}.asm --out-hex build/build.hex
cat src/comp_driver.sv | sed -e "s/{PROG}/build/g" > build/comp_driver.sv

# first output gate-level netlists for int_div_regfile.sv
python verigpu/run_yosys.py --in-verilog src/const.sv src/mem_delayed_large.sv \
    src/assert_ignore.sv src/int/int_div_regfile.sv --top-module int_div_regfile >/dev/null

# now try running with proc, comp etc
iverilog -g2012 tech/osu018/osu018_stdcells.v build/netlist/6.v src/const.sv \
    src/assert_ignore.sv src/mem_delayed_large.sv \
    src/float/float_params.sv src/float/float_add_pipeline.sv \
    src/op_const.sv src/proc.sv src/mem_delayed.sv src/comp.sv src/comp_driver.sv

./a.out
