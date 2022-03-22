#!/bin/bash

# run gate-level simulation on src/int_div_regfile.sv, as part of entire proc, comp etc

set -ex
set -o pipefail

prog=test_divu_modu

python toy_proc/assembler.py --in-asm examples/${prog}.asm --out-hex build/build.hex
cat src/comp_driver.sv | sed -e "s/{PROG}/build/g" > build/comp_driver.sv

# now try running with proc, comp etc
iverilog -g2012 src/const.sv src/op_const.sv src/mem_delayed_large.sv \
    src/assert.sv \
    src/float_params.sv src/float_add_pipeline.sv \
    src/int_div_regfile.sv src/proc.sv src/mem_delayed.sv src/comp.sv src/comp_driver.sv

./a.out
