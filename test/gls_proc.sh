#!/bin/bash

# run gate-level simulation on src/proc.sv

# we'll use the original sv for mem_delayed, comp and comp_driver, since mem_delayed takes
# forever to synthesize...

if [[ x$1 == x ]]; then {
    echo "Usage: $0 [progname]"
    exit 1
} fi

set -ex
set -o pipefail

python toy_proc/assembler.py --in-asm examples/$1.asm --out-hex build/build.hex
cat src/comp_driver.sv | sed -e "s/{PROG}/build/g" > build/comp_driver.sv

# python toy_proc/run_yosys.py --in-verilog src/op_const.sv src/const.sv src/int_div_regfile.sv src/proc.sv --top-module proc >/dev/null
iverilog -g2012 tech/osu018/osu018_stdcells.v build/netlist/6.v src/const.sv src/mem_delayed.sv \
    src/comp.sv build/comp_driver.sv
./a.out | tee build/out.txt
cat build/out.txt | grep '^out ' > build/out_only.txt

if diff build/out_only.txt examples/$1_expected.txt; then {
    echo SUCCESS
} else {
    echo FAIL
    exit 1
} fi
