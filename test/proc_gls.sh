#!/bin/bash

# run gate-level simulation on src/proc.sv

# we'll use the original sv for mem_delayed, comp and comp_driver, since mem_delayed takes
# forever to synthesize...

set -ex
set -o pipefail

progs="prog2 prog3 prog4 prog5 prog6 prog7 prog8 prog9 prog10 prog11 prog12 prog13 prog14 prog15 \
        prog16 prog17 prog18 prog19 prog20 prog21 prog22"

if [[ x$1 != x ]]; then {
    progs=$1
} fi

python toy_proc/run_yosys.py --in-verilog src/op_const.sv src/const.sv src/int_div_regfile.sv src/proc.sv \
    --top-module proc >/dev/null

for prog in ${progs}; do {
    python toy_proc/assembler.py --in-asm examples/${prog}.asm --out-hex build/prog.hex
    cat src/comp_driver.sv | sed -e "s/{PROG}/prog/g" > build/comp_driver.sv

    iverilog -g2012 tech/osu018/osu018_stdcells.v build/netlist/6.v src/const.sv src/mem_delayed.sv \
        src/comp.sv build/comp_driver.sv
    ./a.out | tee build/out.txt
    if  ! cat build/out.txt | grep '^out[ \.]' > build/out_only.txt; then {
        echo "grep failed"
        echo "" > build/out.txt
    } fi

    if diff build/out_only.txt examples/${prog}_expected.txt; then {
        echo SUCCESS
    } else {
        echo FAIL
        exit 1
    } fi
} done
