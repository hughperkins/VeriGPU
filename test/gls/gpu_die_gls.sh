#!/bin/bash

# run GLS on entire gpu_die.sv

# todo: we need to actually create a driver for this

# (following re-architecture from comp.sv/proc.sv into gpu_card.sv, gpu_die.sv, core.sv, ec)

set -ex
set -o pipefail

progs=$(ls -b examples/direct/*.asm)

if [[ x$1 != x ]]; then {
    progs=$1
} fi

if [[ -f build/netlist/4.v ]]; then {
    rm build/netlist/4.v
} fi

if [[ -f build/netlist/6.v ]]; then {
    rm build/netlist/6.v
} fi

python verigpu/run_yosys.py --in-verilog src/assert_ignore.sv \
    src/op_const.sv src/const.sv src/int/int_div_regfile.sv src/core.sv \
    src/float/float_params.sv src/float/float_add_pipeline.sv \
    src/int/chunked_add_task.sv src/int/chunked_sub_task.sv \
    src/mem_large.sv src/global_mem_controller.sv src/gpu_die.sv \
    --top-module gpu_die >/dev/null

for prog in ${progs}; do {
    python verigpu/assembler.py --in-asm examples/direct/${prog}.asm --out-hex build/prog.hex
    cat test/behav//gpu_die_driver.sv | sed -e "s/{PROG}/prog/g" > build/gpu_die_driver.sv

    iverilog -g2012 tech/osu018/osu018_stdcells.v build/netlist/6.v src/assert_ignore.sv src/const.sv \
        src/mem_large.sv build/gpu_die_driver.sv
    ./a.out | tee build/out.txt
    if  ! cat build/out.txt | grep '^out[ \.]' > build/out_only.txt; then {
        echo "grep failed"
        echo "" > build/out.txt
    } fi

    if cat build/out.txt | grep 'ERROR'; then {
        echo "ERROR"
        exit 1
    } fi

    if diff build/out_only.txt examples/direct/expected/${prog}_expected.txt; then {
        echo SUCCESS
    } else {
        echo FAIL
        exit 1
    } fi
} done

