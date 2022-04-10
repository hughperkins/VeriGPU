#!/bin/bash

# like proc_gls.sh, but using verilator

# run gate-level simulation on src/proc.sv

# we'll use the original sv for mem_delayed, comp and comp_driver, since mem_delayed takes
# forever to synthesize...

set -ex
set -o pipefail

progs=$(cd examples/direct; ls -b *.asm | sed -e 's/.asm//')

if [[ x$1 != x ]]; then {
    progs=$1
} fi

if [[ -f build/netlist/4.v ]]; then {
    rm build/netlist/4.v
} fi

if [[ -f build/netlist/6.v ]]; then {
    rm build/netlist/6.v
} fi

python verigpu/run_yosys.py --in-verilog src/assert_ignore.sv src/op_const.sv src/const.sv \
    src/float/float_params.sv src/float/float_add_pipeline.sv \
    src/int/chunked_add_task.sv src/int/chunked_sub_task.sv \
    src/generated/mul_pipeline_cycle_24bit_2bpc.sv src/float/float_mul_pipeline.sv \
    src/generated/mul_pipeline_cycle_32bit_2bpc.sv src/int/mul_pipeline_32bit.sv \
    src/int/int_div_regfile.sv src/core.sv \
    --top-module core >/dev/null

for prog in ${progs}; do {
    python verigpu/assembler.py --offset 128 --in-asm examples/direct/${prog}.asm --out-hex build/prog.hex
    cat test/behav/single_core_mounted_driver.sv | sed -e "s/{PROG}/prog/g" > build/single_core_mounted_driver.sv

    bash test/behav/verilator/run_netlist.sh | tee build/out.txt
    if  ! cat build/out.txt | grep '^out[ \.]' > build/out_only.txt; then {
        echo "grep failed"
        echo "" > build/out.txt
    } fi

    if cat build/out.txt | grep 'ERROR'; then {
        echo "ERROR"
        exit 1
    } fi

    cat examples/direct/expected/${prog}_expected.txt > /tmp/expected.txt
    cat examples/direct/expected/${prog}_expected.txt >> /tmp/expected.txt
    cat examples/direct/expected/${prog}_expected.txt >> /tmp/expected.txt

    echo "output"
    cat build/out.txt

    echo "expected"
    cat /tmp/expected.txt

    if diff build/out_only.txt /tmp/expected.txt; then {
        echo SUCCESS
    } else {
        echo FAIL
        exit 1
    } fi
} done
