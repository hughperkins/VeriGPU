#!/bin/bash

# run the various _test.sv files

set -e
set -x

run_verilog() {
    iverilog -g2012 "$@"
    ./a.out | tee /tmp/out.txt
    set +x
    if grep ERROR /tmp/out.txt; then {
        echo "ERROR"
        exit 1
    } fi
    set -x
}

run_verilog test/mem_delayed_test.sv src/mem_delayed.sv src/const.sv
run_verilog test/apu_test.sv src/const.sv src/apu.sv src/int_div_regfile.sv
run_verilog test/int_div_regfile_test.sv src/const.sv src/int_div_regfile.sv
run_verilog test/apu_regfile_test.sv src/apu.sv src/reg_file.sv src/const.sv src/int_div_regfile.sv

echo "PASS"
