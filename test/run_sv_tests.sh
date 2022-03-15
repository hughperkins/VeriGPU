#!/bin/bash

set -e
set -x

run_verilog() {
    $1 | tee /tmp/out.txt
    set +x
    if grep ERROR /tmp/out.txt; then {
        echo "ERROR"
        exit 1
    } fi
    set -x
}

run_verilog test/int_div_regfile_test.sh
run_verilog test/mem_delayed_test.sh

echo "PASS"
