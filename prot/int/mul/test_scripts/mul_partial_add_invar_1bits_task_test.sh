#!/bin/bash

# combines mul_partial_driver_clocked with mul_partial_add_task and mul_clocked_test

set -ex

if [[ -f a.out ]]; then {
    rm a.out
} fi
iverilog -g2012 \
    src/const.sv \
    src/assert.sv \
    prot/int/mul/params/bits_per_cycle_1.sv \
    prot/int/mul/mul_partial_add_invar_task.sv \
    prot/int/mul/mul_partial_driver.sv \
    prot/int/mul/test/mul_test.sv
./a.out
