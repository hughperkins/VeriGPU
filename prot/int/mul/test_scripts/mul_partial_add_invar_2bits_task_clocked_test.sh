#!/bin/bash

# combines mul_partial_driver_clocked with mul_partial_add_task and mul_clocked_test

set -ex

iverilog -g2012 \
    src/const.sv \
    src/assert.sv \
    prot/int/mul/params/bits_per_cycle_2.sv \
    prot/int/mul/mul_partial_add_invar_task.sv \
    prot/int/mul/mul_partial_driver_clocked.sv \
    prot/int/mul/test/mul_clocked_test.sv
./a.out
