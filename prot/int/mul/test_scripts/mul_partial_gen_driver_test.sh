#!/bin/bash

# combines mul_partial_driver with mul_partial_add_task

set -ex

iverilog -g2012 \
    src/const.sv \
    src/assert.sv \
    prot/int/mul/params/bits_per_cycle_2.sv \
    build/mul_pipeline_cycle_32bit_2bpc.sv \
    prot/int/mul/mul_partial_gen_driver.sv \
    prot/int/mul/test/mul_test.sv
./a.out
