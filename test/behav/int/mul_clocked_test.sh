#!/bin/bash

# combines mul_partial_driver_clocked with mul_partial_add_task and mul_clocked_test

set -ex

iverilog -g2012 \
    src/const.sv \
    src/assert.sv \
    src/generated/mul_pipeline_cycle_32bit_2bpc.sv \
    src/int/mul_pipeline_32bit.sv \
    test/behav/int/mul_clocked_test.sv

./a.out
