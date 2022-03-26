#!/bin/bash

# combines mul_partial_driver with mul_partial_add_task

set -ex

iverilog -g2012 \
    src/const.sv \
    src/assert.sv \
    prot/int/mul/mul_partial_add_task.sv \
    prot/int/mul/mul_partial_driver.sv \
    prot/int/mul/test/mul_test.sv
./a.out
