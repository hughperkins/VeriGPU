#!/bin/bash

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv src/const.sv \
    prot/int/add/chunked_adder_task.sv \
    prot/int/add/test/add_test.sv
./a.out
