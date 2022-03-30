#!/bin/bash

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv src/const.sv \
    src/int/chunked_sub_task.sv \
    test/behav/int/chunked_sub_test.sv
./a.out
