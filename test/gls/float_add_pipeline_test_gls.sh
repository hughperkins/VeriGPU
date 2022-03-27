#!/bin/bash

set -ex
set -o pipefail

python verigpu/run_yosys.py --in-verilog src/assert_ignore.sv src/const.sv src/float/float_params.sv \
    src/float/float_add_pipeline.sv --top-module float_add_pipeline > /dev/null

iverilog -g2012 src/assert_ignore.sv src/float/float_params.sv \
    tech/osu018/osu018_stdcells.v build/netlist/6.v \
    test/lib/float_test_funcs.sv test/behav/float_add_pipeline_test.sv
./a.out
