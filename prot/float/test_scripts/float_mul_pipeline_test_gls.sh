#!/bin/bash

BASE=prot/float

set -ex
set -o pipefail

python toy_proc/run_yosys.py --in-verilog src/assert_ignore.sv src/const.sv ${BASE}/float_params.sv \
    ${BASE}/float_mul_pipeline.sv --top-module float_mul_pipeline > /dev/null

iverilog -g2012 src/assert_ignore.sv ${BASE}/float_params.sv \
    tech/osu018/osu018_stdcells.v build/netlist/6.v \
    ${BASE}/float_test_funcs.sv ${BASE}/float_mul_pipeline_test.sv
./a.out
