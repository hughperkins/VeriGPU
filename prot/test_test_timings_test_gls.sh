#!/bin/bash

set -ex
set -o pipefail

python toy_proc/run_yosys.py --in-verilog src/assert_ignore.sv prot/test_test_timings.sv --top-module test_test_timings >/dev/null
iverilog -g2012 -Wall tech/osu018/osu018_stdcells.v build/netlist/6.v src/assert.sv prot/test_test_timings_test.sv
./a.out 
