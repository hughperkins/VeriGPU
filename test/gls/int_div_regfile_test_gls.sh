#!/bin/bash

# run gate-level simulation on src/int_div_regfile.sv

set -ex
set -o pipefail

python toy_proc/run_yosys.py --in-verilog src/assert_ignore.sv src/const.sv src/int/int_div_regfile.sv --top-module int_div_regfile >/dev/null
iverilog -g2012 tech/osu018/osu018_stdcells.v build/netlist/6.v src/assert.sv src/const.sv test/behav/int_div_regfile_test.sv
./a.out
