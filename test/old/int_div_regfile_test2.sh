#!/bin/bash

# run gate-level simulation on src/int_div_regfile.sv

set -ex
set -o pipefail

# python verigpu/run_yosys.py --in-verilog src/const.sv src/int_div_regfile.sv --top-module int_div_regfile >/dev/null
iverilog -g2012 src/int_div_regfile.sv src/const.sv test/int_div_regfile_test2.sv
./a.out
