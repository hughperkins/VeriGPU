#!/bin/bash

set -x
set -e
set -o pipefail

# iverilog -g2012 src/op_const.sv src/const.sv src/int_div_regfile.sv src/proc.sv
# check that verilator compilation succeeds (verilator acts as a type of linter here)
verilator -sv --cc src/op_const.sv src/assert.sv src/const.sv src/int_div_regfile.sv src/proc.sv

python -V
yosys -V
pip install -e .

bash test/behav/run_sv_tests.sh
bash test/behav/run_examples.sh
