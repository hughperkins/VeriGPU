#!/bin/bash

set -x
set -e
set -o pipefail

# iverilog -g2012 src/op_const.sv src/const.sv src/int_div_regfile.sv src/proc.sv
# check that verilator compilation succeeds (verilator acts as a type of linter here)
verilator -sv --cc src/op_const.sv src/assert.sv src/const.sv src/int_div_regfile.sv src/proc.sv

python -V
pip freeze
yosys -V
pip install -e .
pip install -r test/py/requirements.txt
pytest -v .
bash test/behav/run_sv_tests.sh
bash test/behav/run_examples.sh

python test/timing/get_prog_cycles.py | tee build/clock-cycles.txt
bash test/gls/gls_tests.sh
bash test/timing/delay_prop_proc.sh | tee build/timing-proc.txt
bash test/timing/delay_prop_comp.sh | tee build/timing-comp.txt
