#!/bin/bash

set -x
set -e
set -o pipefail

iverilog -g2012 src/op_const.sv src/const.sv src/int_div_regfile.sv src/proc.sv
verilator -sv --cc src/op_const.sv src/const.sv src/int_div_regfile.sv src/proc.sv
python -V
pip freeze
yosys -V
pip install -e .
pip install -r test/requirements.txt
pytest -v .
bash test/reg_test.sh
bash test/run_sv_tests.sh

python test/get_prog_cycles.py | tee build/clock-cycles.txt
bash test/gls_tests.sh
bash test/delay_prop_proc.sh | tee build/timing-proc.txt
