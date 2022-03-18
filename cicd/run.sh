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
pip install -r requirements-test.txt
pytest -v .
bash test/reg_test.sh
bash test/run_sv_tests.sh

python test/get_prog_cycles.py | tee build/clock-cycles.txt
bash test/int_div_regfile_test_gls.sh
bash test/int_div_regfile_comp_gls.sh
bash test/proc_gls.sh
bash test/delay_prop_proc.sh | tee build/timing-proc.txt
