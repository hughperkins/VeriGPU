#!/bin/bash

set -x
set -e
set -o pipefail

iverilog -g2012 src/op_const.sv src/const.sv src/proc.sv
verilator -sv --cc src/op_const.sv src/const.sv src/proc.sv
python -V
pip freeze
yosys -V
pip install -e .
pip install -r requirements-test.txt
pytest -v .
bash test/reg_test.sh
bash test/run_sv_tests.sh

python toy_proc/timing.py --in-verilog src/op_const.sv src/proc.sv | tee build/timing-proc.txt
python toy_proc/timing.py --in-verilog src/const.sv src/int_div_regfile.sv src/apu.sv --top-module apu | tee build/timing-apu.txt
