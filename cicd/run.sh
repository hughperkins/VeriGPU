#!/bin/bash

set -x
set -e
iverilog -g2012 src/proc.sv
verilator -sv --cc src/proc.sv
python -V
pip freeze
yosys -V
pip install -e .
pip install -r requirements-test.txt
pytest -v .
bash test/reg_test.sh

python toy_proc/timing.py --in-verilog src/proc.sv | tee build/timing-proc.txt
python toy_proc/timing.py --in-verilog src/const.sv src/int_div_regfile.sv src/apu.sv --top-module apu | tee build/timing-apu.txt
