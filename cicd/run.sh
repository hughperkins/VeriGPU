#!/bin/bash

set -x
set -e
iverilog -g2012 src/proc.sv
verilator -sv --cc src/proc.sv
python -V
pip freeze
yosys -V
pip install -e .
pytest -v .
bash src/reg_test.sh
python toy_proc/timing.py --in-verilog src/proc.sv
