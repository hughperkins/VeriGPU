#!/bin/bash

# just timing tests

python -V
pip freeze
yosys -V
pip install -e .

python test/timing/get_prog_cycles.py | tee build/clock-cycles.txt
bash test/timing/delay_prop_proc.sh | tee build/timing-proc.txt
bash test/timing/delay_prop_whole_comp.sh | tee build/timing-comp.txt
