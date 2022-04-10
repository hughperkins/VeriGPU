#!/bin/bash

# just timing tests

set -ex
set -o pipefail

python -V
pip freeze
yosys -V
pip install -e .

if [[ ! -e build ]]; then {
    mkdir build
} fi

python test/timing/get_prog_cycles.py | tee build/clock-cycles.txt
bash test/timing/delay_prop_core.sh | tee build/timing-core.txt
bash test/timing/delay_prop_gpu_die.sh | tee build/timing-gpu-die.txt
