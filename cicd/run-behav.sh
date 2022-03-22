#!/bin/bash

set -x
set -e
set -o pipefail

test/behav/verilator_compile_proc.sh

python -V
yosys -V
pip install -e .

if [[ ! -e build ]]; then {
    mkdir build
} fi

bash test/behav/run_sv_tests.sh
bash test/behav/run_examples.sh
