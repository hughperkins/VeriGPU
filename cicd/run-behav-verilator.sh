#!/bin/bash

set -x
set -e
set -o pipefail

# test/behav/verilator_compile_proc.sh
# test/behav/verilator_compile_comp.sh

python -V
yosys -V

if ! python -c 'import verigpu'; then {
    pip install -e .
} fi

if [[ ! -e build ]]; then {
    mkdir build
} fi

bash test/behav/run_verilator_unit_tests.sh
bash test/behav/run_examples_verilator.sh
