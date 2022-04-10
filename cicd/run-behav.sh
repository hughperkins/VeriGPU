#!/bin/bash

set -x
set -e
set -o pipefail

python -V
yosys -V

if ! python -c 'import verigpu'; then {
    pip install -e .
} fi

if [[ ! -e build ]]; then {
    mkdir build
} fi

bash test/behav/run_sv_unit_tests.sh
bash test/behav/run_examples.sh
