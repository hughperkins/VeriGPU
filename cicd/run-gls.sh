#!/bin/bash

# Just gate-level simulation tests

python -V
pip freeze
yosys -V
pip install -e .

if [[ ! -e build ]]; then {
    mkdir build
} fi

bash test/gls/gls_tests.sh
