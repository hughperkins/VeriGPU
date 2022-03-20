#!/bin/bash

# run python tests

python -V
pip freeze
yosys -V
pip install -e .
pip install -r test/py/requirements.txt

if [[ ! -e build ]]; then {
    mkdir build
} fi

pytest -v .
