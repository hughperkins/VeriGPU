#!/bin/bash

# Just gate-level simulation tests

python -V
pip freeze
yosys -V
pip install -e .

bash test/gls/gls_tests.sh
