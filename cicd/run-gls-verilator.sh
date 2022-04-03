#!/bin/bash

# Just gate-level simulation tests

set -ex
set -o pipefail

python -V
pip freeze
yosys -V
pip install -e .

if [[ ! -e build ]]; then {
    mkdir build
} fi

bash test/gls/proc_gls_vtor.sh
