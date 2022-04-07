#!/bin/bash

# needs expect to have been installed, for unbuffer

set -ex
set -o pipefail

cd prot/verilator/prot_single_source

if [[ ! -e build ]]; then {
    mkdir build
} fi

cd build

cmake ..
# unbuffer make -j $(nproc)
unbuffer make VERBOSE=1
