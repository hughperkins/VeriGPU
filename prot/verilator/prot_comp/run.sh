#!/bin/bash

set -ex
set -o pipefail

cd prot/verilator

if [[ ! -d build ]]; then {
    mkdir build
} fi

cd build

cmake ..
make -j $(nproc)
./verilator1
