#!/bin/bash

set -ex
set -o pipefail

cd src/verilator

if [[ ! -d build ]]; then {
    mkdir build
} fi

cd build

cmake ..
make -j $(nproc)
./verilator1
