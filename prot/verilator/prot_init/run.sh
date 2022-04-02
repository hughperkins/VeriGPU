#!/bin/bash

set -ex
set -o pipefail

cd prot/verilator/prot_init

if [[ ! -d build ]]; then {
    mkdir build
} fi

cd build

cmake ..
make -j $(nproc)
./prot_init

set +x

for i in {1..10}; do {
    ./prot_init +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
} done
