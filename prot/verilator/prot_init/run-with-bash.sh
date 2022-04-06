#!/bin/bash

set -ex
set -o pipefail

SCRIPTDIR=$(dirname $0)
BASENAME=$(basename $SCRIPTDIR)

VERILATORDIR=/usr/local/share/verilator

cd ${SCRIPTDIR}

if [[ -d build_bash ]]; then {
    rm -R build_bash
} fi

if [[ ! -d build_bash ]]; then {
    mkdir build_bash
} fi

cd build_bash

verilator -sv -Wall -cc ../prot_init.sv --top-module prot_init --prefix prot_init

(
    cd obj_dir
    make -f prot_init.mk
)

g++ -std=c++11 -I obj_dir -I${VERILATORDIR}/include -c ../prot_init.cpp
g++ -std=c++11 -I obj_dir -I${VERILATORDIR}/include -c ${VERILATORDIR}/include/verilated.cpp
g++ -o prot_init prot_init.o verilated.o obj_dir/prot_init__ALL.o

set +x

for i in {1..10}; do {
    ./${BASENAME} +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
} done
