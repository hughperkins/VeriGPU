#!/bin/bash

set -ex
set -o pipefail

SCRIPTDIR=$(dirname $0)
BASENAME=$(basename $SCRIPTDIR)

VERILATORDIR=/usr/local/share/verilator

BASEDIR=$PWD

cd ${SCRIPTDIR}

if [[ -d build_bash ]]; then {
    rm -R build_bash
} fi

if [[ ! -d build_bash ]]; then {
    mkdir build_bash
} fi

cd build_bash

verilator -sv -Wall -cc  ${BASEDIR}/prot/verilator/prot_single_source/controller.sv --top-module controller --prefix controller

(
    cd obj_dir
    make -f controller.mk
)

g++ -std=c++11 -I${BASEDIR}/prot/verilator/prot_single_source -c ../data_transfer.cpp
g++ -std=c++11 -I${VERILATORDIR}/include -c ${VERILATORDIR}/include/verilated.cpp
g++ -std=c++11 -I obj_dir -I${VERILATORDIR}/include -c ${BASEDIR}/prot/verilator/prot_single_source/gpu_runtime.cpp

g++ -o data_transfer data_transfer.o gpu_runtime.o verilated.o obj_dir/controller__ALL.o

set +x

for i in {1..10}; do {
    ./data_transfer +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
} done
