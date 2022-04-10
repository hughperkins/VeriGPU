#!/bin/bash

set -ex
set -o pipefail

SCRIPTDIR=$(dirname $0)
BASENAME=$(basename $SCRIPTDIR)

cd ${SCRIPTDIR}

if [[ ! -d build ]]; then {
    mkdir build
} fi

cd build

cmake .. -DBUILD_NETLIST:bool=true
make -j $(nproc) single_core_netlist
./single_core_netlist +verilator+rand+reset+0
./single_core_netlist +verilator+rand+reset+1
./single_core_netlist +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
