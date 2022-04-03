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
make -j $(nproc) comp_netlist
./comp_netlist +verilator+rand+reset+0
./comp_netlist +verilator+rand+reset+1
./comp_netlist +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
