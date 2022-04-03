#!/bin/bash

set -ex

SCRIPTDIR=$(dirname $0)
BASENAME=$(basename $0 .sh)
echo SCRIPTDIR $SCRIPTDIR BASENAME $BASENAME

cd ${SCRIPTDIR}

if [[ ! -e build ]]; then {
    mkdir build
} fi

cd build
cmake ..

make -j $(nproc)
./${BASENAME} +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
