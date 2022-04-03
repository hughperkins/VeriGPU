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

cmake ..
make -j $(nproc)
./${BASENAME}

set +x

for i in {1..10}; do {
    ./${BASENAME} +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
} done
