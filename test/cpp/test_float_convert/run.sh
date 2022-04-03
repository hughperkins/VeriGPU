#!/bin/bash

set -ex
set -o pipefail

SCRIPTDIR=$(dirname $0)
BASENAME=$(basename $SCRIPTDIR)
echo SCRIPTDIR ${SCRIPTDIR} BASENAME ${BASENAME}

cd ${SCRIPTDIR}

if [[ ! -d build ]]; then {
    mkdir build
} fi

cd build

cmake ..
make -j $(nproc)
./${BASENAME}

# set +x

# for i in {1..10}; do {
#     ./prot_init +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
# } done
