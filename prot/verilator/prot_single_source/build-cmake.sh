#!/bin/bash

# needs expect to have been installed, for unbuffer

set -ex
set -o pipefail

cd prot/verilator/prot_single_source

if [[ $(uname) == Linux ]]; then {
    echo Linux detected
    BUILDDIR=build-cmake-linux
} elif [[ $(uname) == Darwin ]]; then {
    echo Mac detected
    BUILDDIR=build-cmake-mac
} fi

if [[ ! -e ${BUILDDIR} ]]; then {
    mkdir ${BUILDDIR}
} fi

cd ${BUILDDIR}

cmake ..
# unbuffer make -j $(nproc)
unbuffer make VERBOSE=1
