#!/bin/bash

# needs expect to have been installed, for unbuffer

set -ex
set -o pipefail

# cd prot/verilator/prot_single_source

BASEDIR=$PWD

if [[ $(uname) == Linux ]]; then {
    echo Linux detected
    BUILDDIR=build/runtime-linux
} elif [[ $(uname) == Darwin ]]; then {
    echo Mac detected
    BUILDDIR=build/runtime-mac
} fi

if [[ ! -e ${BUILDDIR} ]]; then {
    mkdir ${BUILDDIR}
} fi

cd ${BUILDDIR}

cmake ${BASEDIR}/src/gpu_runtime

# unbuffer make -j $(nproc)
unbuffer make VERBOSE=1
