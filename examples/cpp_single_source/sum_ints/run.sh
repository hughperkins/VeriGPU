#!/bin/bash

# This expects 'expect' to be installed, e.g.
# brew install expect
# or
# sudo apt-get install expect
# (otherwise, delete 'unbuffer' keywords)

set -ex
set -o pipefail

SCRIPTDIR=$(dirname $0)
BASENAME=$(basename $SCRIPTDIR)
BASEDIR=$PWD
SRC=${BASEDIR}/src
export VERIGPUDIR=${BASEDIR}

VERILATORDIR=/usr/local/share/verilator

if [[ $(uname) == Linux ]]; then {
    # assume installed clang 13 using https://apt.llvm.org/
    # if you want to handle other scenarios, please submit a PR :)
    echo Linux detected
    CLANGPP=clang++-13
    CLANG=clang-13
    LLC=llc-13
    GPURUNTIMEDIR=${BASEDIR}/prot/verilator/prot_single_source/build-cmake-linux
    LIBEXPFS=-lstdc++fs
} elif [[ $(uname) == Darwin ]]; then {
    echo Mac detected
    CLANGDIR=/usr/local/opt/llvm-14.0.0
    CLANGPP=${CLANGDIR}/bin/clang++
    CLANG=${CLANGDIR}/bin/clang
    LLC=${CLANGDIR}/bin/llc
    BUILDDIR=cmake-mac
    GPURUNTIMEDIR=${BASEDIR}/prot/verilator/prot_single_source/build-cmake-mac
    MACCLTINCLUDE="-I/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/usr/include"
} fi

cd ${SCRIPTDIR}

if [[ -d build_bash ]]; then {
    rm -R build_bash
} fi

if [[ ! -d build_bash ]]; then {
    mkdir build_bash
} fi

cd build_bash

# building patch_hostside is being migrated into cmake script at 
# /prot/verilator/prot_single_source/CMakeLists.txt

# host-side: -.cu => -hostraw.cll
echo ${BASEDIR}
${CLANGPP} -fPIE \
    -std=c++11 -x cuda -nocudainc --cuda-host-only -emit-llvm \
    -I${CLANGDIR}/include \
    -I${CLANGDIR}/include/c++/v1 \
    ${MACCLTINCLUDE} \
    -I${BASEDIR}/prot/verilator/prot_single_source \
    -S ../sum_ints.cpp \
    -o sum_ints-hostraw.ll

# device-side => sum_ints.ll
${CLANGPP} -fPIE \
    -x cuda \
    --cuda-device-only -emit-llvm \
    -nocudainc \
    -nocudalib \
    -I${CLANGDIR}/include \
    -I${CLANGDIR}/include/c++/v1 \
    ${MACCLTINCLUDE} \
    -I${BASEDIR}/prot/verilator/prot_single_source \
    -S ../sum_ints.cpp \
    -o sum_ints-device.ll

${LLC} sum_ints-device.ll -o sum_ints-device.s --march=riscv32

# now we have to patch hostside...
${GPURUNTIMEDIR}/patch_hostside \
     --devicellfile sum_ints-device.ll \
     --deviceriscvfile sum_ints-device.s \
     --hostrawfile sum_ints-hostraw.ll \
     --hostpatchedfile sum_ints-hostpatched.ll
echo patched hostside

${LLC} sum_ints-hostpatched.ll --relocation-model=pic -o sum_ints-hostpatched.s

g++ -std=c++14 -fPIE -c sum_ints-hostpatched.s
g++ -std=c++14 -fPIE -I${VERILATORDIR}/include -c ${VERILATORDIR}/include/verilated.cpp

g++ -o sum_ints sum_ints-hostpatched.o -L${GPURUNTIMEDIR} -lverigpu_runtime ${LIBEXPFS}

set +x

./sum_ints +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
