#!/bin/bash

# Usage:

# examples/cpp_single_source/run.sh [example name]

# e.g.

# examples/cpp_single_source/run.sh matrix_mul

# This expects 'expect' to be installed, e.g.
# brew install expect
# or
# sudo apt-get install expect
# (otherwise, delete 'unbuffer' keywords)

set -ex
set -o pipefail

if [[ x$1 == x ]]; then {
    echo "Usage: $0 [example name]"
    exit 1
} fi

BASENAME=$1

SCRIPTDIR=$(dirname $0)
# BASENAME=$(basename $SCRIPTDIR)
BASEDIR=$PWD
SRC=${BASEDIR}/src
export VERIGPUDIR=${BASEDIR}
BUILDDIR=build/${BASENAME}
SRC=$PWD/examples/cpp_single_source

VERILATORDIR=/usr/local/share/verilator

if [[ ! -e ${VERIGPUDIR}/build ]]; then {
    mkdir -p ${VERIGPUDIR}/build
} fi

if [[ $(uname) == Linux ]]; then {
    # assume installed clang 14 using https://apt.llvm.org/
    # if you want to handle other scenarios, please submit a PR :)
    echo Linux detected
    CLANGPP=clang++-14
    CLANG=clang-14
    # you need to git clone llvm-project from https://github.com/llvm/llvm-project.git
    # and then `git apply` the patch from https://reviews.llvm.org/D122918
    # (and then use `ccmake` on the `llvm` folder to configure,
    # and then `make -j 8 llc` to build)
    LLC_ZFINX=/usr/local/bin/llc-zfinx
    LLC=llc-14
    GPURUNTIMEDIR=${BASEDIR}/prot/verilator/prot_single_source/build-cmake-linux
    LIBEXPFS=-lstdc++fs
} elif [[ $(uname) == Darwin ]]; then {
    echo Mac detected
    CLANGDIR=/usr/local/opt/llvm-14.0.0
    CLANGPP=${CLANGDIR}/bin/clang++
    CLANG=${CLANGDIR}/bin/clang
    LLC=${CLANGDIR}/bin/llc
    # you need to git clone llvm-project from https://github.com/llvm/llvm-project.git
    # and then `git apply` the patch from https://reviews.llvm.org/D122918
    # (and then use `ccmake` on the `llvm` folder to configure,
    # and then `make -j 8 llc` to build)
    LLC_ZFINX=/usr/local/bin/llc-zfinx
    GPURUNTIMEDIR=${BASEDIR}/prot/verilator/prot_single_source/build-cmake-mac
    MACCLTINCLUDE="-I/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/usr/include"
} fi

# cd ${SCRIPTDIR}

if [[ -d ${BUILDDIR} ]]; then {
    rm -R ${BUILDDIR}
} fi

if [[ ! -d ${BUILDDIR} ]]; then {
    mkdir ${BUILDDIR}
} fi

cd ${BUILDDIR}

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
    -S ${SRC}/${BASENAME}.cpp \
    -o ${BASENAME}-hostraw.ll

# device-side => ${BASENAME}.ll
${CLANGPP} -fPIE \
    -x cuda \
    --cuda-device-only -emit-llvm \
    -nocudainc \
    -nocudalib \
    -I${CLANGDIR}/include \
    -I${CLANGDIR}/include/c++/v1 \
    ${MACCLTINCLUDE} \
    -I${BASEDIR}/prot/verilator/prot_single_source \
    -S ${SRC}/${BASENAME}.cpp \
    -o ${BASENAME}-device.ll

${LLC_ZFINX} ${BASENAME}-device.ll -o ${BASENAME}-device.s --march=riscv32 -mattr=+m,+zfinx

# now we have to patch hostside...
${GPURUNTIMEDIR}/patch_hostside \
     --devicellfile ${BASENAME}-device.ll \
     --deviceriscvfile ${BASENAME}-device.s \
     --hostrawfile ${BASENAME}-hostraw.ll \
     --hostpatchedfile ${BASENAME}-hostpatched.ll
echo patched hostside

${LLC} ${BASENAME}-hostpatched.ll --relocation-model=pic -o ${BASENAME}-hostpatched.s

g++ -std=c++14 -fPIE -c ${BASENAME}-hostpatched.s
g++ -std=c++14 -fPIE -I${VERILATORDIR}/include -c ${VERILATORDIR}/include/verilated.cpp

g++ -o ${BASENAME} ${BASENAME}-hostpatched.o -L${GPURUNTIMEDIR} -lverigpu_runtime ${LIBEXPFS}

set +x

./${BASENAME} +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
