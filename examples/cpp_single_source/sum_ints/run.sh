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
# CLANGDIR=$HOME/Downloads/clang+llvm-14.0.0-x86_64-apple-darwin

if [[ $(uname) == Linux ]]; then {
    # assume installed clang 13 using https://apt.llvm.org/
    # if you want to handle other scenarios, please submit a PR :)
    echo Linux detected
    # alias clang++=clang++-13
    CLANGPP=clang++-13
    CLANG=clang-13
    LLC=llc-13
    # PATHHOSTSIDE=${BASEDIR}/prot/verilator/prot_single_source/build-cmake-linux/patch_hostside
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
    # PATHHOSTSIDE=${BASEDIR}/prot/verilator/prot_single_source/build-cmake-mac/patch_hostside
    MACCLTINCLUDE="-I/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/usr/include"
} fi

cd ${SCRIPTDIR}

# if [[ -d build_bash ]]; then {
    # rm -R build_bash
# } fi

if [[ ! -d build_bash ]]; then {
    mkdir build_bash
} fi

cd build_bash

# LIBRARY_PATH=${CLANGDIR}/

# building patch_hostside is being migrated into cmake script at 
# /prot/verilator/prot_single_source/CMakeLists.txt

# host-side: -.cu => -hostraw.cll
# clang++-13 --version
# CLANGPP="clang++-13"
# ${CLANGPP} --version
# alias
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

# verilator -sv -Wall -cc \
#     ${SRC}/const.sv \
#     ${SRC}/op_const.sv \
#     ${SRC}/assert.sv \
#     ${SRC}/core.sv \
#     ${SRC}/global_mem_controller.sv \
#     ${SRC}/gpu_controller.sv \
#     ${SRC}/gpu_die.sv \
#     ${SRC}/gpu_card.sv \
#     --top-module gpu_card \
#     --prefix gpu_card

# (
#     cd obj_dir
#     make -f gpu_card.mk
# )

# g++ -std=c++11 -I${BASEDIR}/prot/verilator/prot_single_source -c ../sum_ints.cpp
g++ -std=c++14 -fPIE -c sum_ints-hostpatched.s
g++ -std=c++14 -fPIE -I${VERILATORDIR}/include -c ${VERILATORDIR}/include/verilated.cpp
# g++ -std=c++14 -I obj_dir -I${VERILATORDIR}/include -c ${BASEDIR}/prot/verilator/prot_single_source/gpu_runtime.cpp

# g++ -o sum_ints sum_ints-hostpatched.o gpu_runtime.o verilated.o obj_dir/controller__ALL.o
# g++ -c ${BASEDIR}/prot/cpp/hostside/prot_upper_lower.cpp
g++ -o sum_ints sum_ints-hostpatched.o -L${GPURUNTIMEDIR} -lverigpu_runtime ${LIBEXPFS}
# g++ --pie -o sum_ints prot_upper_lower.o -L${GPURUNTIMEDIR} -lverigpu_runtime
# ld --no-pie -o sum_ints sum_ints-hostpatched.o -L${GPURUNTIMEDIR} -lverigpu_runtime

set +x

# for i in {1..10}; do {
    ./sum_ints +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
# } done
