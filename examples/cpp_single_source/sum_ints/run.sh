#!/bin/bash

set -ex
set -o pipefail

SCRIPTDIR=$(dirname $0)
BASENAME=$(basename $SCRIPTDIR)

VERILATORDIR=/usr/local/share/verilator

BASEDIR=$PWD

cd ${SCRIPTDIR}

if [[ -d build_bash ]]; then {
    rm -R build_bash
} fi

if [[ ! -d build_bash ]]; then {
    mkdir build_bash
} fi

cd build_bash

CLANGDIR=$HOME/Downloads/clang+llvm-14.0.0-x86_64-apple-darwin
MACCLTINCLUDEDIR=/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/usr/include

# this was challenging...
# probably specific to your mac version... :/

# TODO COMPILE FOR HOSTSIDE
# ${CLANGDIR}/bin/clang++ \
#     -x cuda \
#     --cuda-device-only -emit-llvm \
#     -nocudainc \
#     -nocudalib \
#     -I${CLANGDIR}/include \
#     -I${CLANGDIR}/include/c++/v1 \
#     -I${MACCLTINCLUDEDIR} \
#     -I${BASEDIR}/prot/verilator/prot_single_source \
#     -S ../sum_ints.cpp \
#     -o sum_ints.ll

${CLANGDIR}/bin/clang++ \
    -x cuda \
    --cuda-device-only -emit-llvm \
    -nocudainc \
    -nocudalib \
    -I${CLANGDIR}/include \
    -I${CLANGDIR}/include/c++/v1 \
    -I${MACCLTINCLUDEDIR} \
    -I${BASEDIR}/prot/verilator/prot_single_source \
    -S ../sum_ints.cpp \
    -o sum_ints.ll

    # -S prot/verilator/prot_unified_source/my_gpu_test_client.cpp \

${CLANGDIR}/bin/llc sum_ints.ll -o sum_ints.s --march=riscv32

verilator -sv -Wall -cc  ${BASEDIR}/prot/verilator/prot_single_source/controller.sv --top-module controller --prefix controller

(
    cd obj_dir
    make -f controller.mk
)

g++ -std=c++11 -I${BASEDIR}/prot/verilator/prot_single_source -c ../sum_ints.cpp
g++ -std=c++11 -I${VERILATORDIR}/include -c ${VERILATORDIR}/include/verilated.cpp
g++ -std=c++11 -I obj_dir -I${VERILATORDIR}/include -c ${BASEDIR}/prot/verilator/prot_single_source/gpu_runtime.cpp

g++ -o sum_ints sum_ints.o gpu_runtime.o verilated.o obj_dir/controller__ALL.o

set +x

for i in {1..10}; do {
    ./sum_ints +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
} done
