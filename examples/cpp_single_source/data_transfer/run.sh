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

VERILATORDIR=/usr/local/share/verilator
# CLANGDIR=$HOME/Downloads/clang+llvm-14.0.0-x86_64-apple-darwin
CLANGDIR=/usr/local/opt/llvm-14.0.0
MACCLTINCLUDEDIR=/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/usr/include

BASEDIR=$PWD
SRC=${BASEDIR}/src

export VERIGPUDIR=${BASEDIR}

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
${CLANGDIR}/bin/clang++ \
    -std=c++11 -x cuda -nocudainc --cuda-host-only -emit-llvm \
    -I${CLANGDIR}/include \
    -I${CLANGDIR}/include/c++/v1 \
    -I${MACCLTINCLUDEDIR} \
    -I${BASEDIR}/prot/verilator/prot_single_source \
    -S ../data_transfer.cpp \
    -o data_transfer-hostraw.ll

# device-side => sum_ints.ll
${CLANGDIR}/bin/clang++ \
    -x cuda \
    --cuda-device-only -emit-llvm \
    -nocudainc \
    -nocudalib \
    -I${CLANGDIR}/include \
    -I${CLANGDIR}/include/c++/v1 \
    -I${MACCLTINCLUDEDIR} \
    -I${BASEDIR}/prot/verilator/prot_single_source \
    -S ../data_transfer.cpp \
    -o data_transfer-device.ll

${CLANGDIR}/bin/llc data_transfer-device.ll -o data_transfer-device.s --march=riscv32

# now we have to patch hostside...
${BASEDIR}/prot/verilator/prot_single_source/build-cmake-mac/patch_hostside \
     --devicellfile data_transfer-device.ll \
     --deviceriscvfile data_transfer-device.s \
     --hostrawfile data_transfer-hostraw.ll \
     --hostpatchedfile data_transfer-hostpatched.ll
echo patched hostside

${CLANGDIR}/bin/llc data_transfer-hostpatched.ll -o data_transfer-hostpatched.s

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
g++ -std=c++14 -c data_transfer-hostpatched.s
g++ -std=c++14 -I${VERILATORDIR}/include -c ${VERILATORDIR}/include/verilated.cpp
# g++ -std=c++14 -I obj_dir -I${VERILATORDIR}/include -c ${BASEDIR}/prot/verilator/prot_single_source/gpu_runtime.cpp

# g++ -o sum_ints sum_ints-hostpatched.o gpu_runtime.o verilated.o obj_dir/controller__ALL.o
g++ -o data_transfer data_transfer-hostpatched.o -L${BASEDIR}/prot/verilator/prot_single_source/build-cmake-mac -lverigpu_runtime

set +x

for i in {1..10}; do {
    ./data_transfer +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
} done
