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
    -S ../sum_ints.cpp \
    -o sum_ints-hostraw.ll

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
    -S ../sum_ints.cpp \
    -o sum_ints-device.ll

${CLANGDIR}/bin/llc sum_ints-device.ll -o sum_ints-device.s --march=riscv32

# now we have to patch hostside...
${BASEDIR}/prot/verilator/prot_single_source/build-cmake-mac/patch_hostside \
     --devicellfile sum_ints-device.ll \
     --deviceriscvfile sum_ints-device.s \
     --hostrawfile sum_ints-hostraw.ll \
     --hostpatchedfile sum_ints-hostpatched.ll
echo patched hostside

${CLANGDIR}/bin/llc sum_ints-hostpatched.ll -o sum_ints-hostpatched.s

verilator -sv -Wall -cc  ${BASEDIR}/prot/verilator/prot_single_source/controller.sv --top-module controller --prefix controller

(
    cd obj_dir
    make -f controller.mk
)

# g++ -std=c++11 -I${BASEDIR}/prot/verilator/prot_single_source -c ../sum_ints.cpp
g++ -std=c++14 -c sum_ints-hostpatched.s
g++ -std=c++14 -I${VERILATORDIR}/include -c ${VERILATORDIR}/include/verilated.cpp
g++ -std=c++14 -I obj_dir -I${VERILATORDIR}/include -c ${BASEDIR}/prot/verilator/prot_single_source/gpu_runtime.cpp

# g++ -o sum_ints sum_ints-hostpatched.o gpu_runtime.o verilated.o obj_dir/controller__ALL.o
g++ -o sum_ints sum_ints-hostpatched.o -L${BASEDIR}/prot/verilator/prot_single_source/build-cmake-mac -lverigpu_runtime

set +x

for i in {1..10}; do {
    ./sum_ints +verilator+rand+reset+2 +verilator+seed+$(($RANDOM * 65536 + $RANDOM))
} done
