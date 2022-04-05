#!/bin/bash

SCRIPTDIR=$(dirname $0)

CLANGDIR=$HOME/Downloads/clang+llvm-14.0.0-x86_64-apple-darwin

# clang++ \
#     -std=c++11 -stdlib=libc++ -nostdinc++ \
#     -I/usr/local/include/c++/11 \
#     -c prot/verilator/prot_unified_source/my_gpu_test_client.cpp


# ${CLANGDIR}/bin/clang++ \
#     -std=c++11 -stdlib=libc++ -nostdinc++ \
#     -I${CLANGDIR}/include/c++/v1 \
#     -c prot/verilator/prot_unified_source/my_gpu_test_client.cpp

# ${CLANGDIR}/bin/clang++ \
#     -std=c++11 -stdlib=libc++ -nostdinc++ \
#     -I${CLANGDIR}/include/c++/v1 \
#     -S prot/verilator/prot_unified_source/my_gpu_test_client.cpp \
#     -o ${SCRIPTDIR}/build/test.ll

# this was challenging...
# probably specific to your mac version... :/

${CLANGDIR}/bin/clang++ \
    -x cuda \
    --cuda-device-only -emit-llvm \
    -nocudainc \
    -nocudalib \
    -I${CLANGDIR}/include \
    -I${CLANGDIR}/include/c++/v1 \
    -I/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/usr/include \
    -S prot/verilator/prot_unified_source/my_gpu_test_client.cpp \
    -o ${SCRIPTDIR}/build/test.ll

${CLANGDIR}/bin/llc prot/verilator/prot_unified_source/build/test.ll -o test.s --march=riscv32

# ${CLANGDIR}/bin/clang++ \
#     -x cuda \
#     --cuda-device-only \
#     -nocudainc \
#     -nocudalib \
#     -I${CLANGDIR}/include \
#     -I${CLANGDIR}/include/c++/v1 \
#     -I/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/usr/include \
#     -S prot/verilator/prot_unified_source/my_gpu_test_client.cpp \
#     -o ${SCRIPTDIR}/build/test.ll
    # -target riscv32

    # -std=c++11 -stdlib=libc++ -nostdinc++ \
    # -I/Library/Developer//CommandLineTools/usr/include/c++/v1/ \

        # [join(CLANG_HOME, 'bin', 'clang++')] +
        # PASS_THRU + [
        #     '-DUSE_CLEW',
        #     '-std=c++11', '-x', 'cuda',
        #     '-D__CORIANDERCC__',
        #     '-D__CUDACC__',
        #     '--cuda-gpu-arch=sm_30', '-nocudalib', '-nocudainc', '--cuda-device-only', '-emit-llvm',
        #     '-O%s' % DEVICE_PARSE_OPT_LEVEL,
        #     '-S'
        # ] + ADDFLAGS + [
        #     '-Wno-gnu-anonymous-struct',
        #     '-Wno-nested-anon-types'
        # ] + LLVM_COMPILE_FLAGS_LIST + [
        #     # '-I%s' % join(COCL_INCLUDE, 'EasyCL'),
        #     # '-I%s' % join(COCL_INCLUDE, 'EasyCL', 'third_party', 'clew', 'include'),
        #     # '-I%s' % join(COCL_INCLUDE),
        #     '-I%s' % join(COCL_INCLUDE, 'cocl'),  # for cuda.h
        #     # '-I%s' % join(COCL_INCLUDE, 'cocl', 'proxy_includes'),
        #     '-include', join(COCL_INCLUDE, 'cocl', 'cocl.h'),
        #     '-include', join(COCL_INCLUDE, 'cocl', 'fake_funcs.h'),
        #     '-include', join(COCL_INCLUDE, 'cocl', 'cocl_deviceside.h'),
        #     '-I%s' % COCL_INCLUDE,
        # ] + INCLUDES + [
        #     INPUTBASEPATH + INPUTPOSTFIX,
        #     '-o', '%s-device-noopt.ll' % OUTPUTBASEPATH
        # ])
