#!/bin/bash

CLANG_HOME=$1
LDLIBFLAGS=$(${CLANG_HOME}/bin/llvm-config --ldflags --libs engine | tr -d '\n')
LDSYSFLAGS=$(${CLANG_HOME}/bin/llvm-config --system-libs engine | tr -d '\n')
echo ${LDLIBFLAGS} ${LDSYSFLAGS}
