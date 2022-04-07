#!/bin/bash

export CLANG_HOME=$1
# echo CLANG_HOME ${CLANG_HOME}
# LIBS=$(${CLANG_HOME}/bin/llvm-config --libnames core | tr -d '\n')
LIBS=$(${CLANG_HOME}/bin/llvm-config --libnames all | tr -d '\n')
# SYSLIBS=$(${CLANG_HOME}/bin/llvm-config --libnames --system-libs core | tr -d '\n')
echo ${LIBS} ${SYSLIBS}