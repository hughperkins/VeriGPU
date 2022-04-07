#!/bin/bash


export CLANG_HOME=$1


if (($(cat /etc/*-release | grep -ce 'arch') > 0)); then
  SYSLIBS=$(${CLANG_HOME}/bin/llvm-config --system-libs core | tr -d '\n' | sed -e 's/ -ltinfo/ -lcurses/')
  echo ${SYSLIBS}
  exit 0
fi

SYSLIBS=$(${CLANG_HOME}/bin/llvm-config --system-libs core | tr -d '\n')
echo ${SYSLIBS}
