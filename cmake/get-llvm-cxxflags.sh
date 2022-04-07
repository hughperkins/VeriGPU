#!/bin/bash

CLANG_HOME=$1
if [[ $(uname) != Darwin ]]; then {
    ${CLANG_HOME}/bin/llvm-config --cxxflags | sed -e 's/-Wcovered-switch-default//' -e 's/-Wstring-conversion//' | tr -d '\n'
} else {
    ${CLANG_HOME}/bin/llvm-config --cxxflags | tr -d '\n'
} fi
