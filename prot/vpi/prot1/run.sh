#!/bin/bash

set -ex

scriptdir=$(dirname $0)
echo scriptdir $scriptdir


(cd $scriptdir

if [[ ! -e build ]]; then {
    mkdir build
} fi

cd build
iverilog-vpi ../hello.cpp
iverilog -g2012 -ohello.vvp ../hello.sv
vvp -M. -mhello hello.vvp
)
