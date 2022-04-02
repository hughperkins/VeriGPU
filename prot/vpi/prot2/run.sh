#!/bin/bash

set -ex

scriptdir=$(dirname $0)
echo scriptdir $scriptdir


(cd $scriptdir

if [[ ! -e build ]]; then {
    mkdir build
} fi

cd build
# iverilog-vpi ../hello.cpp
# iverilog -g2012 -ohello.vvp ../hello.sv
# vvp -M. -mhello hello.vvp

iverilog -g2012 -ohello.vvp ../hello.sv
# g++ -I../install.d/include/iverilog -c test_client.cpp 
g++ -c ../test_client.cpp 
g++ -o test_client test_client.o -lvvp
./test_client

)
