#!/bin/bash

set -x
set -e
yosys -s src/yosys.tcl
tail -n +2 build/netlist.v > build/tmp
mv build/tmp build/netlist.v
sed -i -e 's/(\*.*\*)//g' build/netlist.v
sta -exit build/netlist.v
