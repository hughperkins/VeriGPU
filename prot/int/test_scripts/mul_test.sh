#!/bin/bash

BASE=prot/int

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv src/const.sv ${BASE}/mul.sv ${BASE}/mul_test.sv
./a.out
