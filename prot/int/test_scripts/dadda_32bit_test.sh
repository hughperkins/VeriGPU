#!/bin/bash

BASE=prot/int

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv src/const.sv src/generated/dadda_32bit.sv ${BASE}/dadda_32bit_test.sv
./a.out
