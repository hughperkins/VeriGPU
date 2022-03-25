#!/bin/bash

BASE=prot/int

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv src/const.sv src/generated/dadda_24bit48.sv ${BASE}/dadda_24bit_test.sv
./a.out
