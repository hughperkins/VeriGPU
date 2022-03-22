#!/bin/bash

BASE=prot/float

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv ${BASE}/float_params.sv ${BASE}/float_mul.sv test/lib/float_test_funcs.sv ${BASE}/float_mul_test.sv
./a.out
