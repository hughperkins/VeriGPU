#!/bin/bash

BASE=prot/float

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv ${BASE}/float_params.sv ${BASE}/float_add2.sv ${BASE}/float_test_funcs.sv ${BASE}/float_add_test.sv
./a.out
