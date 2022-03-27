#!/bin/bash

BASE=prot/float

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv ${BASE}/float_params.sv \
    src/generated/mul_pipeline_cycle_24bit_2bpc.sv \
    ${BASE}/float_mul_pipeline5.sv ${BASE}/float_test_funcs.sv \
    ${BASE}/float_mul_pipeline_test.sv
./a.out
