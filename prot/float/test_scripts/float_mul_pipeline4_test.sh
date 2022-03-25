#!/bin/bash

BASE=prot/float

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv ${BASE}/float_params.sv ${BASE}/float_mul_pipeline4.sv ${BASE}/float_test_funcs.sv \
    ${BASE}/chunks/mul_partial_add_task.sv \
    ${BASE}/float_mul_pipeline_test.sv
./a.out
