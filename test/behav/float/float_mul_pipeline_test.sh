#!/bin/bash

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv src/float/float_params.sv \
    src/generated/mul_pipeline_cycle_24bit_2bpc.sv \
    src/float/float_mul_pipeline.sv test/lib/float_test_funcs.sv \
    test/behav/float/float_mul_pipeline_test.sv
./a.out
