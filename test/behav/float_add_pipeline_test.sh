#!/bin/bash

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv src/float_params.sv src/float_add_pipeline.sv test/lib/float_test_funcs.sv \
    test/behav/float_add_pipeline_test.sv
./a.out
