#!/bin/bash

set -ex
set -o pipefail

iverilog -g2012 src/assert.sv src/float_params.sv src/lib/float_test_funcs.sv src/behav/float_test_funcs_test.sv
./a.out
