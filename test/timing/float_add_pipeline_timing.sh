#!/bin/bash

set -ex
set -o pipefail

python verigpu/timing.py --in-verilog src/assert_ignore.sv src/const.sv src/float/float_params.sv \
    src/float/float_add_pipeline.sv --top-module float_add_pipeline
