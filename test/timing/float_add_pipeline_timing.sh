#!/bin/bash

python toy_proc/timing.py --in-verilog src/assert_ignore.sv src/const.sv src/float_params.sv \
    src/float_add_pipeline.sv --top-module float_add_pipeline
