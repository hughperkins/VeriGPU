#!/bin/bash

BASE=prot/float

python toy_proc/timing.py --in-verilog src/assert_ignore.sv src/const.sv ${BASE}/float_params.sv \
    ${BASE}/float_mul_pipeline3.sv --top-module float_mul_pipeline
