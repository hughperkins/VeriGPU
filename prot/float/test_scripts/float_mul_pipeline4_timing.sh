#!/bin/bash

BASE=prot/float

python verigpu/timing.py --in-verilog src/assert_ignore.sv src/const.sv ${BASE}/float_params.sv \
    ${BASE}/chunks/mul_partial_add_task.sv \
    ${BASE}/float_mul_pipeline4.sv --top-module float_mul_pipeline
