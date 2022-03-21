#!/bin/bash

BASE=prot/float

python toy_proc/timing.py --in-verilog src/assert.sv src/const.sv ${BASE}/float_params.sv \
    ${BASE}/float_add2.sv --top-module float_add
