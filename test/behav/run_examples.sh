#!/bin/bash

# run all the examples, in examples folder
# each example comprises an assembler program, that we can compile, and feed to a simulator
# running our processor/gpu verilog

set -e
set -x
set -o pipefail

files=$(ls -b examples/direct/*.asm)

for prog in ${files}; do {
    python run.py --name ${prog}
} done
