#!/bin/bash

set -ex
set -o pipefail

iverilog -Wall -g2012 src/assert.sv src/mem_delayed_small.sv src/const.sv src/mem_delayed.sv test/mem_delayed_test.sv
./a.out
