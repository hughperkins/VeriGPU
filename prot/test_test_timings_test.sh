#!/bin/bash

set -ex
set -o pipefail

iverilog -g2012 -Wall src/assert.sv prot/test_test_timings.sv prot/test_test_timings_test.sv
./a.out 
