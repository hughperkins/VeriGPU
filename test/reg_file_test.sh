#!/bin/bash

set -ex
set -o pipefail

iverilog -g2012 src/reg_file.sv src/const.sv test/reg_file_test.sv && ./a.out
