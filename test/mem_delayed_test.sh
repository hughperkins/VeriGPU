#!/bin/bash

set -ex

iverilog -g2012 src/const.sv src/mem_delayed.sv test/mem_delayed_test.sv && ./a.out
