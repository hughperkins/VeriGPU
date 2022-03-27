#!/bin/bash
# run gate-level simulation on synth_out.sv

set -ex
set -o pipefail

python verigpu/run_yosys.py --in-verilog prot/synth_out.sv --top-module synth_out >/dev/null
iverilog -g2012 tech/osu018/osu018_stdcells.v build/netlist/6.v prot/synth_out_test.sv
./a.out
