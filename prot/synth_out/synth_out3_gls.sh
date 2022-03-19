#!/bin/bash
# run gate-level simulation on synth_out.sv

set -ex
set -o pipefail

iverilog -g2012 prot/synth_out3.sv prot/synth_out3_test.sv
./a.out

python toy_proc/run_yosys.py --in-verilog prot/synth_out3.sv --top-module synth_out3 >/dev/null
iverilog -g2012 tech/osu018/osu018_stdcells.v build/netlist/6.v prot/synth_out3_test.sv
./a.out
