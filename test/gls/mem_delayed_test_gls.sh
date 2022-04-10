#!/bin/bash

# run gate-level simulation on src/mem_delayed.sv

set -ex
set -o pipefail

if [[ -f build/netlist/4.v ]]; then {
    rm build/netlist/4.v
} fi
if [[ -f build/netlist/6.v ]]; then {
    rm build/netlist/6.v
} fi

python verigpu/run_yosys.py --in-verilog src/mem_small.sv \
    src/assert_ignore.sv src/const.sv src/global_mem_controller.sv \
    --top-module global_mem_controller >/dev/null

# cat <<EOF > build/insert_monitor.txt
#   always @(*) begin
#     \$monitor("t=%0d 6.v.always* rst=%0d ena=%0d rd_req=%0d wr_req=%0d addr=%0d rd_data=%0d wr_data=%0d busy=%0d ack=%0d clks_to_wait=%0d",
#       \$time, rst, ena, rd_req, wr_req, addr, rd_data, wr_data, busy, ack, clks_to_wait);
#   end
# EOF
# sed -i -e '/_0000_/r build/insert_monitor.txt' build/netlist/6.v

iverilog -Wall -g2012 tech/osu018/osu018_stdcells.v build/netlist/6.v src/const.sv src/assert.sv \
    src/mem_small.sv test/mem_delayed_test.sv
./a.out 
