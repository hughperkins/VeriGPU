read_verilog build/netlist.v
link_design comp
create_clock -name clk -period 10
report_checks
