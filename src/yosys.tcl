read_verilog -sv src/mem.sv
# read_verilog -sv src/mem.sv
# read_verilog -sv comp_driver.sv
synth
write_rtlil
show
