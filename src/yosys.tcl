read_verilog -sv src/comp.sv
# read_verilog -sv mem.sv
# read_verilog -sv comp_driver.sv
# synth
write_rtlil
show -pause
