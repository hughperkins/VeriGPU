read_verilog -sv src/proc.sv
read_verilog -sv src/mem.sv
read_verilog -sv src/comp.sv
hierarchy -top comp
#proc;;
#memory;;
#techmap;;
synth
write_rtlil
write_verilog build/netlist.v
