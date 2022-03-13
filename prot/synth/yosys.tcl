read_verilog -sv src/proc.sv
# read_verilog -sv src/mem.sv
# read_verilog -sv src/comp.sv
hierarchy -top proc
#proc;;
#memory;;
#techmap;;
synth

dfflibmap -liberty /Users/hp/git/OpenTimer//example/simple/osu018_stdcells.lib
abc -liberty /Users/hp/git/OpenTimer//example/simple/osu018_stdcells.lib
clean

write_rtlil
write_verilog build/netlist.v
# show
