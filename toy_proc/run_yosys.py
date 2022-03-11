#!/usr/bin/env python3
# helper tool to run yosys, generating appropriate yosys script
# python, rather than bash, since commandline arguments etc
# so much more convenenient in python
import argparse
import os


def run(args):
    with open('build/yosys.tcl', 'w') as f:
        f.write(f"""
read_verilog -sv {args.verilog}
synth
dfflibmap -liberty {args.cell_lib}
abc -liberty {args.cell_lib}
clean

write_rtlil build/rtlil.rtl
write_verilog build/netlist.v
ltp
sta
stat
""")
        if args.show:
            f.write('show\n')
    if os.system('yosys -s build/yosys.tcl') != 0:
        raise Exception("Failed")
    # os.system('subl build/rtlil.rtl')
    # os.system('subl build/netlist.v')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verilog', type=str, required=True, help='path to verilog file')
    parser.add_argument('--show', action='store_true', help='show xdot on the result')
    parser.add_argument(
        '--cell-lib', type=str, default='tech/osu018/osu018_stdcells.lib',
        help='e.g. path to osu018_stdcells.lib')
    args = parser.parse_args()
    run(args)