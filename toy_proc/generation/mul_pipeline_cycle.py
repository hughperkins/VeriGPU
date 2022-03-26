"""
generates combinatorial logic for a single clock cycle of a 24-bit multiplier,
for float multiplication

We can flatten the carry completely at the end, or avoid doing the final carry
layer. The latter is probably faster, but harder, so I will flatten completely to start with.

For 24 bit, 1 bit per cycle
Max propagation delay: 59.4 nand units
Area:                  941.5 nand units

for 32-bit, 1 bit per cycle:
Max propagation delay: 63.2 nand units
Area:                  1244.0 nand units

for 32-bit, 2 bit per cycle:

for 32-bit, 4 bit per cycle:

"""
import math
import argparse
from collections import defaultdict, deque


"""
task mul_partial_add_task(
    input [$clog2(width + 1):0] pos,
    input [width - 1:0] a,
    input [width - 1:0] b,
    input [$clog2(width):0] cin,
    output reg [bits_per_cycle - 1:0] sum,
    output reg [$clog2(width):0] cout
);
    reg rst;
    reg [width * 2 - 1:0] a_shifted;

    assign rst = 0;
    // cout = '0;
    // {cout, sum} = cin;
    // `assert_known(b);
    // `assert_known(pos);
    // a_shifted = '0;
    assign a_shifted = a << (width - pos);
    // for(int i = 0; i < width; i++) begin  // iterate through b
    //     {cout, sum} = {cout, sum} + (a_shifted[width + bits_per_cycle - 1 - i -: bits_per_cycle] & {bits_per_cycle{b[i]}} );
    // end
endtask
"""


def log2_ceil(value):
    return int(math.ceil(math.log(value) / math.log(2)))


def run(args):
    carry_width = 1
    while log2_ceil(args.width + carry_width) > carry_width:
        carry_width += 1
    print('carry_width', carry_width)
    log2_bits_per_cycle = log2_ceil(args.bits_per_cycle)
    assert int(math.pow(2, log2_bits_per_cycle)) == args.bits_per_cycle
    dots = defaultdict(deque)
    lines = deque()
    lines.extend(f"""// This is a GENERATED file. Do not modify by hand.
// Created by toy_proc/generation/mul_pipeline_cycle.py

module {args.module_name}(
    input [{log2_ceil(args.width * 2) - 1}:0] pos,
    input [{args.width - 1}:0] {args.a_name},
    input [{args.width - 1}:0] {args.b_name},
    input [{carry_width - 1}:0] cin,
    output [{args.bits_per_cycle - 1}:0] sum,
    output [{carry_width - 1}:0] cout
);
    wire rst;
    wire [{args.width * 2 - 1}:0] a_;

    assign rst = 0;
    //cout = '0;
    // {{ cout, sum }} = cin;
    // `assert_known(b);
    // `assert_known(pos);
    // a_ = '0;
    assign a_ = a << ({args.width} - pos);
""".split('\n'))

    for line in lines:
        print(line)

    wires = []
    assigns = []

    # add { cout, sum } to dots
    # add cin to dots
    # for i in range(args.bits_per_cycle):
    #     dots[i].append(f'sum[{i}]')
    # for i in range(args.bits_per_cycle, carry_width):
    #     dots[i].append(f'cout[{i - args.bits_per_cycle}]')
    for i in range(carry_width):
        dots[i].append(f'cin[{i}]')

    # partial products to dots
    # code we are replacing ours with:
    # for(int i = 0; i < width; i++) begin  // iterate through b
    #     {cout, sum} = {cout, sum} + (a_shifted[width + bits_per_cycle - 1 - i -: bits_per_cycle] & {bits_per_cycle{b[i]}} );
    # end
    for i in range(args.width):
        for j in range(args.bits_per_cycle):
            dots[j].append(f'(b[{i}] & a_[{args.width + args.bits_per_cycle - 1 - i - j}])')

    for i, col in sorted(dots.items()):
        print(i)
        for entry in col:
            print('    ', entry)

    # for i in range(args.width):
    #     for j in range(args.width):
    #         p = i + j
    #         # lines.append('')
    #         # if p >= args.pos and p < args.pos + args.bits_per_cycle:
    #         if p >= 0 and p < args.bits_per_cycle:
    #             dots[p].append(f'({args.a_name}[{i}] & {args.b_name}[{j}])')
    print('dots.keys()', list(dots.keys()))
    for i in range(carry_width):
        dots[i].append(f'cin[{i}]')
    wire_index = 0
    while(True):
        this_chain = ''
        for i, col in sorted(dots.items()):
            print(i, col)
        max_height = max([len(col) for col in dots.values()])
        print('max_height', max_height)
        if max_height == 2:
            # we are done. hand-off to carry adder...
            break
        dots_new = defaultdict(deque)
        for i in range(len(dots)):
            while len(dots[i]) >= 2:
                if len(dots[i]) == 2:
                    this_chain += '-'
                    carry_name = f'wire_{wire_index}'
                    sum_name = f'wire_{wire_index + 1}'
                    wire_index += 2
                    wires.append(f'    wire {carry_name};')
                    wires.append(f'    wire {sum_name};')
                    line = f'    assign {{ {carry_name}, {sum_name} }} = {dots[i][-1]} + {dots[i][-2]};'
                    assigns.append(line)
                    dots_new[i + 1].appendleft(carry_name)
                    dots[i].pop()
                    dots[i].pop()
                    dots_new[i].appendleft(sum_name)
                else:
                    this_chain += '+'
                    carry_name = f'wire_{wire_index}'
                    sum_name = f'wire_{wire_index + 1}'
                    wire_index += 2
                    wires.append(f'    wire {carry_name};')
                    wires.append(f'    wire {sum_name};')
                    line = f'    assign {{ {carry_name}, {sum_name} }} = {dots[i][-1]} + {dots[i][-2]} + {dots[i][-3]};'
                    assigns.append(line)
                    dots_new[i + 1].appendleft(carry_name)
                    dots[i].pop()
                    dots[i].pop()
                    dots[i].pop()
                    dots_new[i].appendleft(sum_name)
            dots_new[i].extend(dots[i])
        dots = dots_new
        print(this_chain)

    # final carried add...
    this_chain = ''
    for i in range(len(dots)):
        while len(dots[i]) >= 2:
            if len(dots[i]) == 2:
                this_chain += '-'
                carry_name = f'wire_{wire_index}'
                sum_name = f'wire_{wire_index + 1}'
                wire_index += 2
                wires.append(f'    wire {carry_name};')
                wires.append(f'    wire {sum_name};')
                p1 = dots[i].pop()
                p2 = dots[i].pop()
                line = f'    assign {{ {carry_name}, {sum_name} }} = {p1} + {p2};'
                assigns.append(line)
                dots[i + 1].appendleft(carry_name)
                dots[i].appendleft(sum_name)
            else:
                this_chain += '+'
                carry_name = f'wire_{wire_index}'
                sum_name = f'wire_{wire_index + 1}'
                wire_index += 2
                wires.append(f'    wire {carry_name};')
                wires.append(f'    wire {sum_name};')
                p1 = dots[i].pop()
                p2 = dots[i].pop()
                p3 = dots[i].pop()
                line = f'    assign {{ {carry_name}, {sum_name} }} = {p1} + {p2} + {p3};'
                assigns.append(line)
                dots[i + 1].appendleft(carry_name)
                dots[i].appendleft(sum_name)
    print(this_chain)

    out_term = '{' + ', '.join([
        dots[i][0] for i in range(len(dots) - 1, -1, -1)]) + '}'
    assigns.append(f"    assign {{ cout, sum }} = {out_term};")

    for wire in wires:
        lines.append(wire)
    for assign in assigns:
        lines.append(assign)

    lines.append('endmodule')

    with open(args.out_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print('wrote to ' + args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--bits-per-cycle', type=int, default=2, help='should be a power of 2')
    parser.add_argument('--module-name', type=str, default='dadda')
    parser.add_argument('--a-name', type=str, default='a')
    parser.add_argument('--b-name', type=str, default='b')
    parser.add_argument('--out-name', type=str, default='out')
    parser.add_argument('--out-path', type=str, default='build/mul_pipeline_cycle_{width}bit.sv')
    args = parser.parse_args()
    args.out_path = args.out_path.format(**args.__dict__)
    run(args)
