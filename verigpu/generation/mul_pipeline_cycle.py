"""
generates combinatorial logic for a single clock cycle of a 24-bit multiplier,
for float multiplication

We can flatten the carry completely at the end, or avoid doing the final carry
layer. The latter is probably faster, but harder, so I will flatten completely to start with.

For 24 bit, 1 bit per cycle
Max propagation delay: 56.4 nand units
Area:                  610.5 nand units

For 24 bit, 2 bit per cycle
Max propagation delay: 59.4 nand units
Area:                  941.5 nand units

For 24 bit, 4 bit per cycle
Max propagation delay: 68.2 nand units
Area:                  1501.5 nand units

for 32-bit, 1 bit per cycle:
Max propagation delay: 58.2 nand units
Area:                  862.5 nand units

for 32-bit, 2 bit per cycle:
Max propagation delay: 63.2 nand units
Area:                  1244.0 nand units

for 32-bit, 4 bit per cycle:
Max propagation delay: 74.8 nand units
Area:                  2018.0 nand units

32-bit, 8 bits per cycle:
Max propagation delay: 93.2 nand units
Area:                  3428.5 nand units
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
    assign a_shifted = a << (width - pos);
endtask
"""


def log2_ceil(value):
    return int(math.ceil(math.log(value) / math.log(2)))


def run(args):
    prev_carry = 0
    new_carry = None
    while new_carry != prev_carry:
        new_sum = prev_carry
        for i in range(args.bits_per_cycle):
            new_sum += args.width * int(math.pow(2, i))
        print('new_sum', new_sum)
        # new_sum = args.width + args.width * 2 + prev_carry
        new_carry = new_sum // int(math.pow(2, args.bits_per_cycle))
        print('new_carry', new_carry)
        # import time
        # time.sleep(1)
        prev_carry = new_carry
    carry_width = log2_ceil(new_carry) + 1
    print('carry_width', carry_width)
    carry_width = 1
    # we have to add one to args.width because of 1 from carry
    while log2_ceil(args.width + 1 + carry_width) > carry_width:
        carry_width += 1
    print('carry_width', carry_width)
    log2_bits_per_cycle = log2_ceil(args.bits_per_cycle)
    assert int(math.pow(2, log2_bits_per_cycle)) == args.bits_per_cycle
    dots = defaultdict(deque)
    dots_path = defaultdict(deque)  # store where each value in dots came from, the path
    lines = deque()
    lines.extend(f"""// This is a GENERATED file. Do not modify by hand.
// Created by verigpu/generation/mul_pipeline_cycle.py

task {args.task_name}(
    input [{log2_ceil(args.width * 2) - 1}:0] pos,
    input [{args.width - 1}:0] {args.a_name},
    input [{args.width - 1}:0] {args.b_name},
    input [{carry_width - 1}:0] cin,
    output reg [{args.bits_per_cycle - 1}:0] sum,
    output reg [{carry_width - 1}:0] cout
);
    reg rst;
    reg [{args.width * 2 - 1}:0] a_;
""".split('\n'))

    for line in lines:
        print(line)

    wires = []
    assigns = []

    for i in range(carry_width):
        dots[i].append(f'cin[{i}]')
        dots_path[i].append(dots[i][-1])

    print('')
    for i, col in sorted(dots.items()):
        print('col', i, 'len', len(col))

    # partial products to dots
    # code we are replacing ours with:
    # for(int i = 0; i < width; i++) begin  // iterate through b
    #     {cout, sum} = {cout, sum} + (
    #         a_shifted[width + bits_per_cycle - 1 - i -: bits_per_cycle] & {bits_per_cycle{b[i]}} );
    # end
    for i in range(args.width):
        for j in range(args.bits_per_cycle):
            dots[j].append(f'(b[{i}] & a_[{args.width + args.bits_per_cycle - 1 - i - j}])')
            dots_path[j].append(dots[j][-1])
    for i, col in sorted(dots.items()):
        print('i', i)
        for term in col:
            print('    ', term)
    print('')
    print('dots path')
    for i, col in sorted(dots_path.items()):
        print('i', i)
        for term in col:
            print('    ', term)

    print('')
    for i, col in sorted(dots.items()):
        print('col', i, 'len', len(col))

    # for i, col in sorted(dots.items()):
    #     print(i)
    #     for entry in col:
    #         print('    ', entry)

    # print('dots.keys()', list(dots.keys()))
    # for i in range(carry_width):
    #     dots[i].append(f'cin[{i}]')
    wire_index = 0
    while(True):
        this_chain = ''
        for i, col in sorted(dots.items()):
            print('i', i, 'len(col)', len(col))
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
                    carry_name = f'carry_{wire_index}'
                    sum_name = f'sum_{wire_index + 1}'
                    wire_index += 2
                    wires.append(f'    reg {carry_name};')
                    wires.append(f'    reg {sum_name};')
                    i1 = dots[i].pop()
                    i2 = dots[i].pop()
                    line = f'    {{ {carry_name}, {sum_name} }} = {i1} + {i2};'
                    assigns.append(line)
                    dots_new[i + 1].appendleft(carry_name)
                    dots_new[i].appendleft(sum_name)
                    p1 = dots_path[i].pop()
                    p2 = dots_path[i].pop()
                    dots_path[i].appendleft(f'({p1} + {p2})')
                    dots_path[i + 1].appendleft(f'(({p1} + {p2}) // 2)')
                else:
                    this_chain += '+'
                    carry_name = f'carry_{wire_index}'
                    sum_name = f'sum_{wire_index + 1}'
                    wire_index += 2
                    wires.append(f'    reg {carry_name};')
                    wires.append(f'    reg {sum_name};')
                    i1 = dots[i].pop()
                    i2 = dots[i].pop()
                    i3 = dots[i].pop()
                    line = f'    {{ {carry_name}, {sum_name} }} = {i1} + {i2} + {i3};'
                    assigns.append(line)
                    dots_new[i + 1].appendleft(carry_name)
                    dots_new[i].appendleft(sum_name)
                    p1 = dots_path[i].pop()
                    p2 = dots_path[i].pop()
                    p3 = dots_path[i].pop()
                    dots_path[i].appendleft(f'({p1} + {p2} + {p3})')
                    dots_path[i + 1].appendleft(f'({p1} + {p2} + {p3}) // 2)')
            dots_new[i].extend(dots[i])
        dots = dots_new
        print('len(dots)', len(dots))
        print(this_chain)
        print('')
        print('dots:')
        for i, col in sorted(dots.items()):
            print('i', i)
            for term in col:
                print('    ', term)

        print('')
        print('dots_path:')
        for i, col in sorted(dots_path.items()):
            print('i', i)
            if i != 1:
                continue
            for term in col:
                print('    ', term)

    # final carried add...
    this_chain = ''
    for i in range(len(dots)):
        while len(dots[i]) >= 2:
            if len(dots[i]) == 2:
                this_chain += '-'
                carry_name = f'carry_{wire_index}'
                sum_name = f'sum_{wire_index + 1}'
                wire_index += 2
                wires.append(f'    reg {carry_name};')
                wires.append(f'    reg {sum_name};')
                p1 = dots[i].pop()
                p2 = dots[i].pop()
                line = f'    {{ {carry_name}, {sum_name} }} = {p1} + {p2};'
                assigns.append(line)
                dots[i + 1].appendleft(carry_name)
                dots[i].appendleft(sum_name)
            else:
                this_chain += '+'
                carry_name = f'carry_{wire_index}'
                sum_name = f'sum_{wire_index + 1}'
                wire_index += 2
                wires.append(f'    reg {carry_name};')
                wires.append(f'    reg {sum_name};')
                p1 = dots[i].pop()
                p2 = dots[i].pop()
                p3 = dots[i].pop()
                line = f'    {{ {carry_name}, {sum_name} }} = {p1} + {p2} + {p3};'
                assigns.append(line)
                dots[i + 1].appendleft(carry_name)
                dots[i].appendleft(sum_name)
    print(this_chain)
    print('len(dots)', len(dots))

    out_term = '{' + ', '.join([
        dots[i][0] for i in range(len(dots) - 1, -1, -1)]) + '}'
    assigns.append(f"    {{ cout, sum }} = {out_term};")

    lines.extend(wires)
    lines.append("    rst = 0;")
    lines.append(f"    a_ = a << ({args.width} - pos);")
    lines.extend(assigns)

    lines.append('endtask')

    with open(args.out_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    print('wrote to ' + args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--bits-per-cycle', type=int, default=1, help='should be a power of 2')
    parser.add_argument('--out-dir', type=str, default='build')
    parser.add_argument('--task-name', type=str, default='mul_pipeline_cycle_{width}bit_{bits_per_cycle}bpc')
    parser.add_argument('--a-name', type=str, default='a')
    parser.add_argument('--b-name', type=str, default='b')
    parser.add_argument('--out-name', type=str, default='out')
    parser.add_argument(
        '--out-path', type=str,
        default='{out_dir}/{task_name}.sv')
    args = parser.parse_args()
    args.task_name = args.task_name.format(**args.__dict__)
    args.out_path = args.out_path.format(**args.__dict__)
    run(args)
