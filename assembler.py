import argparse
import os


def run(args):
    if not os.path.isdir('build'):
        os.makedirs('build')
    with open(args.in_asm) as f:
        assembly = f.read()
    hex_lines = []
    for line in assembly.split('\n'):
        if line.strip() == '':
            continue
        # print(line)
        split_line = line.split()
        cmd = split_line[0].lower()
        p1 = split_line[1] if len(split_line) >= 2 else None
        p2 = split_line[2] if len(split_line) >= 3 else None
        # print(cmd, p1)
        if cmd == 'out':
            if p1.endswith('x'):
                p1 = p1[:-1]
            else:
                raise ValueError("param " + p1 + " not recognized")
            hex_line = '01' + p1
            hex_lines.append(hex_line)
        elif cmd == 'outloc':
            if p1.endswith('x'):
                p1 = p1[:-1]
            else:
                raise ValueError("param " + p1 + " not recognized")
            hex_line = '02' + p1
            hex_lines.append(hex_line)
        elif cmd == 'li':
            if p2.endswith('x'):
                p2 = p2[:-1]
            else:
                raise ValueError("param " + p2 + " not recognized")
            binary_op = '0011'
            assert p1.startswith('x') and len(p1) == 2
            reg_select = int(p1[1:])
            binary_reg = format(reg_select, '04b')
            print('binary reg', binary_reg, 'binary op', binary_op, 'p2', p2)
            hex_line = hex(int(binary_reg + binary_op, 2))[2:] + p2
            # hex_line = '03' + p1
            hex_lines.append(hex_line)
        elif cmd == 'outr':
            assert p1.startswith('x') and len(p1) == 2
            reg_select = p1[1:]
            hex_line = reg_select + '400'
            hex_lines.append(hex_line)
        elif cmd == 'short':
            assert p1.endswith('x')
            hex_line = p1[:-1]
            hex_lines.append(hex_line)
        elif cmd.endswith(':'):
            cmd = cmd[:-1]
            assert cmd.endswith('x')
            cmd = cmd[:-1]
            location = int(cmd, 16) // 2
            while len(hex_lines) < location:
                hex_lines.append('0000')
        else:
            raise Exception('cmd ' + cmd + ' not recognized')
        # print('hex_line', hex_line)
    with open(args.out_hex, 'w') as f:
        for hex_line in hex_lines:
            f.write(hex_line + '\n')
    with open(args.out_hex) as f:
        for line in f:
            print(line.strip())
    print('wrote ' + args.out_hex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-asm', type=str, default='prog6.asm')
    parser.add_argument('--out-hex', type=str, default='build/prog6.hex')
    args = parser.parse_args()
    run(args)
