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
        # print(cmd, p1)
        if cmd == 'out':
            # op = {
            #     'out': '01'
            # }[cmd]
            # print('op', op)
            if p1.endswith('x'):
                p1 = p1[:-1]
            else:
                raise ValueError("param " + p1 + " not recognized")
            hex_line = '01' + p1
            hex_lines.append(hex_line)
        elif cmd == 'outloc':
            # op = {
            #     'out': '01'
            # }[cmd]
            # print('op', op)
            if p1.endswith('x'):
                p1 = p1[:-1]
            else:
                raise ValueError("param " + p1 + " not recognized")
            hex_line = '02' + p1
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
    parser.add_argument('--in-asm', type=str, default='prog3.asm')
    parser.add_argument('--out-hex', type=str, default='build/prog3.hex')
    args = parser.parse_args()
    run(args)
