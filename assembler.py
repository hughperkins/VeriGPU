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
        cmd, p1 = line.split()
        cmd = cmd.lower()
        # print(cmd, p1)
        op = {
            'out': '01'
        }[cmd]
        # print('op', op)
        if p1.endswith('x'):
            p1 = p1[:-1]
        else:
            raise ValueError("param " + p1 + " not recognized")
        hex_line = op + p1
        # print('hex_line', hex_line)
        hex_lines.append(hex_line)
    with open(args.out_hex, 'w') as f:
        for hex_line in hex_lines:
            f.write(hex_line + '\n')
    print('wrote ' + args.out_hex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-asm', type=str, default='prog2.asm')
    parser.add_argument('--out-hex', type=str, default='build/prog2.hex')
    args = parser.parse_args()
    run(args)
