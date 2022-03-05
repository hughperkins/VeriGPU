import argparse
import os


def int_to_binary(int_value, num_bits):
    return format(int_value, f'#0{num_bits + 2}b')[2:]


def hex_to_binary(hex_value, num_bits):
    assert hex_value.endswith('x')
    hex_value = hex_value[:-1]
    bits = int_to_binary(int(hex_value, 16), num_bits)
    assert len(bits) == num_bits
    return bits


def bits_to_hex(bits):
    return hex(int(bits, 2))[2:]


def run(args):
    if not os.path.isdir('build'):
        os.makedirs('build')
    with open(args.in_asm) as f:
        assembly = f.read()
    hex_lines = []
    for line in assembly.split('\n'):
        if line.strip() == '':
            continue

        split_line = line.split()
        cmd = split_line[0].lower()
        p1 = split_line[1] if len(split_line) >= 2 else None
        p2 = split_line[2] if len(split_line) >= 3 else None

        if cmd == 'out':
            print('p1', p1)
            if p1.endswith('x'):
                p1 = p1[:-1]
            else:
                raise ValueError("param " + p1 + " not recognized")
            imm_bits = int_to_binary(int(p1, 16), 7)
            op_bits = int_to_binary(1, 7)
            instr_bits = f'{imm_bits}{"0" * 18}{op_bits}'
            print('instr_bits', instr_bits)
            assert len(instr_bits) == 32
            hex_line = hex(int(instr_bits, 2))[2:]
            hex_lines.append(hex_line)
        elif cmd == 'outloc':
            imm_bits = hex_to_binary(p1, 7)
            print('p1', p1)
            print('imm_bits', imm_bits)
            op_bits = int_to_binary(2, 7)
            instr_bits = f'{imm_bits}{"0" * 18}{op_bits}'
            assert len(instr_bits) == 32
            hex_line = bits_to_hex(instr_bits)
            hex_lines.append(hex_line)
        elif cmd == 'li':
            assert p1.startswith('x')
            p1 = p1[1:]
            assert p2.endswith('x')
            p2 = p2[:-1]

            op_bits = int_to_binary(3, 7)
            imm_bits = int_to_binary(int(p2, 16), 7)
            assert len(imm_bits) == 7
            reg_select_bits = int_to_binary(int(p1, 16), 5)
            assert len(reg_select_bits) == 5
            instr_bits = f'{imm_bits}{"0" * 13}{reg_select_bits}{op_bits}'
            print(instr_bits, len(instr_bits))
            assert len(instr_bits) == 32
            hex_line = bits_to_hex(instr_bits)
            hex_lines.append(hex_line)
        elif cmd == 'outr':
            assert p1.startswith('x')
            rd_str = p1[1:] + 'x'
            rd_bits = hex_to_binary(rd_str, 5)
            op_bits = int_to_binary(4, 7)
            instr_bits = f'{"0" * 20}{rd_bits}{op_bits}'
            assert len(instr_bits) == 32
            hex_line = hex(int(instr_bits, 2))[2:]
            hex_lines.append(hex_line)
        elif cmd == 'half':
            assert p1.endswith('x')
            hex_line = p1[:-1]
            hex_lines.append('0000' + hex_line)
        elif cmd == 'word':
            assert p1.endswith('x')
            hex_line = p1[:-1]
            hex_lines.append(hex_line)
        elif cmd == 'halt':
            hex_lines.append('0000' + '0500')
        elif cmd.endswith(':'):
            cmd = cmd[:-1]
            assert cmd.endswith('x')
            cmd = cmd[:-1]
            location = int(cmd, 16) // 4
            while len(hex_lines) < location:
                hex_lines.append('00000000')
        else:
            raise Exception('cmd ' + cmd + ' not recognized')
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
