import argparse
import os
from collections import deque


def int_to_binary(int_value, num_bits):
    return format(int_value, f'#0{num_bits + 2}b')[2:]


def hex_to_binary(hex_value, num_bits):
    assert hex_value.startswith('0x')
    hex_value = hex_value[2:]
    bits = int_to_binary(int(hex_value, 16), num_bits)
    assert len(bits) == num_bits
    return bits


def int_str_to_int(int_str):
    if int_str.startswith('0x'):
        int_value = int(int_str[2:], 16)
    else:
        int_value = int(int_str)
    return int_value


def int_str_to_bits(int_str, num_bits):
    int_value = int_str_to_int(int_str)
    bits = int_to_binary(int_value, num_bits)
    assert len(bits) == num_bits
    return bits


def reg_str_to_bits(reg_str, num_bits: int = 5):
    assert reg_str.startswith('x')
    reg_str = reg_str[1:]
    bits = int_str_to_bits(reg_str, num_bits=num_bits)
    return bits


def bits_to_hex(bits, num_bytes: int = 4):
    hex_str = hex(int(bits, 2))[2:]
    hex_str = hex_str.rjust(num_bytes * 2, "0")
    assert len(hex_str) == num_bytes * 2
    return hex_str


def run(args):
    if not os.path.isdir('build'):
        os.makedirs('build')
    with open(args.in_asm) as f:
        assembly = f.read()
    asm_cmds = deque(assembly.split('\n'))
    hex_lines = []
    while len(asm_cmds) > 0:
        line = asm_cmds.popleft()
        if line.strip() == '' or line.strip().startswith('#'):
            continue

        line = line.replace(',', ' ').replace("(", " ").replace(")", " ").replace(
            '  ', ' ').replace("  ", " ")
        split_line = line.split()
        cmd = split_line[0].lower()
        p1 = split_line[1] if len(split_line) >= 2 else None
        p2 = split_line[2] if len(split_line) >= 3 else None
        p3 = split_line[3] if len(split_line) >= 4 else None

        try:
            if cmd == 'sw':
                # e.g.
                # sw x2,  0      (x3)
                #    rs2  offset rs1
                print('sw', p1, p2, p3)
                op_bits = "0100011"
                rs1_bits = reg_str_to_bits(p3)
                rs2_bits = reg_str_to_bits(p1)
                # offset_int = int_str_to_int(p2)
                offset_bits = int_str_to_bits(p2, 12)
                offset1_bits = offset_bits[:7]
                offset2_bits = offset_bits[7:]
                # assert offset_int == 0
                instr_bits = f'{offset1_bits}{rs2_bits}{rs1_bits}010{offset2_bits}{op_bits}'
                hex_lines.append(bits_to_hex(instr_bits))
            elif cmd == 'lw':
                # e.g.
                # lw x2,  0      (x3)
                #    rd   offset rs1
                print('lw', p1, p2, p3)
                op_bits = "0000011"
                rs1_bits = reg_str_to_bits(p3)
                rd_bits = reg_str_to_bits(p1)
                offset_bits = int_str_to_bits(p2, 12)
                instr_bits = f'{offset_bits}{rs1_bits}010{rd_bits}{op_bits}'
                hex_lines.append(bits_to_hex(instr_bits))
            elif cmd == 'addi':
                print('addi', p1, p2, p3)
                # e.g.
                # addi x1,    x2,    123
                #      rd     rs1    imm
                op_bits = "0010011"
                imm_bits = int_str_to_bits(p3, 12)
                rd_bits = reg_str_to_bits(p1)
                rs1_bits = reg_str_to_bits(p2)
                funct_bits = '000'
                instr_bits = f'{imm_bits}{rs1_bits}{funct_bits}{rd_bits}{op_bits}'
                hex_lines.append(bits_to_hex(instr_bits))
            elif cmd == 'out':
                # e.g.: out 0x1b
                # virtual instruction. map to storing to location 1000

                # migrate to use store at location 1000
                # need to push store in first, since we are pushing in reverse order
                asm_cmds.appendleft('sw x30, 0(x31)')
                asm_cmds.appendleft(f'addi x30, x0, {p1}')
                asm_cmds.appendleft('addi x31, x0, 1000')
                continue
            elif cmd == 'outloc':
                # e.g. outloc 0x20
                # virtual command, maps to li followed by sw to location 1000
                asm_cmds.appendleft('sw x30, 0(x31)')
                asm_cmds.appendleft('addi x31, x0, 1000')
                asm_cmds.appendleft(f'lw x30, 0(x31)')
                asm_cmds.appendleft(f'li x31, {p1}')
                continue
            elif cmd == 'li':
                # e.g.: li x1 0x12
                # virtual command; convert to e.g. addi x1, x0, 0x12

                asm_cmds.appendleft(f'addi {p1}, x0, {p2}')
                continue
            elif cmd == 'outr':
                # virtual instruction. map to storing to location 1000

                # migrate to use store at location 1000, with load before that
                # need to push store in first, since we are pushing in reverse order

                asm_cmds.appendleft(f'sw {p1}, 0(x31)')
                asm_cmds.appendleft('addi x31, x0, 1000')
                continue
            elif cmd == 'half':
                bits = int_str_to_bits(p1, 16)
                hex_lines.append("0000" + bits_to_hex(bits, num_bytes=2))
            elif cmd == 'word':
                bits = int_str_to_bits(p1, 32)
                assert len(bits) == 32
                hex_lines.append(bits_to_hex(bits, num_bytes=4))
            elif cmd == 'halt':
                # virtual instruction
                # write to location 1001 instead
                asm_cmds.appendleft('sw x30, 0(x31)')
                asm_cmds.appendleft('addi x31, x0, 1004')
                continue
            elif cmd.endswith(':'):
                cmd = cmd[:-1]
                loc_int = int_str_to_int(cmd)
                location = loc_int // 4
                assert len(hex_lines) <= location
                while len(hex_lines) < location:
                    hex_lines.append('00000000')
            else:
                raise Exception('cmd ' + cmd + ' not recognized')
        except Exception as e:
            print('cmd:', line)
            raise e
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
