import argparse
import os
import math
from collections import deque


op_bits_by_op = {
    'STORE':    '0100011',
    'OPIMM':    '0010011',
    'LOAD':     '0000011',
    'BRANCH':   '1100011',
    'OP':       '0110011',
    'LUI':      '0110111',
    'MADD':     '1000011',
    'LOADFP':   '0000111',
    'STOREFP':  '0100111',
    'MSUB':     '1000111',
    'JALR':     '1100111',
    'NMSUB':    '1001011',
    'NMADD':    '1001111',
    'OPFP':     '1010011',
    'AUIPC':    '0010111',
    'OP32':     '0111011',
    'OPIMM32':  '0011011'
}


funct_bits_branch = {
    'BEQ':  'b000',
    'BNE':  'b001',
    'BLT':  'b100',
    'BGE':  'b101',
    'BLTU': 'b110',
    'BGEU': 'b111'
}


funct_bits_op = {
    'ADD':    '0000000000',
    'SLT':    '0000000010',
    'SLTU':   '0000000011',
    'AND':    '0000000111',
    'OR':     '0000000110',
    'XOR':    '0000000100',
    'SLL':    '0000000001',
    'SRL':    '0000000101',
    'SUB':    '0100000000',
    'SRA':    '0100000101',

    # RV32M extension:
    'MUL':    '0000001000',
    'MULH':   '0000001001',
    'MULHSU': '0000001010',
    'MULHU':  '0000001011',
    'DIV':    '0000001100',
    'DIVU':   '0000001101',
    'REM':    '0000001110',
    'REMU':   '0000001111'
}


funct_bits_opimm = {
    'ADDI':   '000',
    'SLTI':   '010',
    'SLTIU':  '011',
    'XORI':   '100',
    'ORI':    '110',
    'ANDI':   '111',
    'SLLI':   '001',
    'SRLI':   '010',
    'SRAI':   '101'
}


def int_to_binary(int_value, num_bits):
    if int_value < 0:
        offset = int(math.pow(2, num_bits))
        int_value += offset
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
    elif int_str.startswith('0b'):
        int_value = int(int_str[2:], 2)
    else:
        int_value = int(int_str)
    return int_value


def int_str_to_bits(int_str, num_bits):
    int_value = int_str_to_int(int_str)
    bits = int_to_binary(int_value, num_bits)
    assert len(bits) == num_bits
    return bits


def float_to_bits(float_value, exp_bits: int = 8, sig_bits: int = 23):
    # bits_check = ''.join([format(byte, '#010b')[2:] for byte in list(struct.pack('!f', 23.567))])
    sign_bit = 1 if float_value < 0 else 0
    if float_value == 0:
        return '0' * 32
    elif float_value != float_value:
        return '0' + '1' * 31
    else:
        float_value = float_value if float_value >= 0 else - float_value
        exp = 0
        signif = float_value
        while signif >= 2:
            signif /= 2
            exp += 1
        while signif < 1:
            signif *= 2
            exp -= 1
        exp_to_store = exp + 127
        exp_bits = int_to_binary(exp_to_store, 8)
        signif_to_store = (signif - 1) * math.pow(2, 23)
        signif_bits = int_to_binary(int(signif_to_store), 23)
    return f'{sign_bit}{exp_bits}{signif_bits}'
    # return bits


def numeric_str_to_bits(numeric_str, num_bits):
    if '.' in numeric_str:
        float_value = float(numeric_str)
        print('float_value', float_value)
        bits = float_to_bits(float_value, exp_bits=8, sig_bits=23)
    else:
        int_value = int_str_to_int(numeric_str)
        bits = int_to_binary(int_value, num_bits)
    assert len(bits) == num_bits
    return bits


def reg_str_to_bits(reg_str, num_bits: int = 5):
    assert reg_str.startswith('x')
    assert len(reg_str) >= 2
    reg_str = reg_str[1:]
    bits = int_str_to_bits(reg_str, num_bits=num_bits)
    return bits


def bits_to_hex(bits, num_bytes: int = 4):
    hex_str = hex(int(bits, 2))[2:]
    hex_str = hex_str.rjust(num_bytes * 2, "0")
    assert len(hex_str) == num_bytes * 2
    return hex_str


# def bits_to_int(bits: str) -> int:
#     return int(bits, 2)


def word_bits_to_lui_addi_bits(word_bits: str):
    """
    eg given 00000111010110111100110100010101
    ew have upper bits 00000111010110111100
    and lower bits     110100010101
    ... but the lower bits cannot exceed 2048,
    so we need to subtract 4096 from lower bits,
    and add one to upper bits
    """
    upper_bits = word_bits[:20]
    lower_bits = word_bits[20:]
    upper_int = int(upper_bits, 2)
    lower_int = int(lower_bits, 2)
    if lower_int > 2047:
        lower_int -= 4096
        upper_int += 1
    upper_bits2 = int_to_binary(upper_int, 20)
    lower_bits2 = int_to_binary(lower_int, 12)
    return upper_bits2, lower_bits2


def run(args):
    if not os.path.isdir('build'):
        os.makedirs('build')
    with open(args.in_asm) as f:
        assembly = f.read()
    asm_cmds = deque(assembly.split('\n'))

    # first we run a pass to expand pseudocommands, and record the locations of labels
    # then we run the assembly pass

    label_pos_by_name = {}
    new_asm_cmds = deque()
    while len(asm_cmds) > 0:
        line = asm_cmds.popleft()
        if line.strip() == '' or line.strip().startswith('#') or line.strip().startswith(';'):
            continue

        line = line.split(';')[0]
        line = line.replace(',', ' ').replace("(", " ").replace(")", " ").replace(
            '  ', ' ').replace("  ", " ")
        split_line = line.split()
        cmd = split_line[0].lower()
        p1 = split_line[1] if len(split_line) >= 2 else None
        p2 = split_line[2] if len(split_line) >= 3 else None
        p3 = split_line[3] if len(split_line) >= 4 else None

        try:
            if cmd == 'li':
                # e.g.: li x1 0x12
                # virtual command; convert to e.g. addi x1, x0, 0x12

                if '.' not in p2:
                    # not float
                    imm_int = int_str_to_int(p2)
                    if abs(imm_int) < 2048:
                        # small ints can be loaded with single addi
                        asm_cmds.appendleft(f'addi {p1}, x0, {p2}')
                        continue
                    imm_bits = int_str_to_bits(p2, 32)
                else:
                    # float
                    imm_bits = numeric_str_to_bits(p2, 32)

                lui_bits, addi_bits = word_bits_to_lui_addi_bits(imm_bits)

                asm_cmds.appendleft(f'addi {p1}, {p1}, 0b{addi_bits}')
                asm_cmds.appendleft(f'lui {p1}, 0b{lui_bits}')
                continue
            elif cmd == 'out':
                # e.g.: out 0x1b
                # virtual instruction. map to storing to location 1000

                # migrate to use store at location 1000
                # need to push store in first, since we are pushing in reverse order
                asm_cmds.appendleft('sw x30, 0(x31)')
                asm_cmds.appendleft(f'addi x30, x0, {p1}')
                asm_cmds.appendleft('addi x31, x0, 1000')
                continue
            elif cmd == 'outr':
                # virtual instruction. map to storing to location 1000

                # migrate to use store at location 1000, with load before that
                # need to push store in first, since we are pushing in reverse order

                asm_cmds.appendleft(f'sw {p1}, 0(x31)')
                asm_cmds.appendleft('addi x31, x0, 1000')
                continue
            elif cmd == 'outr.s':
                # virtual instruction. map to storing to location 1008

                # migrate to use store at location 1000, with load before that
                # need to push store in first, since we are pushing in reverse order

                asm_cmds.appendleft(f'sw {p1}, 0(x31)')
                asm_cmds.appendleft('addi x31, x0, 1008')
                continue
            elif cmd == 'outloc':
                # e.g. outloc 0x20
                # virtual command, maps to li followed by sw to location 1000
                asm_cmds.appendleft('sw x30, 0(x31)')
                asm_cmds.appendleft('addi x31, x0, 1000')
                asm_cmds.appendleft('lw x30, 0(x31)')
                asm_cmds.appendleft(f'li x31, {p1}')
                continue
            elif cmd == 'mv':
                # e.g.
                # mv rd, rs
                # pseudoinstruction
                asm_cmds.appendleft(f'addi {p1} {p2} 0')
                continue
            elif cmd == 'nop':
                asm_cmds.appendleft('addi x0 x0 0')
                continue
            elif cmd == 'neg':
                # neg rd, rs
                #     p1  p2
                asm_cmds.appendleft(f'sub {p1} x0 {p2}')
                continue
            elif cmd == 'beqz':
                # beqz rs, offset
                asm_cmds.appendleft(f'beq {p1} x0 {p2}')
                continue
            elif cmd == 'bnez':
                # beqz rs, offset
                asm_cmds.appendleft(f'bne {p1} x0 {p2}')
                continue
            elif cmd == 'blez':
                # blez rs, offset
                # p1 <= x0
                # x0 >= p1
                asm_cmds.appendleft(f'bge x0 {p1} {p2}')
                continue
            elif cmd == 'bgez':
                # bgez rs, offset
                # p1 >= x0
                asm_cmds.appendleft(f'bge {p1} x0 {p2}')
                continue
            elif cmd == 'bltz':
                # bltz rs, offset
                # p1 < x0
                asm_cmds.appendleft(f'blt {p1} x0 {p2}')
                continue
            elif cmd == 'bgtz':
                # blgz rs, offset
                # p1 > 0
                # 0 < p1
                asm_cmds.appendleft(f'blt x0 {p1} {p2}')
                continue
            elif cmd == 'bgt':
                # bgt rs, rt offset
                # p1 > p2
                # p2 < p1
                asm_cmds.appendleft(f'blt {p2} {p1} {p3}')
                continue
            elif cmd == 'ble':
                # bge rs, rt offset
                # p1 <= p2
                # p2 >= p1
                asm_cmds.appendleft(f'bge {p2} {p1} {p3}')
                continue
            elif cmd == 'bgtu':
                # bge rs, rt offset
                # p1 > p2
                # p2 < p1
                asm_cmds.appendleft(f'bltu {p2} {p1} {p3}')
                continue
            elif cmd == 'bleu':
                # bge rs, rt offset
                # p1 <= p2
                # p2 >= p1
                asm_cmds.appendleft(f'bgeu {p2} {p1} {p3}')
                continue
            elif cmd == 'halt':
                # virtual instruction
                # write to location 1001 instead
                asm_cmds.appendleft('sw x30, 0(x31)')
                asm_cmds.appendleft('addi x31, x0, 1004')
                continue
            elif cmd.endswith(':') and p1 is None:
                # label
                label = cmd.strip().replace(':', '')
                label_pos_by_name[label] = len(new_asm_cmds) * 4
            else:
                pass
                # ignore everything else, let it through...
                # well, add it to the new queue
                new_asm_cmds.append(line)
        except Exception as e:
            print('cmd:', line)
            raise e

    print('')
    print('cmds after expanding pseudocommands:')
    for line in new_asm_cmds:
        print(line)
    print('')

    print('label pos by name:')
    for label, pos in label_pos_by_name.items():
        print('    ', label, pos)
    asm_cmds = new_asm_cmds

    hex_lines = []
    while len(asm_cmds) > 0:
        line = asm_cmds.popleft()
        if line.strip() == '' or line.strip().startswith('#') or line.strip().startswith(';'):
            continue

        line = line.split(';')[0]
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
                op_bits = op_bits_by_op['STORE']  # "0100011"
                rs1_bits = reg_str_to_bits(p3)
                rs2_bits = reg_str_to_bits(p1)
                offset_bits = int_str_to_bits(p2, 12)
                offset1_bits = offset_bits[:7]
                offset2_bits = offset_bits[7:]
                instr_bits = f'{offset1_bits}{rs2_bits}{rs1_bits}010{offset2_bits}{op_bits}'
                hex_lines.append(bits_to_hex(instr_bits))
            elif cmd == 'lw':
                # e.g.
                # lw x2,  0      (x3)
                #    rd   offset rs1
                op_bits = op_bits_by_op['LOAD']  # "0000011"
                rs1_bits = reg_str_to_bits(p3)
                rd_bits = reg_str_to_bits(p1)
                offset_bits = int_str_to_bits(p2, 12)
                instr_bits = f'{offset_bits}{rs1_bits}010{rd_bits}{op_bits}'
                hex_lines.append(bits_to_hex(instr_bits))
            elif cmd in ['addi', 'slti', 'sltiu', 'xori', 'ori', 'andi', 'slli', 'srli', 'srai']:
                # e.g.
                # addi x1,    x2,    123
                #      rd     rs1    imm
                op_bits = op_bits_by_op['OPIMM']  # "0010011"
                imm_bits = int_str_to_bits(p3, 12)
                print('addi imm_bits', imm_bits)
                rd_bits = reg_str_to_bits(p1)
                rs1_bits = reg_str_to_bits(p2)

                funct_bits = funct_bits_opimm[cmd.upper()]
                # funct_bits = '000'
                instr_bits = f'{imm_bits}{rs1_bits}{funct_bits}{rd_bits}{op_bits}'
                hex_lines.append(bits_to_hex(instr_bits))
            elif cmd in ['lui', 'auipc']:
                # eg lui x1, 0xdeadb
                #        rd  imm
                #        p1  p2
                op_bits = op_bits_by_op[cmd.upper()]
                rd_bits = reg_str_to_bits(p1)
                imm_bits = int_str_to_bits(p2, 20)
                instr_bits = f'{imm_bits}{rd_bits}{op_bits}'
                print('lui', imm_bits)
                assert len(instr_bits) == 32
                hex_lines.append(bits_to_hex(instr_bits))

            elif cmd == 'half':
                bits = int_str_to_bits(p1, 16)
                hex_lines.append("0000" + bits_to_hex(bits, num_bytes=2))
            elif cmd == 'word':
                bits = int_str_to_bits(p1, 32)
                assert len(bits) == 32
                hex_lines.append(bits_to_hex(bits, num_bytes=4))
            elif cmd in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu']:
                # beq rs1, rs2, immed
                op_bits = op_bits_by_op['BRANCH']  # "1100011"
                funct_bits = {
                    'beq': '000',
                    'bne': '001',
                    'blt': '100',
                    'bge': '101',
                    'bltu': '110',
                    'bgeu': '111'
                }[cmd]
                rs1_bits = reg_str_to_bits(p1)
                rs2_bits = reg_str_to_bits(p2)
                label = p3
                label_pos = label_pos_by_name[label]
                pc = len(hex_lines) * 4
                label_offset = label_pos - pc
                label_offset_bits = int_to_binary(label_offset // 2, 12)
                l_bits_12 = label_offset_bits[-12]
                l_bits_11 = label_offset_bits[-11]
                l_bits_10_5 = label_offset_bits[-10:-4]
                l_bits_4_1 = label_offset_bits[-4:]
                instr_bits = f'{l_bits_12}{l_bits_10_5}{rs2_bits}{rs1_bits}{funct_bits}{l_bits_4_1}{l_bits_11}{op_bits}'
                hex_lines.append(bits_to_hex(instr_bits))
            elif cmd in [
                 'add', 'slt', 'sltu', 'and', 'or', 'xor', 'sll', 'srl', 'sub', 'sra',
                 'mul', 'mulh', 'mulhsu', 'mulhu', 'div', 'divu', 'rem', 'remu',
                 'udiv', 'div', 'mul', 'mulu', 'mod', 'modu'
            ]:
                # e.g.
                # add rd, rs1, rs2
                op_bits = op_bits_by_op['OP']  # "0110011"
                funct_bits = funct_bits_op[cmd.upper()]

                rd_bits = reg_str_to_bits(p1)
                rs1_bits = reg_str_to_bits(p2)
                rs2_bits = reg_str_to_bits(p3)
                instr_bits = f'{funct_bits[:7]}{rs2_bits}{rs1_bits}{funct_bits[-3:]}{rd_bits}{op_bits}'
                hex_lines.append(bits_to_hex(instr_bits))
            elif cmd == 'location':
                # non risc-v command, to continue writing our assembler output at a new
                # location
                assert p1.endswith(':')
                loc_int = int_str_to_int(p1[:-1])
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
    if args.dump_hex:
        with open(args.out_hex) as f:
            for line in f:
                print(line.strip())
    print('wrote ' + args.out_hex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-asm', type=str, default='prog6.asm')
    parser.add_argument('--out-hex', type=str, default='build/prog6.hex')
    parser.add_argument('--dump-hex', action='store_true')
    args = parser.parse_args()
    run(args)
