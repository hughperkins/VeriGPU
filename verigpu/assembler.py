import argparse
import os
import math
from typing import Dict, Union, List
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
    'OPIMM32':  '0011011',
    'JAL':      '1101111'
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
    'SRLI':   '101',
    'SRAI':   '101'
}


flt_fmt_bits = {
    'S': '00',
    'D': '01',
    'H': '10',
    'Q': '11'
}


flt_rm_bits = {
    'RNE': '000',
    'RTZ': '001',
    'RDN': '010',
    'RUP': '011',
    'RMM': '100',
    'DYN': '111'
}


funct5_bits_float = {
    'FADD':  '00000',
    'FSUB':  '00001',
    'FMUL':  '00010',
    'FDIV':  '00011',
    'FSQRT': '01011'
}

reg_aliases = {
    'sp': 'x2',
    'ra': 'x1',
    'a0': 'x10',
    'a1': 'x11',
    'a2': 'x12',
    'a3': 'x13',
    'a4': 'x14',
    'a5': 'x15',
    'a6': 'x16',
    'a7': 'x17',
    's0': 'x8',
    'zero': 'x0'  # is this right? need to check
}


def int_to_bits(int_value: int, num_bits: int) -> str:
    """
    1234
    """
    if int_value < 0:
        offset = int(math.pow(2, num_bits))
        int_value += offset
    return format(int_value, f'#0{num_bits + 2}b')[2:]


def hex_to_binary(hex_value: str, num_bits: int) -> str:
    """
    0xabcd
    """
    assert hex_value.startswith('0x')
    hex_value = hex_value[2:]
    bits = int_to_bits(int(hex_value, 16), num_bits)
    assert len(bits) == num_bits
    return bits


def int_str_to_int(int_str: str) -> int:
    """
    example formats:

    1234
    0xabcd
    0b0101

    (so they always start with a digit)
    """
    if int_str.startswith('0x'):
        int_value = int(int_str[2:], 16)
    elif int_str.startswith('0b'):
        int_value = int(int_str[2:], 2)
    else:
        int_value = int(int_str)
    return int_value


def int_str_to_bits(int_str: str, num_bits: int) -> str:
    int_value = int_str_to_int(int_str)
    bits = int_to_bits(int_value, num_bits)
    if len(bits) != num_bits:
        print('len(bits)', len(bits), bits)
    assert len(bits) == num_bits
    return bits


def float_to_bits(float_value, num_exp_bits: int = 8, num_sig_bits: int = 23):
    sign_bit = 1 if float_value < 0 else 0
    if float_value == 0:
        return '0' * (1 + num_exp_bits + num_sig_bits)
    elif float_value != float_value:
        return '0' + '1' * (num_exp_bits + num_sig_bits)
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
        exp_bits = int_to_bits(exp_to_store, num_exp_bits)
        signif_to_store = (signif - 1) * math.pow(2, num_sig_bits)
        signif_bits = int_to_bits(int(signif_to_store), num_sig_bits)
    return f'{sign_bit}{exp_bits}{signif_bits}'


def numeric_str_to_value(numeric_str) -> Union[float, int]:
    if '.' in numeric_str:
        return float(numeric_str)
    return int_str_to_int(numeric_str)


def numeric_str_to_bits(numeric_str, num_bits):
    val = numeric_str_to_value(numeric_str)
    if isinstance(val, float):
        bits = float_to_bits(val, num_exp_bits=8, num_sig_bits=23)
    else:
        bits = int_to_bits(val, num_bits)
    assert len(bits) == num_bits
    return bits


def reg_str_to_bits(reg_str, num_bits: int = 5):
    reg_str = reg_aliases.get(reg_str, reg_str)
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


def bits_to_int(bits: str) -> int:
    return int(bits, 2)


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
    upper_bits2 = int_to_bits(upper_int, 20)
    lower_bits2 = int_to_bits(lower_int, 12)
    return upper_bits2, lower_bits2


def offset_to_auipc_jalr_offset(offset: int):
    label_offset_bits = int_to_bits(offset, 32)
    print('label_offse_bits', label_offset_bits)
    l_offset_31_12 = label_offset_bits[-32:-12]
    print('l_offset_31_12', l_offset_31_12)
    assert len(l_offset_31_12) == 20
    l_offset_31_12_int = bits_to_int(l_offset_31_12)
    l_offset_11_0 = label_offset_bits[-12:]
    assert len(l_offset_11_0) == 12
    auipc_offset = l_offset_31_12_int + int(label_offset_bits[-11])
    print('auipc_offset', auipc_offset)
    print('2^20', int(math.pow(2, 20)))
    print('2^19', int(math.pow(2, 19)))
    if auipc_offset >= int(math.pow(2, 20)):
        auipc_offset -= int(math.pow(2, 20))
    jalr_offset = bits_to_int(l_offset_11_0)
    return auipc_offset, jalr_offset


def imm_to_val(label_pos_by_name: Dict[str, int], imm_str: str, offset_start: int) -> Union[int, float]:
    """
    imm_str could be a number, or a label
    if a label, comess back as a relative offset, relative to offset_start
    otherwise comes back as an integer, or a float
    """
    imm_str = imm_str.strip()
    assert len(imm_str) > 0
    if imm_str[0] in "0123456789-":
        # return int_str_to_int(imm_str)
        return numeric_str_to_value(imm_str)
    else:
        label_pos = label_pos_by_name[imm_str]
        offset = label_pos - offset_start
        return offset


def process_li(p1: str, p2: str, label_pos_by_name: Dict[str, int]) -> List[str]:
    # e.g.: li x1 0x12
    # virtual command; convert to e.g. addi x1, x0, 0x12
    #
    # immediate can be a number or a label
    # if a label, will be converted into offset from 0
    p2 = p2.strip()
    assert len(p2) > 0

    rd_str = p1
    imm_val = imm_to_val(label_pos_by_name=label_pos_by_name, imm_str=p2, offset_start=0)

    if isinstance(imm_val, int):
        if abs(imm_val) < 2048:
            # small ints can be loaded with single addi
            return [f'addi {rd_str}, x0, {imm_val}']
        imm_bits = int_to_bits(imm_val, 32)
    else:
        # float
        imm_bits = float_to_bits(imm_val)

    lui_bits, addi_bits = word_bits_to_lui_addi_bits(imm_bits)
    lui_value = bits_to_int(lui_bits)
    addi_value = bits_to_int(addi_bits)

    cmds = []
    # cmds.append(f'lui {rd_str}, 0b{lui_bits}')
    # cmds.append(f'addi {rd_str}, {rd_str}, 0b{addi_bits}')
    cmds.append(f'lui {rd_str}, {lui_value}')
    cmds.append(f'addi {rd_str}, {rd_str}, {addi_value}')
    return cmds


def run(args):
    label_pos_by_name = {}

    if not os.path.isdir('build'):
        os.makedirs('build')
    with open(args.in_asm) as f:
        assembly = f.read()
    asm_cmds = deque(assembly.split('\n'))

    # first we run a pass to expand pseudocommands, and record the locations of labels
    # then we run the assembly pass

    new_asm_cmds = deque()
    while len(asm_cmds) > 0:
        line = asm_cmds.popleft()
        if line.strip() == '' or line.strip().startswith('#') or line.strip().startswith(';'):
            continue
        if '#' in line:
            line = line.split('#')[0].strip()  # remove comments

        line = line.split(';')[0]
        line = line.replace(',', ' ').replace("(", " ").replace(")", " ").replace(
            '  ', ' ').replace("  ", " ")
        split_line = line.split()
        cmd = split_line[0].lower()
        p1 = split_line[1] if len(split_line) >= 2 else None
        p2 = split_line[2] if len(split_line) >= 3 else None
        p3 = split_line[3] if len(split_line) >= 4 else None

        try:
            # if cmd.startswith('.'):
            #     # ignore, for now
            #     continue
            if cmd == 'li':
                # e.g.: li x1 0x12
                # virtual command; convert to e.g. addi x1, x0, 0x12
                #
                # immediate can be a number or a label
                # if a label, will be converted into offset from 0
                # if it's a label, we will just put placeholders for now, and handle later
                # we will put two placeholders
                p2 = p2.strip()
                assert len(p2) > 0
                if p2[0] in "-1234566790":
                    # numeric
                    cmds = process_li(
                        p1=p1,
                        p2=p2,
                        label_pos_by_name=label_pos_by_name)
                    while len(cmds) > 0:
                        asm_cmds.appendleft(cmds.pop())
                    continue
                else:
                    # not numeric, it's a label, just put placeholders for now
                    new_asm_cmds.append(line)
                    new_asm_cmds.append('addi x0, x0, 0')  # NOP; we can fill this in later
                    continue
            elif cmd == 'out':
                # e.g.: out 0x1b
                # virtual instruction. map to storing to location 1000

                # migrate to use store at location 1000
                # need to push store in first, since we are pushing in reverse order
                asm_cmds.appendleft('sw x30, 0(x31)')
                asm_cmds.appendleft(f'addi x30, x0, {p1}')
                asm_cmds.appendleft('li x31, 1000000')
                continue
            elif cmd == 'outr':
                # virtual instruction. map to storing to location 1000

                # migrate to use store at location 1000, with load before that
                # need to push store in first, since we are pushing in reverse order

                asm_cmds.appendleft(f'sw {p1}, 0(x31)')
                asm_cmds.appendleft('li x31, 1000000')
                continue
            elif cmd == 'outr.s':
                # virtual instruction. map to storing to location 1008

                # migrate to use store at location 1000, with load before that
                # need to push store in first, since we are pushing in reverse order

                asm_cmds.appendleft(f'sw {p1}, 0(x31)')
                asm_cmds.appendleft('li x31, 1000008')
                continue
            elif cmd == 'outloc':
                # e.g. outloc 0x20
                # virtual command, maps to li followed by sw to location 1000
                asm_cmds.appendleft('sw x30, 0(x31)')
                asm_cmds.appendleft('li x31, 1000000')
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
                asm_cmds.appendleft('li x31 1000004')
                continue
            elif cmd == 'j':
                # eg
                # j offset
                #   p1
                asm_cmds.appendleft(f'jal x0, {p1}')
                continue
            elif cmd == 'jal' and p2 is None:
                # eg jal offset
                #        p1
                asm_cmds.appendleft(f'jal x1, {p1}')
                continue
            elif cmd == 'jr':
                # eg
                # jr rs
                #    p1
                asm_cmds.appendleft(f'jalr x0, 0({p1})')
                continue
            elif cmd == 'jalr' and p2 is None:
                # eg jalr x1
                #         p1
                asm_cmds.appendleft(f'jalr x1, 0({p1})')
                continue
            elif cmd == 'ret':
                asm_cmds.appendleft('jalr x0, 0(x1)')
                continue
            elif cmd == 'call':
                # e.g.
                # call label
                #      p1
                new_asm_cmds.append(line)
                new_asm_cmds.append('addi x0, x0, 0')  # NOP; we can fill this in later
                continue
            elif cmd.endswith(':') and p1 is None:
                # label
                # label = cmd.strip().replace(':', '')
                label = line.strip().replace(':', '')
                if label in label_pos_by_name:
                    raise Exception('label ', label, 'already defined at ', label_pos_by_name[label])
                label_pos_by_name[label] = len(new_asm_cmds) * 4 + args.offset
            elif cmd.startswith('.'):
                # ignore
                continue
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
    for i, line in enumerate(new_asm_cmds):
        print(i * 4 + args.offset, ':', line)
    print('')
    with open('build/after_expand.asm', 'w') as f:
        for i, line in enumerate(new_asm_cmds):
            f.write(str(i * 4 + args.offset) + ':' + line + '\n')

    print('label pos by name:')
    for label, pos in label_pos_by_name.items():
        print('    ', label, pos)
    asm_cmds = new_asm_cmds

    # instnatiate any new virutal instructions we need to handle, such as call
    # in this pass, we are not allowed to change positions
    do_another_pass = True
    while do_another_pass:
        do_another_pass = False
        new_asm_cmds = deque()
        while len(asm_cmds) > 0:
            line = asm_cmds.popleft()
            split_line = line.split()
            cmd = split_line[0].lower()
            p1 = split_line[1] if len(split_line) >= 2 else None
            p2 = split_line[2] if len(split_line) >= 3 else None
            p3 = split_line[3] if len(split_line) >= 4 else None

            try:
                if cmd == 'li':
                    # e.g.: li x1 0x12
                    # virtual command; convert to e.g. addi x1, x0, 0x12
                    #
                    # by this point, it should be a label
                    # anyway, we have two placehodlers for the new instructions
                    # this insturction, and the next
                    cmds = process_li(
                        p1=p1,
                        p2=p2,
                        label_pos_by_name=label_pos_by_name)

                    assert len(cmds) in [1, 2]
                    if len(cmds) == 2:
                        # pop the nop
                        asm_cmds.popleft()
                    while len(cmds) > 0:
                        asm_cmds.appendleft(cmds.pop())
                    continue
                elif cmd in ['call', 'tail']:
                    # e.g.
                    # call label
                    #      p1
                    # we will repalce the folllowing dummy nop too
                    pivot_reg1 = {
                        'call': 'x1',
                        'tail': 'x6'
                    }[cmd]
                    pivot_reg2 = {
                        'call': 'x1',
                        'tail': 'x0'
                    }[cmd]
                    label = p1
                    print(cmd, p1)
                    label_pos = label_pos_by_name[label]
                    pc = len(new_asm_cmds) * 4 + args.offset
                    label_offset = label_pos - pc
                    print('label_offset', label_offset)
                    auipc_offset, jalr_offset = offset_to_auipc_jalr_offset(label_offset)

                    print('auipc_offset', auipc_offset)
                    print('jalr offset', jalr_offset)

                    new_asm_cmds.append(f'auipc {pivot_reg1}, {auipc_offset}')
                    new_asm_cmds.append(f'jalr {pivot_reg2}, {jalr_offset}({pivot_reg1})')

                    # pop the nop
                    asm_cmds.popleft()

                    do_another_pass = False  # we dont need to handle these in intermeidate pass
                else:
                    new_asm_cmds.append(line)
            except Exception as e:
                print('cmd:', line)
                raise e

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
            if cmd.startswith('.'):
                # ignore, for now
                continue
            elif cmd == 'sw':
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
            elif cmd in ['slli', 'srli', 'srai']:
                # e.g.
                # slli x1, x2, 3
                # slri x1, x2, 2
                #      p1  p2  p3
                assert cmd != 'srai'  # not supported yet
                op_bits = op_bits_by_op['OPIMM']  # "0010011"
                imm_bits = int_str_to_bits(p3, 5)
                print('addi imm_bits', imm_bits)
                rd_bits = reg_str_to_bits(p1)
                rs1_bits = reg_str_to_bits(p2)
                funct_bits = funct_bits_opimm[cmd.upper()]

                instr_bits = f'0000000{imm_bits}{rs1_bits}{funct_bits}{rd_bits}{op_bits}'
                hex_lines.append(bits_to_hex(instr_bits))
            elif cmd in ['addi', 'slti', 'sltiu', 'xori', 'ori', 'andi']:
                # e.g.
                # addi x1,    x2,    123
                #      rd     rs1    imm
                op_bits = op_bits_by_op['OPIMM']  # "0010011"
                imm_bits = int_str_to_bits(p3, 12)
                # print('addi imm_bits', imm_bits)
                rd_bits = reg_str_to_bits(p1)
                rs1_bits = reg_str_to_bits(p2)

                funct_bits = funct_bits_opimm[cmd.upper()]
                # funct_bits = '000'
                instr_bits = f'{imm_bits}{rs1_bits}{funct_bits}{rd_bits}{op_bits}'
                hex_lines.append(bits_to_hex(instr_bits))
            elif cmd.startswith('f'):
                cmd, _, fmt = cmd.partition('.')
                op_bits = op_bits_by_op['OPFP']
                fmt_bits = flt_fmt_bits[fmt.upper()]
                funct5_bits = funct5_bits_float[cmd.upper()]
                rm_bits = flt_rm_bits['RNE']

                rd_bits = reg_str_to_bits(p1)
                rs1_bits = reg_str_to_bits(p2)
                rs2_bits = reg_str_to_bits(p3)

                if cmd in ['fadd', 'fsub', 'fmul', 'fdiv', 'fsqrt']:
                    instr_bits = f'{funct5_bits}{fmt_bits}{rs2_bits}{rs1_bits}{rm_bits}{rd_bits}{op_bits}'
                    hex_lines.append(bits_to_hex(instr_bits))
                else:
                    print(line)
                    raise Exception('unhandled cmd', cmd)

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
                bits = numeric_str_to_bits(p1, 32)
                assert len(bits) == 32
                hex_lines.append(bits_to_hex(bits, num_bytes=4))
            elif cmd in ['jal']:
                # eg
                # jal rd, label
                #     p1  p2
                # stores pc + 4 into rd, and jumps to label
                print('jal', p1, p2)
                rd_bits = reg_str_to_bits(p1)
                label = p2
                label_pos = label_pos_by_name[label]
                pc = len(hex_lines) * 4 + args.offset
                # print('jal pc', pc)
                label_offset = label_pos - pc
                print('jal label_offset', label_offset)
                label_offset_bits = int_to_bits(label_offset, 21)
                assert label_offset_bits[-1] == '0'
                # print('jal label_offset_bits', label_offset_bits)
                l_bits_20 = label_offset_bits[-21]
                l_bits_10_1 = label_offset_bits[-11:-1]
                # print('jal l_bits_10_1', l_bits_10_1)
                assert len(l_bits_10_1) == 10
                l_bits_11 = label_offset_bits[-12]
                # print('jal l_bits_11', l_bits_11)
                l_bits_19_12 = label_offset_bits[-20:-12]
                # print('jal l_bits_19_12', l_bits_19_12)

                opcode_bits = op_bits_by_op['JAL']

                instr_bits = f'{l_bits_20}{l_bits_10_1}{l_bits_11}{l_bits_19_12}{rd_bits}{opcode_bits}'
                # print('instr_bits', instr_bits)
                assert len(instr_bits) == 32
                hex_lines.append(bits_to_hex(instr_bits))

            elif cmd in ['jalr']:
                # eg
                # jalr rd, imm(rs1)
                #      p1  p2  p3
                # stores next pc in rd, and jumps to rs1 + imm
                # p2 can be a number; doesnt have to be a label
                print('JALR', p1, p2, p3)
                opcode_bits = op_bits_by_op['JALR']
                rd_bits = reg_str_to_bits(p1)
                # label = p2
                pc = len(hex_lines) * 4 + args.offset
                imm_val = imm_to_val(label_pos_by_name=label_pos_by_name, imm_str=p2, offset_start=pc)
                print('jalr imm val', imm_val)
                rs1_bits = reg_str_to_bits(p3)
                # label_pos = label_pos_by_name[label]
                # label_offset = label_pos - pc
                imm_bits = int_to_bits(imm_val, 12)

                instr_bits = f'{imm_bits}{rs1_bits}000{rd_bits}{opcode_bits}'
                hex_lines.append(bits_to_hex(instr_bits))
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
                pc = len(hex_lines) * 4 + args.offset
                label_offset = label_pos - pc
                label_offset_bits = int_to_bits(label_offset // 2, 12)
                l_bits_12 = label_offset_bits[-12]
                l_bits_11 = label_offset_bits[-11]
                l_bits_10_5 = label_offset_bits[-10:-4]
                l_bits_4_1 = label_offset_bits[-4:]
                print('rs1_bits', rs1_bits, 'rs2_bits', rs2_bits, 'label', label, 'label_offset', label_offset)
                print('label_offset_bits', label_offset_bits)
                instr_bits = f'{l_bits_12}{l_bits_10_5}{rs2_bits}{rs1_bits}{funct_bits}{l_bits_4_1}{l_bits_11}{op_bits}'
                print('instr_bits', instr_bits)
                assert len(instr_bits) == 32
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
                if len(hex_lines) + args.offset // 4 > location:
                    print("len(hex_lines) + offset // 4", len(hex_lines) + args.offset // 4, "loc_int//4", location)
                assert len(hex_lines) + args.offset // 4 <= location
                while len(hex_lines) + args.offset // 4 < location:
                    hex_lines.append('00000000')
            elif cmd.startswith('.'):
                # ignore
                continue
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
    parser.add_argument('--offset', type=int, default=0, help='at what address will this be located?')
    args = parser.parse_args()
    run(args)
