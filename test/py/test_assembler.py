import pytest
import struct
import math
from typing import Dict, List

from verigpu import assembler


def bits_to_float(b):
    bf = int.to_bytes(int(b, 2), 4, byteorder='big')
    return struct.unpack('>f', bf)[0]


@pytest.mark.parametrize(
    "float_value", [
        0,
        -2.5,
        123.456,
        4,
        2,
        1,
        0.5,
        0.25,
        10.123,
        123,
        0.0123,
        1.23,
        0.123,
        1.2345678
    ]
)
def test_float_to_bits(float_value: float):
    bits = assembler.float_to_bits(float_value)
    reconstr_float = bits_to_float('0b' + bits)
    print(float_value, bits, reconstr_float)
    assert float_value == pytest.approx(reconstr_float, rel=1e-7)


@pytest.mark.parametrize(
    "word_bits", [
        '00000111010110111100110100010101',
        '11111111111111111111111111111111',
        '00000000000000000000000000000000',
        '00000000000000000000000000000001',
        '10000000000000000000000000000000',
        '00000000000000000000010000000000',
        '00000000000000000000100000000000',
        '00000000000000000001000000000000',
        '00000000000000000000110000000000',
        '00000000000000000001100000000000',
        '00000000000000000001110000000000',
        '00000000000000000001110000000001',
        '10000000000000000001110000000001',
    ]
)
def test_word_bits_to_lui_addi_bits(word_bits: str):
    original_int = int(word_bits, 2)
    print('original_int', original_int)
    lui_bits, addi_bits = assembler.word_bits_to_lui_addi_bits(word_bits)
    print('word bits       ', word_bits)
    print('lui bits', lui_bits, 'addi_bits', addi_bits)
    lui_int = int(lui_bits, 2) * int(math.pow(2, 12))
    addi_int = int(addi_bits, 2)
    if addi_int > 2047:
        addi_int -= 4096
    reconstr_int = lui_int + addi_int
    print('reconstr_int', reconstr_int)
    reconstr_bits = assembler.int_to_bits(reconstr_int, 32)
    assert reconstr_bits == word_bits


@pytest.mark.parametrize(
    "offset", [
        0,
        1000,
        -1000,
        5000,
        -5000
    ]
)
def test_offset_to_auipc_jalr(offset: int):
    auipc, jalr = assembler.offset_to_auipc_jalr_offset(offset)
    print('auipc', auipc, 'jalr', jalr)
    if jalr >= 2048:
        jalr -= 4096
    reconstr = auipc * int(math.pow(2, 12)) + jalr
    if reconstr >= int(math.pow(2, 31)):
        reconstr -= int(math.pow(2, 32))
    print('reonstr', reconstr)
    assert reconstr == offset


@pytest.mark.parametrize(
    "p2", [
        ("0.123"),
        ("-0.123"),
        ("1234.56"),
    ]
)
def test_li_float(p2: str):
    rd_str = 'x1'
    cmds = assembler.process_li(p1=rd_str, p2=p2, label_pos_by_name={})
    assert len(cmds) == 2
    print('cmds', cmds)
    assert cmds[0].startswith(f'lui {rd_str}, ')
    upper_str = cmds[0].split(f'{rd_str},')[1].strip()
    print('upper_str', upper_str)
    assert cmds[1].startswith(f'addi {rd_str}, {rd_str}, ')
    lower_str = cmds[1].split(f'{rd_str}, {rd_str},')[1].strip()
    print('lower_str', lower_str)
    upper_val = assembler.int_str_to_int(upper_str)
    lower_val = assembler.int_str_to_int(lower_str)
    if lower_val >= 2048:
        lower_val -= 2048
    val = upper_val * int(math.pow(2, 12)) + lower_val
    print('upper_val', upper_val)
    print('lower_val', lower_val)
    val_bits = assembler.int_to_bits(val, 32)
    print('val_bits', val_bits)
    val_float = bits_to_float(val_bits)
    print('val_float', val_float)
    assert val_float == pytest.approx(float(p2), rel=1e-7)


def check_addi_int(cmd: str, rd_str: str, expected_int: int):
    print('cmd', cmd)
    assert cmd.startswith(f'addi {rd_str}, x0, ')
    returned_str = cmd.split(f'{rd_str}, x0,')[1].strip()
    returned_val = int(returned_str)
    if returned_val >= 2048:
        returned_val -= 4096
    assert returned_val == expected_int


@pytest.mark.parametrize(
    "imm_val", [
        123,
        -123,
        1000,
        -1000
    ]
)
def test_li_small_int(imm_val: int):
    cmds = assembler.process_li(p1='x1', p2=str(imm_val), label_pos_by_name={})
    assert len(cmds) == 1
    check_addi_int(cmd=cmds[0], rd_str='x1', expected_int=imm_val)


def check_lui_addi_pair_int(cmds: List[str], rd_str: str, expected_int: int):
    assert len(cmds) == 2
    assert cmds[0].startswith(f'lui {rd_str}, ')
    upper_str = cmds[0].split(f'{rd_str},')[1].strip()
    print('upper_str', upper_str)
    assert cmds[1].startswith(f'addi {rd_str}, {rd_str}, ')
    lower_str = cmds[1].split(f'{rd_str}, {rd_str},')[1].strip()
    print('lower_str', lower_str)
    upper_value = int(upper_str)
    lower_value = int(lower_str)
    if lower_value >= 2048:
        lower_value -= 4096
    overall_value = upper_value * int(math.pow(2, 12)) + lower_value
    if overall_value >= int(math.pow(2, 31)):
        overall_value -= int(math.pow(2, 32))
    assert overall_value == expected_int


@pytest.mark.parametrize(
    "imm_val", [
        10000,
        -10000,
        12345678,
        -12345678
    ]
)
def test_li_large_int(imm_val: int):
    cmds = assembler.process_li(p1='x1', p2=str(imm_val), label_pos_by_name={})
    check_lui_addi_pair_int(cmds, 'x1', imm_val)


@pytest.mark.parametrize(
    "imm_str, label_pos_by_name, expected_int", [
        (".fooZi3", {'blah': 24, '.fooZi3': 124}, 124),
        (".fooZi3", {'blah': 24, '.fooZi3': -124}, -124),
    ]
)
def test_li_label_small_offset(imm_str: str, label_pos_by_name: Dict[str, int], expected_int: int):
    cmds = assembler.process_li(p1='x1', p2=imm_str, label_pos_by_name=label_pos_by_name)
    assert len(cmds) == 1
    check_addi_int(cmd=cmds[0], rd_str='x1', expected_int=expected_int)


@pytest.mark.parametrize(
    "imm_str, label_pos_by_name, expected_int", [
        (".fooZi3", {'blah': 24, '.fooZi3': -10000}, -10000),
        (".fooZi3", {'blah': 24, '.fooZi3': 10000}, 10000),
    ]
)
def test_li_label_large_offset(imm_str: str, label_pos_by_name: Dict[str, int], expected_int: int):
    cmds = assembler.process_li(p1='x1', p2=imm_str, label_pos_by_name=label_pos_by_name)
    check_lui_addi_pair_int(cmds, 'x1', expected_int)
