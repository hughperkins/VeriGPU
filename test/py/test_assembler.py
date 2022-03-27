import pytest
import struct
import math

from verigpu import assembler


def bin_str_to_single(b):
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
        1.2345678
    ]
)
def test_float_to_bits(float_value: float):
    bits = assembler.float_to_bits(float_value)
    reconstr_float = bin_str_to_single('0b' + bits)
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
    reconstr_bits = assembler.int_to_binary(reconstr_int, 32)
    assert reconstr_bits == word_bits
