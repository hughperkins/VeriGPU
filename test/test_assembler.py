import pytest
import struct

from toy_proc import assembler


def bin_str_to_single(b):
    bf = int.to_bytes(int(b, 2), 4, byteorder='big')
    return struct.unpack('>f', bf)[0]


@pytest.mark.parametrize(
    "float_value", [
        0,
        -2.5,
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
    assert float_value == pytest.approx(reconstr_float)
