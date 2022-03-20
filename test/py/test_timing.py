import pytest
from typing import Dict, List
from toy_proc import timing


@pytest.mark.parametrize(
    "name_str, vector_bits_by_name, expected_names", [
        ("foo", {}, ["foo"]),
        ("foo", {"foo": [0, 1, 2, 4]}, ["foo[0]", "foo[1]", "foo[2]", "foo[4]"]),
        ("{foo, bar}", {}, ["foo", "bar"]),
        ("{foo [3], bar [5:7]}", {}, ["foo[3]", "bar[7]", "bar[6]", "bar[5]"]),
        ("foo[2:0]", {}, ["foo[2]", "foo[1]", "foo[0]"]),
        ("foo [2:0]", {}, ["foo[2]", "foo[1]", "foo[0]"]),
        ("foo[2]", {}, ["foo[2]"]),
        ("foo [2]", {}, ["foo[2]"]),
        ("\\op_branch$func$src/proc.sv:0$10.branch", {}, ["\\op_branch$func$src/proc.sv:0$10.branch"])
    ]
)
def test_str_to_names(name_str: str, vector_bits_by_name: Dict[str, List[int]], expected_names: List[str]):
    names = timing.str_to_names(vector_bits_by_name, name_str)
    print('names', names)
    assert names == sorted(expected_names)
