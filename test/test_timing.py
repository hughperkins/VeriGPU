import pytest
from typing import Dict, Tuple, List
from toy_proc import timing


@pytest.mark.parametrize(
    "name_str, vector_dims_by_name, expected_names", [
        ("foo", {}, ["foo"]),
        ("foo", {"foo": (0, 2, 1)}, ["foo[0]", "foo[1]"]),
        ("{foo, bar}", {}, ["foo", "bar"]),
        ("foo[2:0]", {}, ["foo[2]", "foo[1]", "foo[0]"])
    ]
)
def test_str_to_names(name_str: str, vector_dims_by_name: Dict[str, Tuple[int, int, int]], expected_names: List[str]):
    names = timing.str_to_names(vector_dims_by_name, name_str)
    print('names', names)
    assert names == sorted(expected_names)
