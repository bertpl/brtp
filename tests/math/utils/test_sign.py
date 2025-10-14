import math

import pytest

from brtp.math.utils import same_sign, sign


@pytest.mark.parametrize(
    "x,expected",
    [
        (1, 1),
        (-1, -1),
        (0, 0),
        (123.456, 1),
        (-987.654, -1),
        (float("inf"), 1),
        (float("-inf"), -1),
        (float("nan"), None),
        (-0.0, 0),
    ],
)
def test_sign(x, expected):
    assert sign(x) == expected


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (1, 2, True),
        (-1, -2, True),
        (0, 0, True),
        (1, -1, False),
        (-1, 1, False),
        (0, 1, False),
        (0, -1, False),
        (1, 0, False),
        (-1, 0, False),
        (float("nan"), 1, False),
        (1, float("nan"), False),
        (float("nan"), float("nan"), False),
        (float("inf"), 1, True),
        (float("-inf"), -1, True),
        (float("inf"), float("-inf"), False),
        (float("inf"), 0, False),
        (0, float("-inf"), False),
        (-0.0, 0.0, True),
        (0.0, -0.0, True),
    ],
)
def test_same_sign(x, y, expected):
    assert same_sign(x, y) == expected
