from typing import Iterable

import pytest

from brtp.misc.argument_handling import (
    all_are_none,
    all_are_not_none,
    count_none,
    count_not_none,
)


@pytest.mark.parametrize(
    "args, expected_result",
    [
        ([], True),
        ([None, None, None], True),
        ([None, 1, None], False),
        ([None, 0, None], False),
        (["", (), {}, dict(), 0], False),
        ([1, 2, 3], False),
    ],
)
def test_all_are_none(args: Iterable, expected_result: bool):
    assert all_are_none(*args) == expected_result


@pytest.mark.parametrize(
    "args, expected_result",
    [
        ([], True),
        ([None, None, None], False),
        ([None, 1, None], False),
        ([None, 0, None], False),
        (["", (), {}, dict(), 0], True),
        ([0], True),
        ([0, 0], True),
        ([1, 2, 3], True),
    ],
)
def test_all_are_not_none(args: Iterable, expected_result: bool):
    assert all_are_not_none(*args) == expected_result


@pytest.mark.parametrize(
    "args, expected_result",
    [
        ([], 0),
        ([None, None, None], 3),
        ([None, 1, None], 2),
        ([None, 0, None], 2),
        (["", (), {}, dict(), 0], 0),
        ([0], 0),
        ([0, 0], 0),
        ([1, 2, 3], 0),
    ],
)
def test_count_none(args: Iterable, expected_result: int):
    assert count_none(*args) == expected_result


@pytest.mark.parametrize(
    "args, expected_result",
    [
        ([], 0),
        ([None, None, None], 0),
        ([None, 1, None], 1),
        ([None, 0, None], 1),
        (["", (), {}, dict(), 0], 5),
        ([0], 1),
        ([0, 0], 2),
        ([1, 2, 3], 3),
    ],
)
def test_count_not_none(args: Iterable, expected_result: int):
    assert count_not_none(*args) == expected_result
