import pytest

from brtp.collections import zip_random


@pytest.mark.parametrize("seed", [None, 0, 42, 1234])
def test_zip_random_invariants(seed: int | None):
    # --- arrange -----------------------------------------
    i1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    i2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

    # --- act ---------------------------------------------
    result = list(zip_random(i1, i2, seed=seed))

    # --- assert ------------------------------------------
    assert {v for v, _ in result} == set(i1), "all elements should be present"
    assert [v for v, _ in result] != i1, "iterable should be randomized"
    assert {v for _, v in result} == set(i2), "all elements should be present"
    assert [v for _, v in result] != i2, "iterable should be randomized"


def test_zip_random_seed():
    # --- arrange -----------------------------------------
    i1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    i2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

    # --- act ---------------------------------------------
    result_1 = list(zip_random(i1, i2, seed=None))
    result_2 = list(zip_random(i1, i2, seed=1))
    result_3 = list(zip_random(i1, i2, seed=2))
    result_4 = list(zip_random(i1, i2, seed=2))

    # --- assert ------------------------------------------
    assert result_1 != result_2
    assert result_1 != result_3
    assert result_2 != result_3
    assert result_3 == result_4, "same seed should give same result"
