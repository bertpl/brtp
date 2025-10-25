import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from brtp.math.aggregation import (
    geo_mean,
    mean,
    ordered_weighted_geo_mean,
    ordered_weighted_mean,
    weighted_geo_mean,
    weighted_mean,
)
from brtp.math.aggregation._means import _exponential_weights


# =================================================================================================
#  Regular Means
# =================================================================================================
@pytest.mark.parametrize(
    "values, expected_result, decimals",
    [
        ([1, 2, 3], 2.0, 14),
        ((1, 2, 3), 2.0, 14),
        ({1, 2, 3}, 2.0, 14),
        ([1.0, 6.0, 11.0], 6.0, 14),
        ([2], 2.0, 14),
        ([], 0.0, 14),
        ([0, 2], 1.0, 14),
        ([0, 0], 0.0, 14),
        ([1e10] * 1000, 1e10, 1),  # fewer decimals for large # of large values
    ],
)
def test_mean(values, expected_result: float, decimals: int):
    assert_almost_equal(mean(values), expected_result, decimal=decimals)


@pytest.mark.parametrize(
    "values, weights, expected_result",
    [
        ([0, 1, 2, 3, 4], [0.0, 0.0, 1.0, 1.0, 2.0], mean([2, 3, 4, 4])),
        ([], [], 0.0),
    ],
)
def test_weighted_mean(values: list[float], weights: list[float], expected_result: float):
    # --- act ---------------------------------------------
    result = weighted_mean(values, weights)

    # --- assert ------------------------------------------
    assert result == pytest.approx(expected_result)


def test_ordered_weighted_mean():
    # --- arrange -----------------------------------------
    values = [10, 1, 5, 2, 6, 3, 4]

    # --- act ---------------------------------------------
    result_minus_2 = ordered_weighted_mean(values, -2)
    result_minus_1 = ordered_weighted_mean(values, -1)
    result_0 = ordered_weighted_mean(values, 0)
    result_plus_1 = ordered_weighted_mean(values, 1)
    result_plus_2 = ordered_weighted_mean(values, 2)

    # --- assert ------------------------------------------
    assert result_0 == pytest.approx(mean(values))
    assert 1.0 < result_minus_2 < result_minus_1 < result_0 < result_plus_1 < result_plus_2 < 10.0


# =================================================================================================
#  Geometric Means
# =================================================================================================
@pytest.mark.parametrize(
    "values, expected_result, decimals",
    [
        ([1, 2, 4], 2.0, 14),
        ((1, 2, 4), 2.0, 14),
        ({1, 2, 4}, 2.0, 14),
        ([1.0, 3.0, 9.0], 3.0, 14),
        ([2], 2.0, 14),
        ([], 1.0, 14),
        ([0, 2], 0.0, 14),
        ([0, 0], 0.0, 14),
        ([1e10] * 1000, 1e10, 1),  # fewer decimals for large # of large values
    ],
)
def test_geo_mean(values, expected_result: float, decimals: int):
    assert_almost_equal(geo_mean(values), expected_result, decimal=decimals)


@pytest.mark.parametrize(
    "values, weights, expected_result",
    [
        ([0, 1, 2, 3, 4], [0.0, 0.0, 1.0, 1.0, 2.0], geo_mean([2, 3, 4, 4])),
        ([0, 1, 2], [0.1, 0.5, 0.2], 0.0),
        ([], [], 1.0),
    ],
)
def test_weighted_geo_mean(values: list[float], weights: list[float], expected_result: float):
    # --- act ---------------------------------------------
    result = weighted_geo_mean(values, weights)

    # --- assert ------------------------------------------
    assert result == pytest.approx(expected_result)


def test_ordered_weighted_geo_mean():
    # --- arrange -----------------------------------------
    values = [10, 1, 5, 2, 6, 3, 4]

    # --- act ---------------------------------------------
    result_minus_2 = ordered_weighted_geo_mean(values, -2)
    result_minus_1 = ordered_weighted_geo_mean(values, -1)
    result_0 = ordered_weighted_geo_mean(values, 0)
    result_plus_1 = ordered_weighted_geo_mean(values, 1)
    result_plus_2 = ordered_weighted_geo_mean(values, 2)

    # --- assert ------------------------------------------
    assert result_0 == pytest.approx(geo_mean(values))
    assert 1.0 < result_minus_2 < result_minus_1 < result_0 < result_plus_1 < result_plus_2 < 10.0


# =================================================================================================
#  Helpers
# =================================================================================================
@pytest.mark.parametrize("n,c", [(10, 1), (15, -2), (100, 5)])
def test_exponential_weights(c: float, n: int):
    # --- arrange -----------------------------------------
    w_expected = np.exp(c * np.linspace(0.0, 1.0, n))

    # --- act ---------------------------------------------
    w = _exponential_weights(c, n)

    # --- assert ------------------------------------------
    assert np.allclose(w, w_expected)
