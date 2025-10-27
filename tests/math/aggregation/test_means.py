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
from brtp.math.aggregation._means import _compute_c_for_target_quantile, _compute_q_afo_c_numba, _exponential_weights
from brtp.math.sampling import linspace


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


def test_ordered_weighted_mean_c():
    # --- arrange -----------------------------------------
    values = [10, 1, 5, 2, 6, 3, 4]

    # --- act ---------------------------------------------
    result_minus_2 = ordered_weighted_mean(values, c=-2)
    result_minus_1 = ordered_weighted_mean(values, c=-1)
    result_0 = ordered_weighted_mean(values, c=0)
    result_plus_1 = ordered_weighted_mean(values, c=1)
    result_plus_2 = ordered_weighted_mean(values, c=2)

    # --- assert ------------------------------------------
    assert result_0 == pytest.approx(mean(values))
    assert 1.0 < result_minus_2 < result_minus_1 < result_0 < result_plus_1 < result_plus_2 < 10.0


def test_ordered_weighted_mean_q():
    # --- arrange -----------------------------------------
    values = [10, 1, 5, 2, 6, 3, 4]

    # --- act ---------------------------------------------
    result_q_01 = ordered_weighted_mean(values, q=0.1)
    result_q_02 = ordered_weighted_mean(values, q=0.2)
    result_q_05 = ordered_weighted_mean(values, q=0.5)
    result_q_08 = ordered_weighted_mean(values, q=0.8)
    result_q_09 = ordered_weighted_mean(values, q=0.9)

    # --- assert ------------------------------------------
    assert result_q_05 == pytest.approx(mean(values))
    assert 1.0 < result_q_01 < result_q_02 < result_q_05 < result_q_08 < result_q_09 < 10.0


def test_ordered_weighted_mean_validation():
    # --- arrange -----------------------------------------
    values = [10, 1, 5, 2, 6, 3, 4]

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        ordered_weighted_mean(values, c=None, q=None)

    with pytest.raises(ValueError):
        ordered_weighted_mean(values, c=1.0, q=0.5)


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


def test_ordered_weighted_geo_mean_c():
    # --- arrange -----------------------------------------
    values = [10, 1, 5, 2, 6, 3, 4]

    # --- act ---------------------------------------------
    result_minus_2 = ordered_weighted_geo_mean(values, c=-2)
    result_minus_1 = ordered_weighted_geo_mean(values, c=-1)
    result_0 = ordered_weighted_geo_mean(values, c=0)
    result_plus_1 = ordered_weighted_geo_mean(values, c=1)
    result_plus_2 = ordered_weighted_geo_mean(values, c=2)

    # --- assert ------------------------------------------
    assert result_0 == pytest.approx(geo_mean(values))
    assert 1.0 < result_minus_2 < result_minus_1 < result_0 < result_plus_1 < result_plus_2 < 10.0


def test_ordered_weighted_geo_mean_q():
    # --- arrange -----------------------------------------
    values = [10, 1, 5, 2, 6, 3, 4]

    # --- act ---------------------------------------------
    result_q_01 = ordered_weighted_geo_mean(values, q=0.1)
    result_q_02 = ordered_weighted_geo_mean(values, q=0.2)
    result_q_05 = ordered_weighted_geo_mean(values, q=0.5)
    result_q_08 = ordered_weighted_geo_mean(values, q=0.8)
    result_q_09 = ordered_weighted_geo_mean(values, q=0.9)

    # --- assert ------------------------------------------
    assert result_q_05 == pytest.approx(geo_mean(values))
    assert 1.0 < result_q_01 < result_q_02 < result_q_05 < result_q_08 < result_q_09 < 10.0


def test_ordered_weighted_geo_mean_validation():
    # --- arrange -----------------------------------------
    values = [10, 1, 5, 2, 6, 3, 4]

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        ordered_weighted_geo_mean(values, c=None, q=None)

    with pytest.raises(ValueError):
        ordered_weighted_geo_mean(values, c=1.0, q=0.5)


# =================================================================================================
#  Helpers
# =================================================================================================
@pytest.mark.parametrize(
    "n,c",
    [
        (10, 1.1),
        (15, -2.3),
        (100, 5.7),
        (10, 0.0),
        (1, -2.718),
        (1, 0.0),
        (1, 3.141),
    ],
)
def test_exponential_weights(c: float, n: int):
    # --- arrange -----------------------------------------
    if n > 1:
        w_expected = np.exp(c * np.linspace(0.0, 1.0, n)) / max(1, np.exp(c))
    else:
        w_expected = np.array([1.0])

    # --- act ---------------------------------------------
    w = _exponential_weights(c, n)

    # --- assert ------------------------------------------
    assert np.allclose(w, w_expected)


@pytest.mark.parametrize("q", [0.5, 0.50001, 0.5001, 0.501, 0.51, 0.6, 0.7, 0.9, 0.99])
def test_compute_c_for_target_quantile_symmetry(q: float):
    """Check if symmetry invariant is satisfied."""

    # --- act ---------------------------------------------
    c_plus = _compute_c_for_target_quantile(q)
    c_minus = _compute_c_for_target_quantile(1 - q)

    # --- assert ------------------------------------------
    assert c_minus == pytest.approx(-c_plus)


@pytest.mark.parametrize(
    "q_values",
    [
        linspace(0.01, 0.99, n=100),
        linspace(0.1, 0.9, n=100),
        linspace(0.5 - 1e-2, 0.5 + 1e-2, n=100),
        linspace(0.5 - 1e-5, 0.5 + 1e-5, n=100),
    ],
)
def test_compute_c_for_target_quantile_accuracy(q_values: list[float]):
    """Check if increasing q-values lead to increasing and distinct c-values, also for tightly spaced q-values."""

    # --- act ---------------------------------------------
    c_values = [_compute_c_for_target_quantile(q) for q in q_values]

    # --- assert ------------------------------------------
    assert c_values == sorted(c_values)
    assert len(set(c_values)) == len(c_values)


def test_compute_c_for_target_quantile_validation():
    """Check if ValueError is raised when triggered out of allowed q-range."""

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _compute_c_for_target_quantile(-0.01)

    with pytest.raises(ValueError):
        _compute_c_for_target_quantile(0.0)

    with pytest.raises(ValueError):
        _compute_c_for_target_quantile(0.01 - 1e-15)

    with pytest.raises(ValueError):
        _compute_c_for_target_quantile(0.99 + 1e-15)

    with pytest.raises(ValueError):
        _compute_c_for_target_quantile(1.0)

    with pytest.raises(ValueError):
        _compute_c_for_target_quantile(1.01)


@pytest.mark.parametrize("c", [1.0, -1.0, -33.6, 0.0, 78.5, 1e-6, -1e-5])
def test_compute_q_afo_c_numba(c: float):
    # --- act ---------------------------------------------
    q = _compute_q_afo_c_numba(c)

    # --- assert ------------------------------------------
    assert 0.0 <= q <= 1.0
    if c > 0:
        assert q > 0.5
    elif c < 0:
        assert q < 0.5
    else:
        assert q == 0.5
