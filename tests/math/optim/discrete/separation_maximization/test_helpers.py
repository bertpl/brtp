import math

import numpy as np
import pytest

from brtp.math.optim.discrete.separation_maximization import CachedDistances, mean_separation, min_separation


# =================================================================================================
#  Metrics
# =================================================================================================
@pytest.mark.parametrize(
    "vectors, expected_result",
    [
        (np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 10.0]]), 0.5),
        (np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]), 5.0),
        (np.array([[1.0, 2.0], [10.0, 5.0], [1.0, 2.0]]), 0.0),
        (np.array([[1.0], [5.0], [2.0]]), 1.0),
        (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), math.sqrt(27)),
    ],
)
def test_min_separation(vectors: np.ndarray, expected_result: float):
    # --- act ---------------------------------------------
    result = min_separation(vectors)

    # --- assert ------------------------------------------
    assert np.isclose(result, expected_result)


@pytest.mark.parametrize(
    "vectors, expected_result",
    [
        (np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 10.0]]), (0.5 + 0.5 + 10) / 3),
        (np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]), 5.0),
        (np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 5.0]]), (1 + 1 + 5) / 3),
    ],
)
def test_mean_separation(vectors: np.ndarray, expected_result: float):
    # --- act ---------------------------------------------
    result = mean_separation(vectors)

    # --- assert ------------------------------------------
    assert np.isclose(result, expected_result)


# =================================================================================================
#  Computing distances
# =================================================================================================
def test_cached_distances():
    # --- arrange -----------------------------------------
    vectors = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 1.0], [0.0, 10.0, 0.0], [0.0, 10.0, 1.0]])
    cached_distances = CachedDistances(vectors)

    # --- act ---------------------------------------------
    d02 = cached_distances.dist(0, 2)
    d23 = cached_distances.dist(2, 3)
    d23_cached = cached_distances.dist(2, 3)
    d32_cached = cached_distances.dist(3, 2)

    # --- assert ------------------------------------------
    assert d02 == 10.0
    assert d23 == 1.0
    assert d23_cached == 1.0
    assert d32_cached == 1.0
