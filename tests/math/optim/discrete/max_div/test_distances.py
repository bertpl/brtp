import math

import numpy as np
import pytest

from brtp.math.optim.discrete.max_div import (
    DistanceMetric,
    PairWiseDistances,
    PairWiseDistances_Eager,
    PairWiseDistances_Lazy,
)


# =================================================================================================
#  Fixtures
# =================================================================================================
@pytest.fixture
def vectors() -> np.ndarray:
    return np.array([[(row / 10) ** (1 + col) for col in range(3)] for row in range(10)])


@pytest.fixture
def true_distances_l1(vectors: np.ndarray) -> np.ndarray:
    """Matrix of true pair-wise L1 distances for the provided vectors."""
    m, n = vectors.shape
    distances = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            dist = sum([abs(vectors[i, k] - vectors[j, k]) for k in range(n)])
            distances[i, j] = dist
    return distances


@pytest.fixture
def true_distances_l2(vectors: np.ndarray) -> np.ndarray:
    """Matrix of true pair-wise L2 distances for the provided vectors."""
    m, n = vectors.shape
    distances = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            dist = math.sqrt(sum([(vectors[i, k] - vectors[j, k]) ** 2 for k in range(n)]))
            distances[i, j] = dist
    return distances


# =================================================================================================
#  PairWiseDistances
# =================================================================================================
@pytest.mark.parametrize(
    "factory_method, expected_cls",
    [
        (PairWiseDistances.eager, PairWiseDistances_Eager),
        (PairWiseDistances.lazy, PairWiseDistances_Lazy),
    ],
)
def test_pair_wise_distances_factory_methods(vectors, factory_method, expected_cls):
    # --- act ---------------------------------------------
    pwd = factory_method(vectors)

    # --- assert ------------------------------------------
    assert isinstance(pwd, PairWiseDistances)
    assert isinstance(pwd, expected_cls)


# =================================================================================================
#  PairWiseDistances_Eager
# =================================================================================================
def test_pair_wise_distances_eager(vectors, true_distances_l1, true_distances_l2):
    # --- arrange -----------------------------------------
    pwd_l1 = PairWiseDistances.eager(vectors, DistanceMetric.L1_Manhattan)
    pwd_l2 = PairWiseDistances.eager(vectors, DistanceMetric.L2_Euclidean)
    m = true_distances_l2.shape[0]

    # --- act & assert ------------------------------------
    for _ in range(2):
        # repeat 2x to make sure caching works correctly, if present
        for i in range(m):
            for j in range(m):
                assert pwd_l1(i, j) == pytest.approx(true_distances_l1[i, j])
                assert pwd_l2(i, j) == pytest.approx(true_distances_l2[i, j])


# =================================================================================================
#  PairWiseDistances_Lazy
# =================================================================================================
def test_pair_wise_distances_lazy(vectors, true_distances_l1, true_distances_l2):
    # --- arrange -----------------------------------------
    pwd_l1 = PairWiseDistances.lazy(vectors, DistanceMetric.L1_Manhattan)
    pwd_l2 = PairWiseDistances.lazy(vectors, DistanceMetric.L2_Euclidean)
    m = true_distances_l2.shape[0]

    # --- act & assert ------------------------------------
    for _ in range(2):
        # repeat 2x to make sure caching works correctly, if present
        for i in range(m):
            for j in range(m):
                assert pwd_l1(i, j) == pytest.approx(true_distances_l1[i, j])
                assert pwd_l2(i, j) == pytest.approx(true_distances_l2[i, j])
