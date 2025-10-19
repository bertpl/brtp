import math

import numpy as np
import pytest

from brtp.math.aggregation import ordered_weighted_mean
from brtp.math.optim.discrete.max_div import mean_separation, min_separation


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
    "vectors, power, expected_result",
    [
        (np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 10.0]]), 0, (0.5 + 0.5 + 10) / 3),
        (np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]), 0, 5.0),
        (np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 5.0]]), 0, (1 + 1 + 5) / 3),
        (np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]]), -1, 5.0),
        (np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 10.0]]), -1, ordered_weighted_mean([0.5, 0.5, 10], -1)),
        (np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 10.0]]), -3, ordered_weighted_mean([0.5, 0.5, 10], -3)),
    ],
)
def test_mean_separation(vectors: np.ndarray, power: float, expected_result: float):
    # --- act ---------------------------------------------
    result = mean_separation(vectors, power)

    # --- assert ------------------------------------------
    assert np.isclose(result, expected_result)
