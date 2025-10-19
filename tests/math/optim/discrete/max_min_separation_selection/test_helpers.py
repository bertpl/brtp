import math

import numpy as np
import pytest

from brtp.math.optim.discrete import min_separation


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
