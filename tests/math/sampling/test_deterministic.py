import pytest

from brtp.math.sampling import linspace, logspace


@pytest.mark.parametrize("min_value, max_value", [(0.0, 1.0), (1.0, 10.0), (1.0, -10.0)])
@pytest.mark.parametrize("n", [2, 5, 10])
@pytest.mark.parametrize("inclusive", [False, True])
def test_linspace(min_value: float, max_value: float, n: int, inclusive: bool):
    # --- arrange -----------------------------------------
    if inclusive:
        expected_first = min_value
        expected_last = max_value
        expected_delta = (max_value - min_value) / (n - 1)
    else:
        expected_first = min_value + (0.5 * (max_value - min_value) / n)
        expected_last = max_value - (0.5 * (max_value - min_value) / n)
        expected_delta = (max_value - min_value) / n

    # --- act ---------------------------------------------
    values = linspace(min_value, max_value, n, inclusive)

    # --- assert ------------------------------------------
    assert isinstance(values, list)
    assert len(values) == n
    assert len(set(values)) == n
    assert values[0] == pytest.approx(expected_first)
    assert values[-1] == pytest.approx(expected_last)
    for i in range(1, n):
        assert values[i] - values[i - 1] == pytest.approx(expected_delta)


@pytest.mark.parametrize("min_value, max_value", [(0.1, 1.0), (1.0, 2.0), (3.0, 0.001)])
@pytest.mark.parametrize("n", [2, 5, 10])
@pytest.mark.parametrize("inclusive", [False, True])
def test_logspace(min_value: float, max_value: float, n: int, inclusive: bool):
    # --- arrange -----------------------------------------
    if inclusive:
        expected_first = min_value
        expected_last = max_value
        expected_ratio = (max_value / min_value) ** (1 / (n - 1))
    else:
        expected_first = min_value * ((max_value / min_value) ** (0.5 / n))
        expected_last = max_value / ((max_value / min_value) ** (0.5 / n))
        expected_ratio = (max_value / min_value) ** (1 / n)

    # --- act ---------------------------------------------
    values = logspace(min_value, max_value, n, inclusive)

    # --- assert ------------------------------------------
    assert isinstance(values, list)
    assert len(values) == n
    assert len(set(values)) == n
    assert values[0] == pytest.approx(expected_first)
    assert values[-1] == pytest.approx(expected_last)
    for i in range(1, n):
        assert values[i] / values[i - 1] == pytest.approx(expected_ratio)
