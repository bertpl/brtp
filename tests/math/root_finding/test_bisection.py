import math
from typing import Callable

import pytest

from brtp.math.root_finding import bisection
from brtp.math.utils import EPS


def fun_cubic(x):
    return x**3 - 3  # Roots at math.cbrt(3)


def fun_linear(x):
    return x  # Root at 0.0


@pytest.mark.parametrize(
    "fun,a,b,x_tol,x_true,max_diff",
    [
        (fun_cubic, 1.0, 2.0, 1e-10, math.cbrt(3), 1e-10),
        (fun_cubic, -3.0, 2.0, 1e-3, math.cbrt(3), 1e-3),
        (fun_cubic, -1e6, 2.0, 1e-50, math.cbrt(3), EPS),
        (fun_cubic, 1.0, 1e6, 0.0, math.cbrt(3), EPS),
        (fun_cubic, 1.0, 1e6, -1.0, math.cbrt(3), EPS),
        (fun_linear, -math.e, math.pi, 1e-10, 0.0, 1e-10),
        (fun_linear, -math.e, math.pi, 1e-3, 0.0, 1e-3),
        (fun_linear, -math.e, math.pi, 1e-20, 0.0, 1e-20),
        (fun_linear, -math.pi, math.e, 1e-50, 0.0, EPS * EPS),
        (fun_linear, -math.pi, math.e, 0.0, 0.0, EPS * EPS),
        (fun_linear, -math.pi, math.e, -1.0, 0.0, EPS * EPS),
    ],
)
def test_bisection(fun: Callable, a: float, b: float, x_tol, x_true: float, max_diff: float):
    # --- arrange -----------------------------------------
    x_min, x_max = x_true - max_diff, x_true + max_diff

    # --- act ---------------------------------------------
    x = bisection(fun, a, b, x_tol=x_tol)

    # --- assert ------------------------------------------
    assert x_min <= x <= x_max, f"Root {x} not in expected range [{x_min}, {x_max}]"


def test_bisection_exceptions():
    # --- arrange -----------------------------------------
    a, b = 1.0, 2.0

    # --- act/assert --------------------------------------
    with pytest.raises(ValueError):
        bisection(fun_cubic, 10.0, 20.0)

    with pytest.raises(ValueError):
        bisection(fun_cubic, -20.0, -10.0)

    with pytest.raises(ValueError):
        bisection(fun_linear, 1.0, 2.0)
