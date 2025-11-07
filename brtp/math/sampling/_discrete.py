"""
Utilities to generate 1 or more samples from discrete-valued collections and/or distributions.
"""

import numpy as np

from brtp.compat import numba


# =================================================================================================
#  sample_discrete
# =================================================================================================
def sample_discrete(
    values: list[int],
    n: int = 1,
    p: list[float] | np.ndarray | None = None,
    replace: bool = True,
    seed: int | None = None,
) -> list[int]:
    pass  # TODO


@numba.njit
def _sample_discrete_numba(
    values: np.ndarray[int],
    n: int,
    p: np.ndarray[int] | None = None,
    replace: bool = True,
    seed: int | None = None,
) -> np.ndarray[int]:
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(values, size=n, replace=replace, p=p)


# =================================================================================================
#  sample_constrained
# =================================================================================================
pass  # TODO
