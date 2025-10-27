from functools import lru_cache
from typing import Iterable

import numpy as np

from brtp.compat import numba


# =================================================================================================
#  Regular mean
# =================================================================================================
def mean(values: Iterable[int | float]) -> float:
    """compute arithmetic mean of provided values"""
    values = list(values)
    if len(values) == 0:
        return 0.0
    else:
        return float(np.mean(values))


def weighted_mean(values: Iterable[int | float], weights: Iterable[int | float]) -> float:
    """compute weighted arithmetic mean of provided values"""
    values = list(values)
    weights = list(weights)
    if len(values) == 0:
        return 0.0
    else:
        v = np.array(values)
        w = np.array(weights)
        return float(np.sum(w * v) / np.sum(w))


def ordered_weighted_mean(values: Iterable[int | float], c: float) -> float:
    """
    Compute Ordered Weighted Average (OWA) of provided values

    See eg: https://www.sciencedirect.com/science/article/abs/pii/S0020025524001889

    Step-wise procedure:
      - Sort values  (ascending order)
      - Compute weights as w ~ e^(c*(i/(n-1))), with i the index into the array & n the array size
      - Compute weighted average of the sorted values with these weights

    Depending on the 'c' parameter, the 'center of gravity' of the weights will lie at a different quantile
    of the sorted values.  (NOTE: this is correlated to the 'orness' = 1 - quantile of the weighted average)

        c =-10.0        -> 10.0% quantile
        c = -5.0        -> 19.3% quantile
        c = -4.0        -> 23.1% quantile
        c = -3.0        -> 28.1% quantile
        c = -2.0        -> 34.3% quantile
        c = -1.0        -> 41.8% quantile

        c =  0.0        -> 50.0% quantile (regular mean)

        c =  1.0        -> 58.2% quantile
        c =  2.0        -> 65.7% quantile
        c =  3.0        -> 71.9% quantile
        c =  4.0        -> 76.9% quantile
        c =  5.0        -> 80.7% quantile
        c = 10.0        -> 90.0% quantile

    Hence, the net effect is that we compute the mean of the provided values, with emphasis on the
      larger values (c > 0) or smaller values (c < 0).
    """
    if c == 0:
        return mean(values)
    else:
        sorted_values = sorted(values)
        return weighted_mean(
            values=sorted_values,
            weights=_exponential_weights(c, len(sorted_values)),
        )


# =================================================================================================
#  Geometric mean
# =================================================================================================
def geo_mean(values: Iterable[int | float]) -> float:
    """compute geometric mean of provided values"""
    values = list(values)
    if len(values) == 0:
        return 1.0
    if any(v == 0 for v in values):
        return 0.0
    else:
        return float(np.exp(np.mean(np.log(np.array(values)))))


def weighted_geo_mean(values: Iterable[int | float], weights: Iterable[int | float]) -> float:
    """compute weighted geometric mean of provided values"""
    values = list(values)
    weights = list(weights)
    if len(values) == 0:
        return 1.0
    if any((v == 0) and (w > 0) for w, v in zip(weights, values)):
        return 0.0
    else:
        # convert to numpy arrays
        v = np.array(values)
        w = np.array(weights) / sum(weights)  # normalized array of weights
        # prune v,w to only positive weights
        v = v[w != 0]
        w = w[w != 0]
        # compute weighted geometric mean
        return float(np.exp(np.sum(w * np.log(v))))


def ordered_weighted_geo_mean(values: Iterable[int | float], c: float) -> float:
    """
    Compute Ordered Weighted Geometric Average (OWGA) of provided values

    See eg: https://www.sciencedirect.com/science/article/abs/pii/S0020025524001889

    Step-wise procedure:
      - Sort values  (ascending order)
      - Compute weights as w ~ e^(c*(i/(n-1))), with i the index into the array & n the array size
      - Compute weighted geomtetric average of the sorted values with these weights

    Depending on the 'c' parameter, the 'center of gravity' of the weights will lie at a different quantile
    of the sorted values.  (NOTE: this is correlated to the 'orness' = 1 - quantile of the weighted average)

        c =-10.0        -> 10.0% quantile
        c = -5.0        -> 19.3% quantile
        c = -4.0        -> 23.1% quantile
        c = -3.0        -> 28.1% quantile
        c = -2.0        -> 34.3% quantile
        c = -1.0        -> 41.8% quantile

        c =  0.0        -> 50.0% quantile (regular mean)

        c =  1.0        -> 58.2% quantile
        c =  2.0        -> 65.7% quantile
        c =  3.0        -> 71.9% quantile
        c =  4.0        -> 76.9% quantile
        c =  5.0        -> 80.7% quantile
        c = 10.0        -> 90.0% quantile

    Hence, the net effect is that we compute the mean of the provided values, with emphasis on the
      larger values (c > 0) or smaller values (c < 0).
    """
    if c == 0:
        return geo_mean(values)
    else:
        sorted_values = sorted(values)
        return weighted_geo_mean(
            values=sorted_values,
            weights=_exponential_weights(c, len(sorted_values)),
        )


# =================================================================================================
#  Internal
# =================================================================================================
@lru_cache(maxsize=100)
def _exponential_weights(c: float, n: int) -> np.ndarray:
    return _exponential_weights_numba(c, n)


@numba.njit
def _exponential_weights_numba(c: float, n: int) -> np.ndarray:
    """
    Computes n exponential weights with parameter c to be used in weighted_(geo_)mean as follows:

      w = np.exp(c * np.linspace(0.0, 1.0, n)) / max(1, exp(c))

    We avoid computing the exponentiation n times by re-using the previous weight to compute the next one.
    Normalization is done to ensure that the maximum weight is 1.0; we prefer underflow to 0.0 over overflow to inf.
    """
    if n == 1:
        return np.ones(1)
    elif c == 0.0:
        return np.ones(n)
    else:
        factor = np.exp(-abs(c) / (n - 1))  # use -abs(c) to always generating decreasing sequence
        w = np.zeros(n)
        w_i = 1.0
        for i in range(n):
            w[i] = w_i
            w_i *= factor
        if c < 0:
            return w
        else:
            return w[::-1]  # -abs(c) flipped the sign of c, so reverse the array
