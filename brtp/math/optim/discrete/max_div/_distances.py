"""
Helper classes to compute pair-wise distances efficiently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import pdist

from brtp.compat import numba

from ._enums import DistanceMetric


# =================================================================================================
#  Base
# =================================================================================================
class PairWiseDistances(ABC):
    def __init__(self, vectors: np.ndarray, metric: DistanceMetric = DistanceMetric.L2_Euclidean):
        """
        Constructor for PairWiseDistances class.
        :param vectors: (M,N)-matrix with vectors stored in rows, i.e. M vectors in N dimensions.
        """
        self._vectors = vectors
        self._metric = metric

    @abstractmethod
    def __call__(self, i: int, j: int) -> float:
        """Compute Euclidean distance between vectors i & j."""
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def lazy(cls, vectors: np.ndarray, metric: DistanceMetric = DistanceMetric.L2_Euclidean) -> PairWiseDistances_Lazy:
        return PairWiseDistances_Lazy(vectors, metric)

    @classmethod
    def eager(
        cls, vectors: np.ndarray, metric: DistanceMetric = DistanceMetric.L2_Euclidean
    ) -> PairWiseDistances_Eager:
        return PairWiseDistances_Eager(vectors, metric)


# =================================================================================================
#  Child Classes
# =================================================================================================
class PairWiseDistances_Eager(PairWiseDistances):
    """
    Class that pre-computes all pair-wise distances eagerly, but efficiently.
    USE CASES: if M*N is small enough (cfr memory) and we anticipate we need to compute most distances anyway
               (e.g. >10%).
    """

    def __init__(self, vectors: np.ndarray, metric: DistanceMetric = DistanceMetric.L2_Euclidean):
        super().__init__(vectors, metric)
        # NOTE: the ._distances matrix is a square matrix in 1D 'condensed' form, i.e only storing
        #       on triangular part of this symmetric matrix with 0 diagonal.
        #       https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
        match metric:
            case DistanceMetric.L1_Manhattan:
                self._distances: np.ndarray = pdist(self._vectors, metric="cityblock")
            case DistanceMetric.L2_Euclidean:
                self._distances: np.ndarray = pdist(self._vectors, metric="euclidean")
        self._m: int = self._vectors.shape[0]

    def __call__(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        else:
            i, j = min(i, j), max(i, j)
            idx = self._m * i + j - ((i + 2) * (i + 1)) // 2
            return float(self._distances[idx])


class PairWiseDistances_Lazy(PairWiseDistances):
    """
    Class that pre-computes all pair-wise distances lazily, to avoid unnecessary computations.
    USE CASES: if M*N is too large (cfr memory) or we anticipate we need to compute only a small subset of distances
               (e.g. <10%).
    """

    def __init__(self, vectors: np.ndarray, metric: DistanceMetric = DistanceMetric.L2_Euclidean):
        super().__init__(vectors, metric)
        self._cache: dict[tuple[int, int], float] = dict()
        match metric:
            case DistanceMetric.L1_Manhattan:
                self._distance_fun = _compute_distance_l1
            case DistanceMetric.L2_Euclidean:
                self._distance_fun = _compute_distance_l2

    def __call__(self, i: int, j: int) -> float:
        i, j = min(i, j), max(i, j)  # exploit dist(i,j) == dist(j,i) to avoid cache misses
        if (result := self._cache.get((i, j))) is not None:
            return result
        else:
            result = self._distance_fun(self._vectors, i, j)
            self._cache[(i, j)] = result
            return result


# =================================================================================================
#  Internal helpers
# =================================================================================================
@numba.njit
def _compute_distance_l1(vectors: np.ndarray, i: int, j: int) -> float:
    s = 0.0
    for k in range(vectors.shape[1]):
        s += abs(vectors[i, k] - vectors[j, k])
    return s


@numba.njit
def _compute_distance_l2(vectors: np.ndarray, i: int, j: int) -> float:
    s = 0.0
    for k in range(vectors.shape[1]):
        d = vectors[i, k] - vectors[j, k]
        s += d * d
    return np.sqrt(s)
