import numpy as np
from scipy.spatial.distance import pdist, squareform

from brtp.compat import numba


# =================================================================================================
#  Metrics
# =================================================================================================
def min_separation(vectors: np.ndarray) -> float:
    """
    Compute minimum separate between a set of M vectors in N dimensions.
    :param vectors: (M x N ndarray) A set of M vectors in N dimensions.
    :return: (float >= 0) The minimum pairwise separation between the vectors (Euclidean distance).
    """
    return float(np.min(pdist(vectors, metric="euclidean")))


def mean_separation(vectors: np.ndarray) -> float:
    """
    Computes the average separation of all vectors from their respective nearest neighbors, given M vectors in N dims.
    :param vectors: (M x N ndarray) A set of M vectors in N dimensions.
    :return: (float >= 0) The mean pairwise separation between the vectors (Euclidean distance).
    """
    d_square = squareform(pdist(vectors, metric="euclidean"))
    np.fill_diagonal(d_square, np.inf)
    nearest_distances = np.min(d_square, axis=1)
    return float(np.mean(nearest_distances))


# =================================================================================================
#  Distance calculation
# =================================================================================================
class CachedDistances:
    def __init__(self, vectors: np.ndarray):
        self.__vectors = vectors
        self.__cache: dict[tuple[int, int], float] = dict()

    def dist(self, i: int, j: int) -> float:
        i, j = min(i, j), max(i, j)  # exploit dist(i,j) == dist(j,i) to avoid cache misses
        if (result := self.__cache.get((i, j))) is not None:
            return result
        else:
            result = _compute_distance(self.__vectors, i, j)
            self.__cache[(i, j)] = result
            return result


@numba.njit
def _compute_distance(vectors: np.ndarray, i: int, j: int) -> float:
    s = 0.0
    for k in range(vectors.shape[1]):
        d = vectors[i, k] - vectors[j, k]
        s += d * d
    return np.sqrt(s)
