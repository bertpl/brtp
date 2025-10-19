import numpy as np
from scipy.spatial.distance import pdist, squareform

from brtp.math.aggregation import ordered_weighted_mean


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


def mean_separation(vectors: np.ndarray, power: float = -1.0) -> float:
    """
    Computes the average separation of all vectors from their respective nearest neighbors, given M vectors in N dims.
    Averaging is done using ordered weighted (arithmetic) mean, with configurable power.  By default, power=-1 is used,
    which emphasizes the shorter distances.  Using this setting in an optimizer, should help drive solutions towards
    an uniform spread across the search space, whenever possible, while still making every point's distance to its
    nearest neighbor count towards the average - as opposed to min_separation.

    :param vectors: (M x N ndarray) A set of M vectors in N dimensions.
    :param power: (float) Power to use when computing the ordered weighted average.
    :return: (float >= 0) The mean pairwise separation between the vectors (Euclidean distance).
    """
    d_square = squareform(pdist(vectors, metric="euclidean"))
    np.fill_diagonal(d_square, np.inf)
    nearest_distances = np.min(d_square, axis=1)
    return float(ordered_weighted_mean(nearest_distances, c=power))
