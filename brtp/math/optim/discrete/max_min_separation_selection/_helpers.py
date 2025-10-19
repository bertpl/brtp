import numpy as np
from scipy.spatial.distance import pdist


def min_separation(vectors: np.ndarray) -> float:
    """
    Compute minimum separate between a set of M points in N dimensions.
    :param vectors: (M x N ndarray) A set of M vectors in N dimensions.
    :return: (float >= 0) The minimum pairwise separation between the vectors (Euclidean distance).
    """
    return float(np.min(pdist(vectors, metric="euclidean")))
