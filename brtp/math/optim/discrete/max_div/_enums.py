from enum import IntEnum


# =================================================================================================
#  Distance Metrics
# =================================================================================================
class DistanceMetric(IntEnum):
    """Enum of distance metrics for MaxDivSolver."""

    L1_Manhattan = 1  # L1 distance (Manhattan distance)
    L2_Euclidean = 2  # L2 distance (Euclidean distance)


# =================================================================================================
#  Diversity Metrics
# =================================================================================================
class DiversityMetric(IntEnum):
    """Enum of diversity metrics for MaxDivSolver."""

    SumDist = 1  # sum of all pairwise distances in the selection
    MinDist = 2  # minimum of all pairwise distances in the selection
    GeoMeanDist = 3  # geometric mean of all vectors' distances to their respective nearest neighbors


# =================================================================================================
#  Constraint Violation Metric
# =================================================================================================
class ConstraintViolationMetric(IntEnum):
    """Enum of constraint violation metrics for MaxDivSolver."""

    AbsoluteSum = 1  # sum of absolute constraint violations
    RelativeSum = 2  # sum of relative constraint violations (wrt group size)
    AbsoluteSquaredSum = 11  # sum of squared absolute constraint violations
    RelativeSquaredSum = 12  # sum of squared relative constraint violations (wrt group size)

    def is_relative(self) -> bool:
        """Returns True if the metric is relative, False if absolute."""
        return (self == ConstraintViolationMetric.RelativeSum) or (self == ConstraintViolationMetric.RelativeSquaredSum)

    def is_squared(self) -> bool:
        """Returns True if the metric is squared, False if linear."""
        return (self == ConstraintViolationMetric.AbsoluteSquaredSum) or (
            self == ConstraintViolationMetric.RelativeSquaredSum
        )
