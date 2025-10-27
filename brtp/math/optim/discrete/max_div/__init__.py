"""
This module contains optimizers to solve "Dispersion" or "Maximum Diversity Sub-Selection" Problems.

These problems can be summarized as follows:

GIVEN
 - a set of M vectors in N dimensions  (represented as a MxN matrix)
 - a desired number of vectors to select 2 <= K <= M
 - additional 'fairness' constraints imposing min/max # of vectors to choose from certain sub-sets

REQUESTED
 - binary decision vector S of size M summing to K, indicating which of the M vectors to select,
   such that the "mean separation" is maximized.

       "mean separation" -> We look at the Ordered Weighted Average (OWA) of distances of all selected vectors to their
                              respective nearest neighbors (within the selection). The OWA operator uses exponential
                              weighting emphasizing smaller values.

                            This ensures the problem is more well-defined (not just 1 pair of vectors determines
                              the final metric) and balanced (we want to keep all vectors well-separated from
                              their neighbors and as uniformly spread as possible).
"""

from ._constraints import FairnessConstraint
from ._distances import PairWiseDistances, PairWiseDistances_Eager, PairWiseDistances_Lazy
from ._enums import ConstraintViolationMetric, DistanceMetric, DiversityMetric
from ._metrics import mean_separation, min_separation
