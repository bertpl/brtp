"""
This module contains optimizers to solve "separation maximization" problems.

These problems can be summarized as follows:

GIVEN
 - a set of M vectors in N dimensions  (represented as a MxN matrix)
 - a desired number of vectors to select 2 <= K <= M

REQUESTED
 - binary decision vector S of size M summing to K, indicating which of the M vectors to select,
   such that the "separation" is maximized.

   --> Different flavors exist, but for now we will focus on "(fair) mean separation maximization", i.e.
       "fair"            -> we can impose some constraints on min/max # of vectors to choose
                                                  from certain sub-sets of vectors (to impose 'fairness' of selection)
       "mean separation" -> we look at the mean distance of all selected vectors to their respective nearest neighbors
                              (within the selection).  This ensures the problem is more well-defined (not just 1 pair
                              of vectors determines the final metric) and balanced (we want to keep all vectors
                              well-separated from their neighbors).
"""

from ._constraints import FairnessConstraint
from ._helpers import CachedDistances, mean_separation, min_separation
