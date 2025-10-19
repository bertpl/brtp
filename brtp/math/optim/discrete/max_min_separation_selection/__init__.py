"""
This module contains optimizers to solve "max-min separation selection" problems.

These problems can be formulated as follows:

GIVEN
 - a set of M vectors in N dimensions  (represented as a MxN matrix)
 - a desired number of vectors to select 2 <= K <= M

REQUESTED
 - binary decision vector S of size M summing to K, indicating which of the M vectors to select,
   such that the minimum pairwise distance observed between the selected vectors is maximized.
"""

from ._helpers import min_separation
