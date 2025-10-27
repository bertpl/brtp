from __future__ import annotations

import math

import numpy as np

from ._constraints import FairnessConstraint
from ._distances import PairWiseDistances
from ._enums import ConstraintViolationMetric, DistanceMetric, DiversityMetric
from ._result import MaxDivResult
from ._solver_state import SolverState


# =================================================================================================
#  Solver
# =================================================================================================
class MaxDivSolver:
    """Solver for Maximum-Diversity problems."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        vectors: np.ndarray,
        k: int,
        constraints: list[FairnessConstraint],
        distance_metric: DistanceMetric = DistanceMetric.L2_Euclidean,
        diversity_metric: DiversityMetric = DiversityMetric.GeoMeanDist,
        constraint_violation_metric: ConstraintViolationMetric = ConstraintViolationMetric.RelativeSquaredSum,
    ):
        # --- validation ------------------------
        if k > vectors.shape[0]:
            raise ValueError(f"Cannot select k={k} elements from only n={vectors.shape[0]} available.")

        # --- core properties -------------------
        self._vectors = vectors
        self._k = k
        self._cons = [FairnessConstraint(indices=set(range(vectors.shape[0])), lb=k, ub=k)] + constraints

        # --- derivative props ------------------
        self._dist = PairWiseDistances.eager(vectors)
        self._state = SolverState.build(vectors, constraints, self._dist)

    # -------------------------------------------------------------------------
    #  Abstract API
    # -------------------------------------------------------------------------
    def solve(self) -> MaxDivResult:
        """
        Solves the Maximum-Diversity problem provided via the constructor of the class.

        We apply the following solving strategy.
        (NOTE: in this initial implementation only PHASE 1 is implemented)

        PHASE 1 - Initial solution
        --------------------------

        Repeat until selection contains k elements:
          - start with an empty initial selection
          - 1-by-1 add a new element to the selection using a greedy approach that adds the element that minimizes
             our metric (see below for metric definition)

        Under most circumstances this should result in a feasible solution with reasonable optimality.  The strategy
        resembles Gonzalez' approach ('farthest point sampling'), which is shown to be optimal within a factor of 2
        for minimum-separation diversity metrics in absence of fairness constraints.


        Phase 2 - Refinement  (to be implemented)
        --------------------

        We apply a configurable number of refinement iterations, in each of which we evaluate if replacing 1 of the
        selected values with an unselected one would improve the metric.

        Repeat <n_refine> times:
          - select element I of selected subset
          - select element J of unselected subset
          - evaluate if replacing I with J would improve the metric and if so, apply the swap

        FURTHER REFINEMENTS:
         - make smart selections of I to improve the probability of a successful swap
         - make informed selection of J - taking into account constraints - to avoid swaps that violate constraints
         - implement multi-element swaps, to unlock improvements that would not be achievable with single-element swaps

        Optimization Metric
        -------------------

        Solving this constrained optimization problem, is done by defining a 2-element metric:

            (constraint_metric, diversity_metric)

        We try to minimize both these metrics with priority for the first.  We use tuple-comparison to establish this.

            constraint_metric:    0.0 if all constraints are satisfied.  Higher the more constraints are violated more
                                  severely.  Appropriate heuristic weighting is implemented to ensure constraints that
                                  seem harder to satisfy are focussed on first.

            diversity_metric:     This is taken as the reciprocal of the 'mean separation'.  This is chosen as the
                                  geometric mean of the distances of all selected elements to their respective nearest
                                  neighbors.

                                  This should...
                                    - maximize the minimum separation between chosen elements
                                    - drive the solution towards a uniform spread across the space spanned by all elements
                                        (geo_mean of a fixed-range set of elements is minimal for equidistant distribution)
                                    - also help achieve the above in parts of the search space where separation is larger
                                       than the global minimum separation due to imposed fairness constraints.

        :return: MaxDivResult
        """

        # --- PHASE 1 -------------------------------------
        while self._state.n_selected() < self._k:
            best_metric = (math.inf, math.inf)
            best_state = None

            for i in self._state.i_unselected():
                # --- evaluate metric of adding this element ----
                state_candidate = self._state.copy()
                state_candidate.modify(add=[i])
                metric_candidate = state_candidate.score()

                # --- check if best so far ----------------------
                if (best_state is None) or (metric_candidate < best_metric):
                    best_metric = metric_candidate
                    best_state = state_candidate

            # --- selecte best_state ----------------------
            self._state = best_state

        # --- PHASE 2 -------------------------------------
        pass  # to be implemented later

        # --- FINISH --------------------------------------
        raise NotImplementedError()
