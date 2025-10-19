from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from brtp.math.aggregation import ordered_weighted_mean

from ._constraints import FairnessConstraint
from ._distances import PairWiseDistances


class SolverState:
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        vectors: np.ndarray,
        cons: list[FairnessConstraint],
        dist: PairWiseDistances,
        i_selected: set[int] | None = None,
        i_unselected: set[int] | None = None,
    ):
        # primary fields
        self._vectors = vectors
        self._m = vectors.shape[0]
        self._n = vectors.shape[1]
        self._cons = cons
        self._dist = dist
        self._i_selected = i_selected or set()
        self._i_unselected = i_unselected or (set(range(self._m)) - self._i_selected)

        # caching
        self.__score_cache: tuple[float, float] | None = None

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def n_selected(self) -> int:
        return len(self._i_selected)

    def copy(self) -> SolverState:
        """
        Create smart copy of the state, such that...
          - copies cannot modify original (i.e. selection sets are deep-copied)
          - we reuse the distance computation object (to avoid unnecessary computations)
        """
        return SolverState(
            vectors=self._vectors,
            cons=self._cons,
            dist=self._dist,
            i_selected=self._i_selected.copy(),
            i_unselected=self._i_unselected.copy(),
        )

    def modify(self, add: Iterable[int] = (), remove: Iterable[int] = ()):
        """Modify state in-place by adding or removing indices from selection."""

        modified = False

        for a in add:
            if a in self._i_selected:
                raise ValueError(f"Cannot add {a}: already selected")
            else:
                self._i_selected.add(a)
                self._i_unselected.remove(a)
                modified = True

        for r in remove:
            if r in self._i_unselected:
                raise ValueError(f"Cannot remove {r}: wasn't selected")
            else:
                self._i_selected.remove(r)
                self._i_unselected.add(r)
                modified = True

        if modified:
            self._clear_cached_values()

    def score(self) -> tuple[float, float]:
        # --- check cache ---------------------------------
        if self.__score_cache is None:
            # --- compute constraint metric ---
            n_selected_per_con = [len(con.indices.intersection(self._i_selected)) for con in self._cons]
            con_violations = [
                max(0.0, n - con.ub, con.lb - n) / con.group_size  # make violations relative to len(con.indices)
                for n, con in zip(n_selected_per_con, self._cons)
            ]
            metric_cons = sum([v * v for v in con_violations])

            # --- compute separation metric ---
            sep_per_vector = [math.inf] * self._m  # distance to nearest neighbor of each selected vector
            n_selected = len(self._i_selected)
            i_selected_sorted = sorted(self._i_selected)
            for idx1 in range(n_selected):
                for idx2 in range(idx1):
                    i, j = i_selected_sorted[idx1], i_selected_sorted[idx2]
                    d = self._dist(i, j)
                    sep_per_vector[idx1] = min(sep_per_vector[idx1], d)
                    sep_per_vector[idx2] = min(sep_per_vector[idx2], d)

            weighted_mean_sep = ordered_weighted_mean(sep_per_vector, c=-1)
            if weighted_mean_sep == 0.0:
                metric_sep = math.inf
            else:
                metric_sep = 1 / weighted_mean_sep

            # --- store ---
            self.__score_cache = (metric_cons, metric_sep)

        # --- return --------------------------------------
        return self.__score_cache

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _clear_cached_values(self):
        self.__score_cache = None

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        constraints: list[FairnessConstraint],
        dist: PairWiseDistances | None = None,
    ) -> SolverState:
        return SolverState(
            vectors=vectors,
            cons=constraints,
            dist=dist or PairWiseDistances.eager(vectors),
        )
