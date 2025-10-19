from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._constraints import FairnessConstraint


@dataclass
class MaxDivResult:
    # --- original problem ----------------------
    vectors: np.ndarray
    constraints: list[FairnessConstraint]

    # --- the solution --------------------------
    i_selected: set[int]

    # --- solution properties -------------------
    constraint_metric: float  # 0.0 if no constraints are violated
    diversity_metric: float  # reciprocal of separation metric

    # @cached_property
    # def mean_separation(self) -> float:
    #     dist_nearest = [
    #         min([ for j in self.i_selected])
    #         for i in self.i_selected
    #     ]
    #
    # @cached_property
    # def cons_satisfied(self) -> list[bool]:
    #     return [v==0 for v in self.con_violations]
    #
    # @cached_property
    # def con_violations(self) -> list[int]:
    #     n_per_con = [len(con.indices.intersection(self.i_selected)) for con in self.constraints]
    #     return [
    #         max(
    #             0,
    #             (n - con.ub) if con.ub is not None else 0,
    #             (con.lb - n) if con.lb is not None else 0,
    #         )
    #         for n, con in zip(n_per_con, self.constraints)
    #     ]
    #
    # @property
    # def sorted_indices(self) -> list[int]:
    #     return sorted(self.i_selected)
