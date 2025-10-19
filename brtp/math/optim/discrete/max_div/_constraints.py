from dataclasses import dataclass
from functools import cached_property


# =================================================================================================
#  Single Constraint
# =================================================================================================
@dataclass
class FairnessConstraint:
    indices: set[int]  # assumed to be smaller than 1e12 elements (which wouldn't fit in memory on most systems anyway)
    lb: int = 0
    ub: int = int(1e12)  # will be replaced with min(ub, len(indices))

    def __post_init__(self):
        # normalize bounds
        self.lb = max(self.lb, 0)
        self.ub = min(self.ub, len(self.indices))

        # validation checks
        if self.ub < 0:
            raise ValueError("'ub' must be non-negative.")
        if self.lb > len(self.indices):
            raise ValueError("'lb' must be <= len(indices).")
        if self.lb > self.ub:
            raise ValueError("'lb' cannot be greater than 'ub'.")
        if min(self.indices) < 0:
            raise ValueError("'indices' must be non-negative.")

    @cached_property
    def sorted_indices(self) -> list[int]:
        return sorted(self.indices)

    @cached_property
    def group_size(self) -> int:
        return len(self.indices)
