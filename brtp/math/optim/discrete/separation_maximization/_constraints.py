from dataclasses import dataclass


@dataclass(frozen=True)
class FairnessConstraint:
    indices: frozenset[int]
    lb: int | None = None
    ub: int | None = None

    def __post_init__(self):
        if self.lb is None and self.ub is None:
            raise ValueError("At least one of 'lb' or 'ub' must be specified.")
        if self.lb is not None and self.lb < 0:
            raise ValueError("'lb' must be non-negative.")
        if self.ub is not None and self.ub < 0:
            raise ValueError("'ub' must be non-negative.")
        if self.lb is not None and self.ub is not None and self.lb > self.ub:
            raise ValueError("'lb' cannot be greater than 'ub'.")

    @property
    def sorted_indices(self) -> list[int]:
        return sorted(self.indices)
