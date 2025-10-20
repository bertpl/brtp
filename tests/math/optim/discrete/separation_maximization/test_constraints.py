import pytest

from brtp.math.optim.discrete.max_sep import FairnessConstraint


def test_fairness_constraint():
    # --- arrange -----------------------------------------
    indices = [0, 1, 2, 10, 5]

    # --- act ---------------------------------------------
    con = FairnessConstraint(
        indices=frozenset(indices),
        lb=2,
        ub=4,
    )

    # --- assert ------------------------------------------
    assert con.sorted_indices == [0, 1, 2, 5, 10]


@pytest.mark.parametrize(
    "indices,lb,ub",
    [
        ([0, 1, 2, 10, 5], None, None),
        ([0, 1, 2, 10, 5], None, -1),
        ([0, 1, 2, 10, 5], -1, None),
        ([0, 1, 2, 10, 5], 4, 3),
        ([0, 1, 2, 10, 5], 10, 10),
        ([0, -1, 2, 10, 5], 2, 4),
    ],
)
def test_fairness_constraint_validation(indices, lb, ub):
    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = FairnessConstraint(frozenset(indices), lb, ub)
