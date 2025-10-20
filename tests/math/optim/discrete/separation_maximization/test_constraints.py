import pytest

from brtp.math.optim.discrete.separation_maximization import FairnessConstraint


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


@pytest.mark.parametrize("lb,ub", [(None, None), (None, -1), (-1, None), (10, 5)])
def test_fairness_constraint_validation(lb, ub):
    # --- arrange -----------------------------------------
    indices = frozenset([0, 1, 2, 10, 5])

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = FairnessConstraint(indices, lb, ub)
