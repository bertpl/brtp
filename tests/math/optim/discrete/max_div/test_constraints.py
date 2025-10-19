import pytest

from brtp.math.optim.discrete.max_div import FairnessConstraint


def test_fairness_constraint():
    # --- arrange -----------------------------------------
    indices = [0, 1, 2, 10, 5]

    # --- act ---------------------------------------------
    con = FairnessConstraint(
        indices=set(indices),
        lb=2,
        ub=4,
    )

    # --- assert ------------------------------------------
    assert con.sorted_indices == [0, 1, 2, 5, 10]


def test_fairness_constraint_normalization():
    # --- arrange -----------------------------------------
    indices = {0, 1, 2, 3, 4}

    # --- act ---------------------------------------------
    con_1 = FairnessConstraint(indices)
    con_2 = FairnessConstraint(indices, lb=0, ub=5)
    con_3 = FairnessConstraint(indices, lb=-1, ub=6)
    con_4 = FairnessConstraint(indices, lb=1, ub=4)

    # --- assert ------------------------------------------
    assert (con_1.lb, con_1.ub) == (0, 5)
    assert (con_2.lb, con_2.ub) == (0, 5)
    assert (con_3.lb, con_3.ub) == (0, 5)
    assert (con_4.lb, con_4.ub) == (1, 4)


@pytest.mark.parametrize(
    "indices,lb,ub",
    [
        ([0, 1, 2, 10, 5], 0, -1),
        ([0, 1, 2, 10, 5], 4, 3),
        ([0, 1, 2, 10, 5], 10, 10),
        ([0, -1, 2, 10, 5], 2, 4),
    ],
)
def test_fairness_constraint_validation(indices, lb, ub):
    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = FairnessConstraint(set(indices), lb, ub)
