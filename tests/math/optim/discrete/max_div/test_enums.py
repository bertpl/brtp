import pytest

from brtp.math.optim.discrete.max_div import ConstraintViolationMetric


@pytest.mark.parametrize(
    "metric, is_relative, is_squared",
    [
        (ConstraintViolationMetric.AbsoluteSum, False, False),
        (ConstraintViolationMetric.RelativeSum, True, False),
        (ConstraintViolationMetric.AbsoluteSquaredSum, False, True),
        (ConstraintViolationMetric.RelativeSquaredSum, True, True),
    ],
)
def test_constraint_violation_metric_enum(metric: ConstraintViolationMetric, is_relative: bool, is_squared: bool):
    # --- act & assert ------------------------------------
    assert metric.is_relative() == is_relative
    assert metric.is_squared() == is_squared
