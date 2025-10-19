from typing import Callable

import numpy as np
import pytest

from brtp.plotting.utils import Transform, TransformLinear, TransformLog


# =================================================================================================
#  Base Class
# =================================================================================================
@pytest.mark.parametrize(
    "factory_method, expected_class",
    [
        (Transform.linear, TransformLinear),
        (Transform.log, TransformLog),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_transform_factory_methods(factory_method: Callable, expected_class: type[Transform], reverse: bool):
    # --- act ---------------------------------------------
    transform = factory_method((0.1, 1.0), (0.0, 100.0), reverse)

    # --- assert ------------------------------------------
    assert isinstance(transform, expected_class)
    assert transform.is_reverse() == reverse
    assert transform.user_range() == (0.1, 1.0)
    assert transform.figure_range() == (0.0, 100.0)


@pytest.mark.parametrize(
    "factory_method, user_range, figure_range, kwargs",
    [
        (Transform.linear, (-0.5, 1.0), (5.0, 75.0), dict()),
        (Transform.log, (0.1, 1.0), (0.0, 100.0), dict()),
        (Transform.lin_log, (-0.5, 100.0), (0.0, 100.0), dict(c_fig_lin_max=0.6, v_user_lin_max=1.0)),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_transform_forward_backward_invariant(
    factory_method: Callable,
    user_range: tuple[float, float],
    figure_range: tuple[float, float],
    kwargs: dict,
    reverse: bool,
):
    # --- arrange -----------------------------------------
    transform: Transform = factory_method(
        user_range=user_range,
        figure_range=figure_range,
        reverse=reverse,
        **kwargs,
    )
    fig_values = np.linspace(figure_range[0], figure_range[1], 100)

    # --- act ---------------------------------------------
    user_values = transform.inv(fig_values)
    fig_values_reconstructed = transform(user_values)

    # --- assert ------------------------------------------
    assert np.allclose(fig_values, fig_values_reconstructed)


@pytest.mark.parametrize(
    "factory_method, user_range, figure_range, kwargs",
    [
        (Transform.linear, (-0.5, 1.0), (5.0, 75.0), dict()),
        (Transform.log, (0.1, 1.0), (0.0, 100.0), dict()),
        (Transform.lin_log, (-0.5, 100.0), (0.0, 100.0), dict(c_fig_lin_max=0.6, v_user_lin_max=1.0)),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_transform_reverse_invariant(
    factory_method: Callable,
    user_range: tuple[float, float],
    figure_range: tuple[float, float],
    kwargs: dict,
    reverse: bool,
):
    # --- arrange -----------------------------------------
    transform: Transform = factory_method(
        user_range=user_range,
        figure_range=figure_range,
        reverse=reverse,
        **kwargs,
    )

    # --- act & assert ------------------------------------

    # test forward transform
    if reverse:
        assert np.allclose(transform(list(user_range)), [figure_range[1], figure_range[0]])
    else:
        assert np.allclose(transform(list(user_range)), [figure_range[0], figure_range[1]])

    # test backward transform
    if reverse:
        assert np.allclose(transform.inv(list(figure_range)), [user_range[1], user_range[0]])
    else:
        assert np.allclose(transform.inv(list(figure_range)), [user_range[0], user_range[1]])


# =================================================================================================
#  Linear
# =================================================================================================
def test_transform_linear_is_linear():
    assert Transform.linear((0.0, 1.0), (0.0, 100.0)).is_linear()


@pytest.mark.parametrize(
    "v_user, expected_v_fig",
    [
        (np.array([0.0, 0.5, 1.0]), np.array([0.0, 50.0, 100.0])),
        ([0.02, 0.7, 0.3], [2, 70, 30]),
        (0.63, 63.0),
    ],
)
def test_transform_linear_forward(v_user, expected_v_fig):
    # --- arrange -----------------------------------------
    transform = TransformLinear((0.0, 1.0), (0.0, 100.0))

    # --- act ---------------------------------------------
    v_fig = transform(v_user)

    # --- assert ------------------------------------------
    assert type(v_fig) == type(expected_v_fig)
    assert np.allclose(v_fig, expected_v_fig)


@pytest.mark.parametrize(
    "v_fig, expected_v_user",
    [
        (np.array([0.0, 50.0, 100.0]), np.array([0.0, 0.5, 1.0])),
        ([2, 70, 30], [0.02, 0.7, 0.3]),
        (63.0, 0.63),
    ],
)
def test_transform_linear_backward(v_fig, expected_v_user):
    # --- arrange -----------------------------------------
    transform = TransformLinear((0.0, 1.0), (0.0, 100.0))

    # --- act ---------------------------------------------
    v_user = transform.inv(v_fig)

    # --- assert ------------------------------------------
    assert type(v_user) == type(expected_v_user)
    assert np.allclose(v_user, expected_v_user)


# =================================================================================================
#  Logarithmic
# =================================================================================================
def test_transform_log_is_linear():
    assert not Transform.log((0.1, 1.0), (0.0, 100.0)).is_linear()


@pytest.mark.parametrize(
    "v_user, expected_v_fig",
    [
        (np.array([0.1, 1.0, 10.0]), np.array([0.0, 50.0, 100.0])),
        ([0.1, 1.0, 10.0], [0.0, 50.0, 100.0]),
        (1.0, 50.0),
    ],
)
def test_transform_log_forward(v_user, expected_v_fig):
    # --- arrange -----------------------------------------
    transform = TransformLog((0.1, 10.0), (0.0, 100.0))

    # --- act ---------------------------------------------
    v_fig = transform(v_user)

    # --- assert ------------------------------------------
    assert type(v_fig) == type(expected_v_fig)
    assert np.allclose(v_fig, expected_v_fig)


@pytest.mark.parametrize(
    "v_fig, expected_v_user",
    [
        (np.array([0.0, 50.0, 100.0]), np.array([0.1, 1.0, 10.0])),
        ([0.0, 50.0, 100.0], [0.1, 1.0, 10.0]),
        (50.0, 1.0),
    ],
)
def test_transform_log_backward(v_fig, expected_v_user):
    # --- arrange -----------------------------------------
    transform = TransformLog((0.1, 10.0), (0.0, 100.0))

    # --- act ---------------------------------------------
    v_user = transform.inv(v_fig)

    # --- assert ------------------------------------------
    assert type(v_user) == type(expected_v_user)
    assert np.allclose(v_user, expected_v_user)


# =================================================================================================
#  Linear-Logarithmic
# =================================================================================================
def test_transform_lin_log_is_linear():
    assert not Transform.lin_log(
        user_range=(0.1, 1.0),
        figure_range=(0.0, 100.0),
        c_fig_lin_max=0.5,
        v_user_lin_max=0.2,
    ).is_linear()


@pytest.mark.parametrize(
    "c_fig_lin_max, v_user_lin_max",
    [
        (0.5, 0.2),
        (0.3, 5.0),
        (0.7, 20.0),
    ],
)
def test_transform_lin_log_mid_point(c_fig_lin_max: float, v_user_lin_max: float):
    # --- arrange -----------------------------------------
    transform = Transform.lin_log(
        user_range=(0.1, 100.0),
        figure_range=(0.0, 100.0),
        c_fig_lin_max=c_fig_lin_max,
        v_user_lin_max=v_user_lin_max,
    )

    # --- act & assert ------------------------------------
    assert np.isclose(transform(v_user_lin_max), c_fig_lin_max * 100.0)
    assert np.isclose(transform.inv(c_fig_lin_max * 100.0), v_user_lin_max)
