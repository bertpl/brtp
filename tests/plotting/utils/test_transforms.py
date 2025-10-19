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


@pytest.mark.parametrize("factory_method", [Transform.linear, Transform.log])
@pytest.mark.parametrize("reverse", [False, True])
def test_transform_forward_backward_invariant(factory_method: Callable, reverse: bool):
    # --- arrange -----------------------------------------
    transform: Transform = factory_method((0.1, 1.0), (0.0, 100.0), reverse)
    fig_values = np.linspace(0.0, 100.0, 100)

    # --- act ---------------------------------------------
    user_values = transform.inv(fig_values)
    fig_values_reconstructed = transform(user_values)

    # --- assert ------------------------------------------
    assert np.allclose(fig_values, fig_values_reconstructed)


@pytest.mark.parametrize("factory_method", [Transform.linear, Transform.log])
@pytest.mark.parametrize("reverse", [False, True])
def test_transform_reverse_invariant(factory_method: Callable, reverse: bool):
    # --- arrange -----------------------------------------
    transform: Transform = factory_method((0.1, 1.0), (0.0, 100.0), reverse)

    # --- act & assert ------------------------------------

    # test forward transform
    if reverse:
        assert np.allclose(transform([0.1, 1.0]), [100.0, 0.0])
    else:
        assert np.allclose(transform([0.1, 1.0]), [0.0, 100.0])

    # test backward transform
    if reverse:
        assert np.allclose(transform.inv([0.0, 100.0]), [1.0, 0.1])
    else:
        assert np.allclose(transform.inv([0.0, 100.0]), [0.1, 1.0])


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
