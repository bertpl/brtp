import numpy as np
import pytest

from brtp.plotting.canvas import CanvasRange, RangeSpecs
from brtp.plotting.utils import Transform


# =================================================================================================
#  Fixtures
# =================================================================================================
@pytest.fixture
def xyx_transforms() -> tuple[Transform, Transform, Transform]:
    x_transform = Transform.linear((0.0, 10.0), (0.0, 100.0))
    y_transform = Transform.log((1.0, 100.0), (0.0, 200.0), reverse=True)
    z_transform = Transform.linear((0.0, 2.0), (10.0, 12.0))
    return x_transform, y_transform, z_transform


# =================================================================================================
#  Tests
# =================================================================================================
def test_canvas_range_transform_props(xyx_transforms):
    # --- arrange -----------------------------------------
    x_transform, y_transform, z_transform = xyx_transforms

    # --- act ---------------------------------------------
    canvas_range = CanvasRange(x_transform, y_transform, z_transform)

    # --- assert ------------------------------------------
    assert canvas_range.x_transform is x_transform
    assert canvas_range.y_transform is y_transform
    assert canvas_range.z_transform is z_transform


def test_canvas_range_user_fig_ranges(xyx_transforms):
    # --- arrange -----------------------------------------
    x_transform, y_transform, z_transform = xyx_transforms

    # --- act ---------------------------------------------
    canvas_range = CanvasRange(x_transform, y_transform, z_transform)

    # --- assert ------------------------------------------
    assert canvas_range.user_range == RangeSpecs(
        x_min=0.0,
        x_max=10.0,
        y_min=1.0,
        y_max=100.0,
        z_min=0.0,
        z_max=2.0,
        top=1.0,
        bottom=100.0,
        left=0.0,
        right=10.0,
    )
    assert canvas_range.fig_range == RangeSpecs(
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=200.0,
        z_min=10.0,
        z_max=12.0,
        top=200.0,
        bottom=0.0,
        left=0.0,
        right=100.0,
    )


def test_canvas_range_transforms(xyx_transforms):
    # --- arrange -----------------------------------------
    x_transform, y_transform, z_transform = xyx_transforms
    canvas_range = CanvasRange(x_transform, y_transform, z_transform)

    user_x_range = np.linspace(x_transform.user_range()[0], x_transform.user_range()[1], 100)
    user_y_range = np.linspace(y_transform.user_range()[0], y_transform.user_range()[1], 100)
    user_z_range = np.linspace(z_transform.user_range()[0], z_transform.user_range()[1], 100)

    # --- act ---------------------------------------------
    fig_x_range, fig_y_range, fig_z_range = canvas_range.user_to_fig(user_x_range, user_y_range, user_z_range)
    user_x_range_2, user_y_range_2, user_z_range_2 = canvas_range.fig_to_user(fig_x_range, fig_y_range, fig_z_range)

    # --- assert ------------------------------------------
    assert np.allclose(user_x_range_2, user_x_range)
    assert np.allclose(user_y_range_2, user_y_range)
    assert np.allclose(user_z_range_2, user_z_range)
