from unittest.mock import Mock

import pytest
from matplotlib import pyplot as plt

from brtp.plotting.canvas import Canvas, CanvasRange, LineStyle, RangeSpecs
from brtp.plotting.utils import Transform


# =================================================================================================
#  Fixtures
# =================================================================================================
@pytest.fixture
def canvas_range() -> CanvasRange:
    x_transform = Transform.linear((0.0, 1.0), (0.0, 10.0))
    y_transform = Transform.linear((0.0, 1.0), (10.0, 11.0), reverse=True)
    z_transform = Transform.linear((0.0, 1.0), (4.0, 5.0))
    return CanvasRange(x_transform, y_transform, z_transform)


# =================================================================================================
#  Tests
# =================================================================================================
def test_canvas_props(canvas_range):
    # --- arrange -----------------------------------------
    ax = Mock()
    canvas = Canvas(canvas_range, ax=ax)

    # --- act ---------------------------------------------
    user_range = canvas.user_range
    fig_range = canvas.fig_range
    ax_obj = canvas.ax

    # --- assert ------------------------------------------
    assert isinstance(user_range, RangeSpecs)
    assert isinstance(fig_range, RangeSpecs)
    assert isinstance(ax_obj, Mock)

    assert user_range.x_min == 0.0
    assert user_range.x_max == 1.0

    assert fig_range.x_min == 0.0
    assert fig_range.y_min == 10.0
    assert fig_range.z_min == 4.0

    assert ax_obj is ax


def test_canvas_plot(fig_ax, canvas_range):
    # --- arrange -----------------------------------------
    fig, ax = fig_ax
    canvas = Canvas(canvas_range, ax=ax)
    ls = LineStyle(color="red", width=2.0, zorder=0.5)

    # --- act ---------------------------------------------
    canvas.plot([0.0, 0.5, 1.0], [10.0, 10.5, 11.0], ls)

    # --- assert ------------------------------------------
    lines = ax.get_lines()
    assert len(lines) == 1
    line = lines[0]
    assert line.get_color() == "red"
    assert line.get_linewidth() == 2.0
    assert line.get_zorder() == 4.5


def test_canvas_plot_sample(fig_ax, canvas_range):
    # --- arrange -----------------------------------------
    fig, ax = fig_ax
    canvas = Canvas(canvas_range, ax=ax)
    ls = LineStyle(color="orange", width=0.5, zorder=0.1)

    # --- act ---------------------------------------------
    canvas.plot_sample(0.2, 0.8, 0.5, ls)

    # --- assert ------------------------------------------
    lines = ax.get_lines()
    assert len(lines) == 2  # 2nd one represents the marker
    line = lines[0]
    assert line.get_color() == "orange"
    assert line.get_linewidth() == 0.5
    assert line.get_zorder() == 4.1


def test_canvas_plot_vline(fig_ax, canvas_range):
    # --- arrange -----------------------------------------
    fig, ax = fig_ax
    canvas = Canvas(canvas_range, ax=ax)
    ls = LineStyle(color="blue", width=1.0, zorder=0.2)

    # --- act ---------------------------------------------
    canvas.vline(0.3, ls, y_min=None, y_max=None)
    canvas.vline([0.5, 0.7, 0.8], ls, y_min=0.1, y_max=0.9)

    # --- assert ------------------------------------------
    lines = ax.get_lines()
    assert len(lines) == 4
    for line in lines:
        assert line.get_color() == "blue"
        assert line.get_linewidth() == 1.0
        assert line.get_zorder() == 4.2


def test_canvas_plot_hline(fig_ax, canvas_range):
    # --- arrange -----------------------------------------
    fig, ax = fig_ax
    canvas = Canvas(canvas_range, ax=ax)
    ls = LineStyle(color="green", width=1.5, zorder=0.3)

    # --- act ---------------------------------------------
    canvas.hline(0.2, ls, x_min=0.2, x_max=0.8)
    canvas.hline([0.5, 0.8], ls, x_min=None, x_max=None)

    # --- assert ------------------------------------------
    lines = ax.get_lines()
    assert len(lines) == 3
    for line in lines:
        assert line.get_color() == "green"
        assert line.get_linewidth() == 1.5
        assert line.get_zorder() == 4.3


def test_canvas_rectangle(fig_ax, canvas_range):
    # --- arrange -----------------------------------------
    fig, ax = fig_ax
    canvas = Canvas(canvas_range, ax=ax)

    # --- act ---------------------------------------------
    canvas.rectangle(
        x_min=0.2,
        x_max=0.6,
        y_min=0.3,
        y_max=0.7,
        fill_color="yellow",
        edgecolor="black",
        linewidth=1.0,
    )

    # --- assert ------------------------------------------
    patches = ax.patches
    assert len(patches) == 1
    rect = patches[0]
    assert rect.get_facecolor() == (1.0, 1.0, 0.0, 1.0)  # RGBA for yellow
    assert rect.get_edgecolor() == (0.0, 0.0, 0.0, 1.0)  # RGBA for black
    assert rect.get_linewidth() == 1.0
