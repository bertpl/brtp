import dataclasses
from typing import Any
from unittest.mock import Mock

import pytest

from brtp.plotting.canvas import LineStyle


def test_linestyle_defaults():
    # --- act ---------------------------------------------
    ls = LineStyle()

    # --- assert ------------------------------------------
    assert isinstance(ls, LineStyle)
    assert ls.color == (0.0, 0.0, 0.0)
    assert ls.width == 1.0
    assert ls.style == "-"


@pytest.mark.parametrize("line_enabled", [True, False])
def test_linestyle_line_kwargs(line_enabled: bool):
    # --- arrange ----------------------------------------
    ls = LineStyle(
        color=(0.1, 0.2, 0.3),
        width=2.0,
        style="--",
        line_enabled=line_enabled,
        alpha=0.8,
        zorder=5.0,
    )

    # --- act ---------------------------------------------
    line_kwargs = ls.get_line_kwargs()

    # --- assert ------------------------------------------
    if line_enabled:
        assert line_kwargs == {
            "color": (0.1, 0.2, 0.3),
            "linewidth": 2.0,
            "linestyle": "--",
            "alpha": 0.8,
            "zorder": 5.0,
        }
    else:
        assert line_kwargs == {"linewidth": 0.0}


def test_linestyle_get_marker_kwargs():
    # --- arrange ----------------------------------------
    ls = LineStyle(
        color=(0.4, 0.5, 0.6),
        width=1.5,
        marker="o",
        marker_size=8.0,
        marker_filled=True,
        alpha=0.7,
        zorder=3.0,
    )

    # --- act ---------------------------------------------
    marker_kwargs = ls.get_marker_kwargs()

    # --- assert ------------------------------------------
    assert marker_kwargs == {
        "marker": "o",
        "markersize": 8.0,
        "markerfacecolor": (0.4, 0.5, 0.6, 0.7),
        "markeredgecolor": (0.4, 0.5, 0.6, 0.7),
        "markeredgewidth": 1.5,
        "zorder": 3.0,
    }


@pytest.mark.parametrize(
    "modify_kwargs",
    [
        {"color": (1.0, 0.0, 0.0)},
        {"width": 2.0, "style": "--"},
        {"line_enabled": False, "marker": "o", "marker_size": 5.0},
        {"alpha": 0.5, "zorder": 10.0},
        {"marker_filled": False},
    ],
)
def test_linestyle_modify(modify_kwargs: dict[str, Any]):
    # --- arrange ----------------------------------------
    ls = LineStyle()
    ls_fields = [f.name for f in dataclasses.fields(LineStyle)]

    # --- act ---------------------------------------------
    ls_modified = ls.modify(**modify_kwargs)

    # --- assert ------------------------------------------
    for k, v in modify_kwargs.items():
        assert getattr(ls_modified, k) == v

    for f in ls_fields:
        if f not in modify_kwargs.keys():
            assert getattr(ls_modified, f) == getattr(ls, f)


def test_linestyle_plot():
    # --- arrange -----------------------------------------
    ax = Mock(plot=Mock())
    ls = LineStyle()

    # --- act ---------------------------------------------
    ls.plot(ax, x=[1], y=[0, 1])

    # --- assert ------------------------------------------
    ax.plot.assert_called_once_with([1, 1], [0, 1], **(ls.get_line_kwargs() | ls.get_marker_kwargs()))


def test_linestyle_plot_value_error():
    # --- arrange -----------------------------------------
    ax = Mock(plot=Mock())
    ls = LineStyle()

    # --- act / assert ------------------------------------
    with pytest.raises(ValueError):
        ls.plot(ax, x=[1, 2], y=[0, 1, 2])  # x and y length mismatch


def test_linestyle_plot_sample():
    # --- arrange -----------------------------------------
    ax = Mock(plot=Mock())
    ls = LineStyle()

    # --- act ---------------------------------------------
    ls.plot_sample(ax, x=[1, 2], y=3)

    # --- assert ------------------------------------------
    ax.plot.assert_any_call([1, 2], [3, 3], **ls.get_line_kwargs())  # line sample
    ax.plot.assert_any_call(1.5, 3, **ls.get_marker_kwargs())  # marker at midpoint


def test_linestyle_plot_sample_value_error():
    # --- arrange -----------------------------------------
    ax = Mock(plot=Mock())
    ls = LineStyle()

    # --- act / assert ------------------------------------
    with pytest.raises(ValueError):
        ls.plot_sample(ax, x=[1, 2, 3], y=4)  # x length mismatch for line sample
