import pytest
from matplotlib import pyplot as plt


@pytest.fixture
def fig_ax() -> tuple[plt.Figure, plt.Axes]:
    # create fig, ax
    fig, ax = plt.subplots()

    # yield
    yield fig, ax

    # cleanup
    plt.close(fig)
