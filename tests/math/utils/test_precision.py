from brtp.math.utils import EPS


def test_eps():
    assert 1.0 + EPS != 1.0
    assert 1.0 + (EPS / 2) == 1.0
