import pytest

from brtp.benchmarking import benchmark, high_precision_sleep


@pytest.mark.parametrize("t_sleep", [1e-5, 1e-4, 1e-3])
@pytest.mark.parametrize("silent", [True, False])
@pytest.mark.flaky(reruns=10, reruns_delay=0.1)  # benchmark tests are flaky in GitHub Actions
def test_micro_benchmark(t_sleep: float, silent: bool):
    # --- arrange -----------------------------------------
    def f_test():
        high_precision_sleep(t_sleep)

    # --- act ---------------------------------------------
    t_est = benchmark(f_test, t_per_run=0.1, n_warmup=5, n_benchmark=10, silent=silent)

    # --- assert ------------------------------------------
    assert 0.5 * t_sleep <= t_est <= 2 * t_sleep
