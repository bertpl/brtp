from time import perf_counter_ns

import pytest

from brtp.benchmarking._micro_benchmark import benchmark


@pytest.mark.parametrize("t_sleep", [1e-5, 1e-4, 1e-3])
@pytest.mark.parametrize("silent", [True, False])
def test_micro_benchmark(t_sleep: float, silent: bool):
    # --- arrange -----------------------------------------
    def f_test():
        t_start = perf_counter_ns()
        t_end = t_start + (1e9 * t_sleep)
        while perf_counter_ns() < t_end:
            pass

    # --- act ---------------------------------------------
    t_est = benchmark(f_test, t_per_run=1e-2, n_warmup=10, n_benchmark=20, silent=silent)

    # --- assert ------------------------------------------
    assert 0.5 * t_sleep <= t_est <= 2 * t_sleep
