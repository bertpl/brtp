import time

import pytest

from brtp.benchmarking import high_precision_sleep


@pytest.mark.parametrize("total_duration_sec,n_chunks", [(1e-3, 100), (1e-3, 10), (1e-2, 20), (1e-1, 50), (1e-1, 1)])
def test_high_precision_sleep(total_duration_sec: float, n_chunks: int):
    # --- arrange -----------------------------------------
    sleep_per_iter = total_duration_sec / n_chunks

    # --- act ---------------------------------------------
    t_start = time.perf_counter()
    for _ in range(n_chunks):
        high_precision_sleep(sleep_per_iter)
    t_end = time.perf_counter()

    t_total = t_end - t_start

    print(t_total, total_duration_sec)

    # --- assert ------------------------------------------
    assert 0.5 * total_duration_sec <= t_total <= 2.0 * total_duration_sec
