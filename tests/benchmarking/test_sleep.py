import time

import pytest

from brtp.benchmarking import high_precision_sleep


@pytest.mark.parametrize("sleep_duration_sec", [1e-5, 1e-4, 1e-3, 1e-2])
@pytest.mark.flaky(reruns=10, reruns_delay=0.1)  # benchmark tests are flaky in GitHub Actions
def test_high_precision_sleep(sleep_duration_sec: float):
    # --- act ---------------------------------------------
    t_start = time.perf_counter()
    high_precision_sleep(sleep_duration_sec)
    t_end = time.perf_counter()

    actual_sleep_sec = t_end - t_start

    # --- assert ------------------------------------------
    assert 0.5 * sleep_duration_sec <= actual_sleep_sec <= 2.0 * sleep_duration_sec
