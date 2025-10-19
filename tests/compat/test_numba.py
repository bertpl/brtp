from brtp.compat import is_numba_installed, numba


def test_is_numba_installed():
    # --- arrange -----------------------------------------
    is_numba_truly_installed = hasattr(numba, "double")

    # --- act ---------------------------------------------
    is_numba_deemed_installed = is_numba_installed()

    # --- assert ------------------------------------------
    assert is_numba_deemed_installed == is_numba_truly_installed


def test_numba_decorator_always_works():
    # --- arrange -----------------------------------------
    @numba.jit
    def func_jit(x):
        return x + 1

    @numba.njit
    def func_njit(x):
        return x + 2

    @numba.njit(fastmath=True)
    def func_njit2(x):
        return x + 3

    # --- act ---------------------------------------------
    result_jit = func_jit(10)
    result_njit = func_njit(10)
    result_njit2 = func_njit2(10)

    # --- assert ------------------------------------------
    assert result_jit == 11, "numba.jit decorator did not work as expected."
    assert result_njit == 12, "numba.njit decorator did not work as expected."
    assert result_njit2 == 13, "numba.njit(...) decorator did not work as expected."
