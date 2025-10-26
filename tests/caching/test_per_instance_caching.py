from brtp.caching import per_instance_cache, per_instance_lru_cache


# =================================================================================================
#  per_instance_lru_cache
# =================================================================================================
def test_per_instance_lru_cache_correct():
    # --- arrange -----------------------------------------
    class MyClass:
        call_count = 0

        @per_instance_lru_cache
        def compute_something(self, x: int) -> int:
            MyClass.call_count += 1
            return x * x

        @per_instance_lru_cache(maxsize=1000, typed=True)
        def compute_something_else(self, y: int) -> int:
            MyClass.call_count += 1
            return y + y

    obj = MyClass()

    # --- act ---------------------------------------------
    result_1 = obj.compute_something(1)
    result_2 = obj.compute_something(2)
    result_3 = obj.compute_something(1)  # should be cached

    result_4 = obj.compute_something_else(3)

    # --- assert ------------------------------------------
    assert obj.compute_something.cache_info().maxsize == 128
    assert obj.compute_something.cache_parameters() == dict(maxsize=128, typed=False)

    assert obj.compute_something_else.cache_info().maxsize == 1000
    assert obj.compute_something_else.cache_parameters() == dict(maxsize=1000, typed=True)

    assert result_1 == 1
    assert result_2 == 4
    assert result_3 == 1
    assert result_4 == 6
    assert MyClass.call_count == 3  # 2 actual calls to compute_something, 1 to compute_something_else


def test_per_instance_lru_cache_independent_instances():
    # --- arrange -----------------------------------------
    class MyClass:
        call_count = 0

        @per_instance_lru_cache
        def compute_something(self, x: int) -> int:
            MyClass.call_count += 1
            return x * x

    obj_1 = MyClass()
    obj_2 = MyClass()

    # --- act ---------------------------------------------
    _ = obj_1.compute_something(3)
    _ = obj_1.compute_something(4)
    _ = obj_2.compute_something(4)  # should not be cached from obj_1
    _ = obj_2.compute_something(5)

    # --- assert ------------------------------------------
    assert MyClass.call_count == 4  # all calls are independent


def test_per_instance_lru_cache_cache_clear():
    # --- arrange -----------------------------------------
    class MyClass:
        call_count = 0

        @per_instance_lru_cache
        def compute_something(self, x: int) -> int:
            MyClass.call_count += 1
            return x * x

    obj_1 = MyClass()
    obj_2 = MyClass()

    # --- act ---------------------------------------------
    _ = obj_1.compute_something(3)
    _ = obj_1.compute_something(4)

    _ = obj_2.compute_something(3)
    _ = obj_2.compute_something(4)
    _ = obj_2.compute_something(5)

    obj_1.compute_something.cache_clear()

    # --- assert ------------------------------------------
    assert MyClass.call_count == 5
    assert obj_1.compute_something.cache_info().currsize == 0
    assert obj_2.compute_something.cache_info().currsize == 3


# =================================================================================================
#  per_instance_cache
# =================================================================================================
def test_per_instance_cache_correct():
    # --- arrange -----------------------------------------
    class MyClass:
        call_count = 0

        @per_instance_cache
        def compute_something(self, x: int) -> int:
            MyClass.call_count += 1
            return x * x

    obj = MyClass()

    # --- act ---------------------------------------------
    result_1 = obj.compute_something(1)
    result_2 = obj.compute_something(2)
    result_3 = obj.compute_something(1)  # should be cached

    # --- assert ------------------------------------------
    assert obj.compute_something.cache_parameters()["maxsize"] is None

    assert result_1 == 1
    assert result_2 == 4
    assert result_3 == 1
    assert MyClass.call_count == 2  # only two actual calls


def test_per_instance_cache_independent_instances():
    # --- arrange -----------------------------------------
    class MyClass:
        call_count = 0

        @per_instance_cache
        def compute_something(self, x: int) -> int:
            MyClass.call_count += 1
            return x * x

    obj_1 = MyClass()
    obj_2 = MyClass()

    # --- act ---------------------------------------------
    _ = obj_1.compute_something(3)
    _ = obj_1.compute_something(4)
    _ = obj_2.compute_something(4)  # should not be cached from obj_1
    _ = obj_2.compute_something(5)

    # --- assert ------------------------------------------
    assert MyClass.call_count == 4  # all calls are independent


def test_per_instance_cache_cache_clear():
    # --- arrange -----------------------------------------
    class MyClass:
        call_count = 0

        @per_instance_cache
        def compute_something(self, x: int) -> int:
            MyClass.call_count += 1
            return x * x

    obj_1 = MyClass()
    obj_2 = MyClass()

    # --- act ---------------------------------------------
    _ = obj_1.compute_something(3)
    _ = obj_1.compute_something(4)

    _ = obj_2.compute_something(3)
    _ = obj_2.compute_something(4)
    _ = obj_2.compute_something(5)

    obj_1.compute_something.cache_clear()

    # --- assert ------------------------------------------
    assert MyClass.call_count == 5
    assert obj_1.compute_something.cache_info().currsize == 0
    assert obj_2.compute_something.cache_info().currsize == 3
