import numpy as np

from example._core import sum_of_rows


def call1():
    """Succeeds by passing a reference of the input array to the rust component."""
    return sum_of_rows(np.array([[1, 2, 3], [4, 5, 6]], dtype="uint32"))


def call2():
    """Succeeds but allocates extra memory for rebuilding the input array inside the rust component."""
    return sum_of_rows([[1, 2, 3], iter([4, 5, 6]), np.array([7, 8, 9], dtype="int64")])


def call3():
    """Raises an error since the input array is expected to be two dimensional."""
    return sum_of_rows([1, 2, 3])


def call4():
    """Raises an error since the input array contains a value which cannot be safely casted to u32."""
    return sum_of_rows([[2**32, 0, 0], [0, 0, 0]])
