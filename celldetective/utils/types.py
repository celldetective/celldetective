import numpy
import numpy as np


def is_integer_array(arr: np.ndarray) -> bool:

    # Mask out NaNs
    non_nan_values = arr[arr == arr].flatten()
    test = np.all(np.mod(non_nan_values, 1) == 0)

    if test:
        return True
    else:
        return False


def test_bool_array(array):
    if array.dtype == "bool":
        return np.array(array, dtype=int)
    else:
        return array
