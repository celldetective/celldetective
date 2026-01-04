import numpy as np


def is_integer_array(arr: np.ndarray) -> bool:

    # Mask out NaNs
    non_nan_values = arr[arr == arr].flatten()
    test = np.all(np.mod(non_nan_values, 1) == 0)

    if test:
        return True
    else:
        return False
