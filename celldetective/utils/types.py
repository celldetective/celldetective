import numpy
import numpy as np


def is_integer_array(arr: np.ndarray) -> bool:
    """
    Check if array consists of integers.

    Parameters
    ----------
    arr : ndarray
        Input array.

    Returns
    -------
    bool
        True if all elements are integers, False otherwise.
    """

    # Mask out NaNs
    non_nan_values = arr[arr == arr].flatten()
    test = np.all(np.mod(non_nan_values, 1) == 0)

    if test:
        return True
    else:
        return False


def test_bool_array(array: np.ndarray) -> np.ndarray:
    """
    Convert boolean array to integer array if needed.

    Parameters
    ----------
    array : ndarray
        Input array.

    Returns
    -------
    ndarray
        Integer array if input was boolean, else original array.
    """
    if array.dtype == "bool":
        return np.array(array, dtype=int)
    else:
        return array
