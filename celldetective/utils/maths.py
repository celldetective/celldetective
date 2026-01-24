from typing import Union, List
import numpy as np


from celldetective import get_logger

logger = get_logger()


def step_function(t: Union[np.ndarray, List], t_shift: float, dt: float) -> np.ndarray:
    """
    Computes a step function using the logistic sigmoid function.

    This function calculates the value of a sigmoid function, which is often used to model
    a step change or transition. The sigmoid function is defined as:

    .. math::
            f(t) = \\frac{1}{1 + \\exp{\\left( -\\frac{t - t_{shift}}{dt} \\right)}}

    where `t` is the input variable, `t_shift` is the point of the transition, and `dt` controls
    the steepness of the transition.

    Parameters
    ----------
    t : array_like
            The input values for which the step function will be computed.
    t_shift : float
            The point in the `t` domain where the transition occurs.
    dt : float
            The parameter that controls the steepness of the transition. Smaller values make the
            transition steeper, while larger values make it smoother.

    Returns
    -------
    array_like
            The computed values of the step function for each value in `t`.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.array([0, 1, 2, 3, 4, 5])
    >>> t_shift = 2
    >>> dt = 1
    >>> step_function(t, t_shift, dt)
    array([0.26894142, 0.37754067, 0.5       , 0.62245933, 0.73105858, 0.81757448])
    """

    with np.errstate(over="raise", divide="raise"):
        try:
            return 1 / (1 + np.exp(-(t - t_shift) / dt))
        except FloatingPointError as e:
            logger.warning(
                f"Math warning in step_function: {e}. t_shift={t_shift}, dt={dt}. Range of t: [{np.min(t)}, {np.max(t)}]"
            )
            with np.errstate(over="ignore", divide="ignore"):
                return 1 / (1 + np.exp(-(t - t_shift) / dt))


def derivative(x, timeline, window, mode="bi"):
    """
    Compute the derivative of a given array of values with respect to time using a specified numerical differentiation method.

    Parameters
    ----------
    x : array_like
            The input array of values.
    timeline : array_like
            The array representing the time points corresponding to the input values.
    window : int
            The size of the window used for numerical differentiation. Must be a positive odd integer.
    mode : {'bi', 'forward', 'backward'}, optional
            The numerical differentiation method to be used:
            - 'bi' (default): Bidirectional differentiation using a symmetric window.
            - 'forward': Forward differentiation using a one-sided window.
            - 'backward': Backward differentiation using a one-sided window.

    Returns
    -------
    dxdt : ndarray
            The computed derivative values of the input array with respect to time.

    Raises
    ------
    AssertionError
            If the window size is not an odd integer and mode is 'bi'.

    Notes
    -----
    - For 'bi' mode, the window size must be an odd number.
    - For 'forward' mode, the derivative at the edge points may not be accurate due to the one-sided window.
    - For 'backward' mode, the derivative at the first few points may not be accurate due to the one-sided window.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 4, 7, 11])
    >>> timeline = np.array([0, 1, 2, 3, 4])
    >>> window = 3
    >>> derivative(x, timeline, window, mode='bi')
    array([3., 3., 3.])

    >>> derivative(x, timeline, window, mode='forward')
    array([1., 2., 3.])

    >>> derivative(x, timeline, window, mode='backward')
    array([3., 3., 3., 3.])
    """

    # modes = bi, forward, backward
    dxdt = np.zeros(len(x))
    dxdt[:] = np.nan

    if mode == "bi":
        assert window % 2 == 1, "Please set an odd window for the bidirectional mode"
        lower_bound = window // 2
        upper_bound = len(x) - window // 2
    elif mode == "forward":
        lower_bound = 0
        upper_bound = len(x) - window
    elif mode == "backward":
        lower_bound = window
        upper_bound = len(x)

    for t in range(lower_bound, upper_bound):
        if mode == "bi":
            dxdt[t] = (x[t + window // 2] - x[t - window // 2]) / (
                timeline[t + window // 2] - timeline[t - window // 2]
            )
        elif mode == "forward":
            dxdt[t] = (x[t + window] - x[t]) / (timeline[t + window] - timeline[t])
        elif mode == "backward":
            dxdt[t] = (x[t] - x[t - window]) / (timeline[t] - timeline[t - window])
    return dxdt


def differentiate_per_track(tracks, measurement, window_size=3, mode="bi"):

    groupby_cols = ["TRACK_ID"]
    if "position" in list(tracks.columns):
        groupby_cols = ["position"] + groupby_cols

    tracks = tracks.sort_values(by=groupby_cols + ["FRAME"], ignore_index=True)
    tracks = tracks.reset_index(drop=True)
    for tid, group in tracks.groupby(groupby_cols):
        indices = group.index
        timeline = group["FRAME"].values
        signal = group[measurement].values
        dsignal = derivative(signal, timeline, window_size, mode=mode)
        tracks.loc[indices, "d/dt." + measurement] = dsignal
    return tracks


def velocity_per_track(tracks, window_size=3, mode="bi"):

    groupby_cols = ["TRACK_ID"]
    if "position" in list(tracks.columns):
        groupby_cols = ["position"] + groupby_cols

    tracks = tracks.sort_values(by=groupby_cols + ["FRAME"], ignore_index=True)
    tracks = tracks.reset_index(drop=True)
    for tid, group in tracks.groupby(groupby_cols):
        indices = group.index
        timeline = group["FRAME"].values
        x = group["POSITION_X"].values
        y = group["POSITION_Y"].values
        v = velocity(x, y, timeline, window=window_size, mode=mode)
        v_abs = magnitude_velocity(v)
        tracks.loc[indices, "velocity"] = v_abs
    return tracks


def velocity(x, y, timeline, window, mode="bi"):
    """
    Compute the velocity vector of a given 2D trajectory represented by arrays of x and y coordinates
    with respect to time using a specified numerical differentiation method.

    Parameters
    ----------
    x : array_like
            The array of x-coordinates of the trajectory.
    y : array_like
            The array of y-coordinates of the trajectory.
    timeline : array_like
            The array representing the time points corresponding to the x and y coordinates.
    window : int
            The size of the window used for numerical differentiation. Must be a positive odd integer.
    mode : {'bi', 'forward', 'backward'}, optional
            The numerical differentiation method to be used:
            - 'bi' (default): Bidirectional differentiation using a symmetric window.
            - 'forward': Forward differentiation using a one-sided window.
            - 'backward': Backward differentiation using a one-sided window.

    Returns
    -------
    v : ndarray
            The computed velocity vector of the 2D trajectory with respect to time.
            The first column represents the x-component of velocity, and the second column represents the y-component.

    Raises
    ------
    AssertionError
            If the window size is not an odd integer and mode is 'bi'.

    Notes
    -----
    - For 'bi' mode, the window size must be an odd number.
    - For 'forward' mode, the velocity at the edge points may not be accurate due to the one-sided window.
    - For 'backward' mode, the velocity at the first few points may not be accurate due to the one-sided window.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 4, 7, 11])
    >>> y = np.array([0, 3, 5, 8, 10])
    >>> timeline = np.array([0, 1, 2, 3, 4])
    >>> window = 3
    >>> velocity(x, y, timeline, window, mode='bi')
    array([[3., 3.],
               [3., 3.]])

    >>> velocity(x, y, timeline, window, mode='forward')
    array([[2., 2.],
               [3., 3.]])

    >>> velocity(x, y, timeline, window, mode='backward')
    array([[3., 3.],
               [3., 3.]])
    """

    v = np.zeros((len(x), 2))
    v[:, :] = np.nan

    v[:, 0] = derivative(x, timeline, window, mode=mode)
    v[:, 1] = derivative(y, timeline, window, mode=mode)

    return v


def magnitude_velocity(v_matrix):
    """
    Compute the magnitude of velocity vectors given a matrix representing 2D velocity vectors.

    Parameters
    ----------
    v_matrix : array_like
            The matrix where each row represents a 2D velocity vector with the first column
            being the x-component and the second column being the y-component.

    Returns
    -------
    magnitude : ndarray
            The computed magnitudes of the input velocity vectors.

    Notes
    -----
    - If a velocity vector has NaN components, the corresponding magnitude will be NaN.
    - The function handles NaN values in the input matrix gracefully.

    Examples
    --------
    >>> import numpy as np
    >>> v_matrix = np.array([[3, 4],
    ...                      [2, 2],
    ...                      [3, 3]])
    >>> magnitude_velocity(v_matrix)
    array([5., 2.82842712, 4.24264069])

    >>> v_matrix_with_nan = np.array([[3, 4],
    ...                               [np.nan, 2],
    ...                               [3, np.nan]])
    >>> magnitude_velocity(v_matrix_with_nan)
    array([5., nan, nan])
    """

    magnitude = np.zeros(len(v_matrix))
    magnitude[:] = np.nan
    for i in range(len(v_matrix)):
        if v_matrix[i, 0] == v_matrix[i, 0]:
            magnitude[i] = np.sqrt(v_matrix[i, 0] ** 2 + v_matrix[i, 1] ** 2)
    return magnitude


def orientation(v_matrix):
    """
    Compute the orientation angles (in radians) of 2D velocity vectors given a matrix representing velocity vectors.

    Parameters
    ----------
    v_matrix : array_like
            The matrix where each row represents a 2D velocity vector with the first column
            being the x-component and the second column being the y-component.

    Returns
    -------
    orientation_array : ndarray
            The computed orientation angles of the input velocity vectors in radians.
            If a velocity vector has NaN components, the corresponding orientation angle will be NaN.

    Examples
    --------
    >>> import numpy as np
    >>> v_matrix = np.array([[3, 4],
    ...                      [2, 2],
    ...                      [-3, -3]])
    >>> orientation(v_matrix)
    array([0.92729522, 0.78539816, -2.35619449])

    >>> v_matrix_with_nan = np.array([[3, 4],
    ...                               [np.nan, 2],
    ...                               [3, np.nan]])
    >>> orientation(v_matrix_with_nan)
    array([0.92729522, nan, nan])
    """

    orientation_array = np.zeros(len(v_matrix))
    for t in range(len(orientation_array)):
        if v_matrix[t, 0] == v_matrix[t, 0]:
            orientation_array[t] = np.arctan2(v_matrix[t, 0], v_matrix[t, 1])
    return orientation_array


def safe_log(array):
    """
    Safely computes the base-10 logarithm for numeric inputs, handling invalid or non-positive values.

    Parameters
    ----------
    array : int, float, list, or numpy.ndarray
            The input value or array for which to compute the logarithm.
            Can be a single number (int or float), a list, or a numpy array.

    Returns
    -------
    float or numpy.ndarray
            - If the input is a single numeric value, returns the base-10 logarithm as a float, or `np.nan` if the value is non-positive.
            - If the input is a list or numpy array, returns a numpy array with the base-10 logarithm of each element.
              Invalid or non-positive values are replaced with `np.nan`.

    Notes
    -----
    - Non-positive values (`<= 0`) are considered invalid and will result in `np.nan`.
    - NaN values in the input array are preserved in the output.
    - If the input is a list, it is converted to a numpy array for processing.

    Examples
    --------
    >>> safe_log(10)
    1.0

    >>> safe_log(-5)
    nan

    >>> safe_log([10, 0, -5, 100])
    array([1.0, nan, nan, 2.0])

    >>> import numpy as np
    >>> safe_log(np.array([1, 10, 100]))
    array([0.0, 1.0, 2.0])
    """

    array = np.asarray(array, dtype=float)
    result = np.where(array > 0, np.log10(array), np.nan)

    return result.item() if np.isscalar(array) else result
