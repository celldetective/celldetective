import numpy as np
from scipy.ndimage import zoom


def _rescale_labels(lbl: np.ndarray, scale_model: float = 1) -> np.ndarray:
    """
    Rescale label image.

    Parameters
    ----------
    lbl : ndarray
        Label image.
    scale_model : float, optional
        Scaling factor. Default is 1.

    Returns
    -------
    ndarray
        Rescaled label image.
    """
    return zoom(lbl, [1.0 / scale_model, 1.0 / scale_model], order=0)
