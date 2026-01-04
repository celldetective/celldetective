from scipy.ndimage import zoom


def _rescale_labels(lbl, scale_model=1):
    return zoom(lbl, [1.0 / scale_model, 1.0 / scale_model], order=0)
