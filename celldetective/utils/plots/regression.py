import numpy as np
from matplotlib import pyplot as plt


def regression_plot(y_pred, y_true, savepath=None):
    """

    Create a regression plot to compare predicted and ground truth values.

    Parameters
    ----------
    y_pred : array-like
            Predicted values.
    y_true : array-like
            Ground truth values.
    savepath : str or None, optional
            File path to save the plot. If None, the plot is displayed but not saved. Default is None.

    Returns
    -------
    None

    Notes
    -----
    This function creates a scatter plot comparing the predicted values (`y_pred`) to the ground truth values (`y_true`)
    for regression analysis. The plot also includes a diagonal reference line to visualize the ideal prediction scenario.

    If `savepath` is provided, the plot is saved as an image file at the specified path. The file format and other
    parameters can be controlled by the `savepath` argument.

    Examples
    --------
    >>> y_pred = [1.5, 2.0, 3.2, 4.1]
    >>> y_true = [1.7, 2.1, 3.5, 4.2]
    >>> regression_plot(y_pred, y_true)
    # Create a scatter plot comparing the predicted values to the ground truth values.

    >>> regression_plot(y_pred, y_true, savepath="regression_plot.png")
    # Create a scatter plot and save it as "regression_plot.png".

    """

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.scatter(y_pred, y_true)
    ax.set_xlabel("prediction")
    ax.set_ylabel("ground truth")
    line = np.linspace(np.amin([y_pred, y_true]), np.amax([y_pred, y_true]), 1000)
    ax.plot(line, line, linestyle="--", c="k", alpha=0.7)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=300)
    # plt.pause(2)
    plt.close()
