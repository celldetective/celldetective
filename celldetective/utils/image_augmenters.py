import random

import numpy as np
from scipy.ndimage import shift
from skimage.filters import gaussian
from skimage.util import random_noise

def augmenter(
    x,
    y,
    flip=True,
    gauss_blur=True,
    noise_option=True,
    shift=True,
    channel_extinction=True,
    extinction_probability=0.1,
    clip=False,
    max_sigma_blur=4,
    apply_noise_probability=0.5,
    augment_probability=0.9,
):
    """
    Applies a series of augmentation techniques to images and their corresponding masks for deep learning training.

    This function randomly applies a set of transformations including flipping, rotation, Gaussian blur,
    additive noise, shifting, and channel extinction to input images (x) and their masks (y) based on specified
    probabilities. These augmentations introduce variability in the training dataset, potentially improving model
    generalization.

    Parameters
    ----------
    x : ndarray
            The input image to be augmented, with dimensions (height, width, channels).
    y : ndarray
            The corresponding mask or label image for `x`, with the same spatial dimensions.
    flip : bool, optional
            Whether to randomly flip and rotate the images. Default is True.
    gauss_blur : bool, optional
            Whether to apply Gaussian blur to the images. Default is True.
    noise_option : bool, optional
            Whether to add random noise to the images. Default is True.
    shift : bool, optional
            Whether to randomly shift the images. Default is True.
    channel_extinction : bool, optional
            Whether to randomly set entire channels of the image to zero. Default is False.
    extinction_probability : float, optional
            The probability of an entire channel being set to zero. Default is 0.1.
    clip : bool, optional
            Whether to clip the noise-added images to stay within valid intensity values. Default is False.
    max_sigma_blur : int, optional
            The maximum sigma value for Gaussian blur. Default is 4.
    apply_noise_probability : float, optional
            The probability of applying noise to the image. Default is 0.5.
    augment_probability : float, optional
            The overall probability of applying any augmentation to the image. Default is 0.9.

    Returns
    -------
    tuple
            A tuple containing the augmented image and mask `(x, y)`.

    Raises
    ------
    AssertionError
            If `extinction_probability` is not within the range [0, 1].

    Notes
    -----
    - The augmentations are applied randomly based on the specified probabilities, allowing for
      a diverse set of transformed images from the original inputs.
    - This function is designed to be part of a preprocessing pipeline for training deep learning models,
      especially in tasks requiring spatial invariance and robustness to noise.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.rand(128, 128, 3)  # Sample image
    >>> y = np.random.randint(2, size=(128, 128))  # Sample binary mask
    >>> x_aug, y_aug = augmenter(x, y)
    # The returned `x_aug` and `y_aug` are augmented versions of `x` and `y`.

    """

    r = random.random()
    if r <= augment_probability:

        if flip:
            x, y = random_fliprot(x, y)

        if gauss_blur:
            x = blur(x, max_sigma=max_sigma_blur)

        if noise_option:
            x = noise(x, apply_probability=apply_noise_probability, clip_option=clip)

        if shift:
            x, y = random_shift(x, y)

        if channel_extinction:
            assert (
                extinction_probability <= 1.0
            ), "The extinction probability must be a number between 0 and 1."
            channel_off = [
                np.random.random() < extinction_probability for i in range(x.shape[-1])
            ]
            channel_off[0] = False
            x[:, :, np.array(channel_off, dtype=bool)] = 0.0

    return x, y


def noise(x, apply_probability=0.5, clip_option=False):
    """
    Applies random noise to each channel of a multichannel image based on a specified probability.

    This function introduces various types of random noise to an image. Each channel of the image can be
    modified independently with different noise models chosen randomly from a predefined list. The application
    of noise to any given channel is determined by a specified probability, allowing for selective noise
    addition.

    Parameters
    ----------
    x : ndarray
            The input multichannel image to which noise will be added. The image should be in format with channels
            as the last dimension (e.g., height x width x channels).
    apply_probability : float, optional
            The probability with which noise is applied to each channel of the image. Default is 0.5.
    clip_option : bool, optional
            Specifies whether to clip the corrupted data to stay within the valid range after noise addition.
            If True, the output array will be clipped to the range [0, 1] or [0, 255] depending on the input
            data type. Default is False.

    Returns
    -------
    ndarray
            The noised image. This output has the same shape as the input but potentially altered intensity values
            due to noise addition.

    Notes
    -----
    - The types of noise that can be applied include 'gaussian', 'localvar', 'poisson', and 'speckle'.
    - The choice of noise type for each channel is randomized and the noise is only applied if a randomly
      generated number is less than or equal to `apply_probability`.
    - Zero-valued pixels in the input image remain zero in the output to preserve background or masked areas.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.rand(256, 256, 3)  # Example 3-channel image
    >>> noised_image = noise(x)
    # The image 'x' may have different types of noise applied to each of its channels with a 50% probability.
    """

    x_noise = x.astype(float).copy()
    loc_i, loc_j, loc_c = np.where(x_noise == 0.0)
    options = ["gaussian", "localvar", "poisson", "speckle"]

    for k in range(x_noise.shape[-1]):
        mode_order = random.sample(options, len(options))
        for m in mode_order:
            p = np.random.random()
            if p <= apply_probability:
                try:
                    x_noise[:, :, k] = random_noise(
                        x_noise[:, :, k], mode=m, clip=clip_option
                    )
                except:
                    pass

    x_noise[loc_i, loc_j, loc_c] = 0.0

    return x_noise


def random_fliprot(img, mask):
    """
    Randomly flips and rotates an image and its corresponding mask.

    This function applies a series of random flips and permutations (rotations) to both the input image and its
    associated mask, ensuring that any transformations applied to the image are also exactly applied to the mask.
    The function is designed to handle multi-dimensional images (e.g., multi-channel images in YXC format where
    channels are last).

    Parameters
    ----------
    img : ndarray
            The input image to be transformed. This array is expected to have dimensions where the channel axis is last.
    mask : ndarray
            The mask corresponding to `img`, to be transformed in the same way as the image.

    Returns
    -------
    tuple of ndarray
            A tuple containing the transformed image and mask.

    Raises
    ------
    AssertionError
            If the number of dimensions of the mask exceeds that of the image, indicating incompatible shapes.

    """

    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_shift(image, mask, max_shift_amplitude=0.1):
    """
    Randomly shifts an image and its corresponding mask along the X and Y axes.

    This function shifts both the image and the mask by a randomly chosen distance up to a maximum
    percentage of the image's dimensions, specified by `max_shift_amplitude`. The shifts are applied
    independently in both the X and Y directions. This type of augmentation can help improve the robustness
    of models to positional variations in images.

    Parameters
    ----------
    image : ndarray
            The input image to be shifted. Must be in YXC format (height, width, channels).
    mask : ndarray
            The mask corresponding to `image`, to be shifted in the same way as the image.
    max_shift_amplitude : float, optional
            The maximum shift as a fraction of the image's dimension. Default is 0.1 (10% of the image's size).

    Returns
    -------
    tuple of ndarray
            A tuple containing the shifted image and mask.

    Notes
    -----
    - The shift values are chosen randomly within the range defined by the maximum amplitude.
    - Shifting is performed using the 'constant' mode where missing values are filled with zeros (cval=0.0),
      which may introduce areas of zero-padding along the edges of the shifted images and masks.
    - This function is designed to support data augmentation for machine learning and image processing tasks,
      particularly in contexts where spatial invariance is beneficial.

    """

    input_shape = image.shape[0]
    max_shift = input_shape * max_shift_amplitude

    shift_value_x = random.choice(np.arange(max_shift))
    if np.random.random() > 0.5:
        shift_value_x *= -1

    shift_value_y = random.choice(np.arange(max_shift))
    if np.random.random() > 0.5:
        shift_value_y *= -1

    image = shift(
        image,
        [shift_value_x, shift_value_y, 0],
        output=np.float32,
        order=3,
        mode="constant",
        cval=0.0,
    )
    mask = shift(
        mask, [shift_value_x, shift_value_y], order=0, mode="constant", cval=0.0
    )

    return image, mask


def blur(x, max_sigma=4.0):
    """
    Applies a random Gaussian blur to an image.

    This function blurs an image by applying a Gaussian filter with a randomly chosen sigma value. The sigma
    represents the standard deviation for the Gaussian kernel and is selected randomly up to a specified maximum.
    The blurring is applied while preserving the range of the image's intensity values and maintaining any
    zero-valued pixels as they are.

    Parameters
    ----------
    x : ndarray
            The input image to be blurred. The image can have any number of channels, but must be in a format
            where the channels are the last dimension (YXC format).
    max_sigma : float, optional
            The maximum value for the standard deviation of the Gaussian blur. Default is 4.0.

    Returns
    -------
    ndarray
            The blurred image. The output will have the same shape and type as the input image.

    Notes
    -----
    - The function ensures that zero-valued pixels in the input image remain unchanged after the blurring,
      which can be important for maintaining masks or other specific regions within the image.
    - Gaussian blurring is commonly used in image processing to reduce image noise and detail by smoothing.
    """

    sigma = np.random.random() * max_sigma
    loc_i, loc_j, loc_c = np.where(x == 0.0)
    x = gaussian(x, sigma, channel_axis=-1, preserve_range=True)
    x[loc_i, loc_j, loc_c] = 0.0

    return x
