from skimage.filters import difference_of_gaussians, threshold_otsu, threshold_local, threshold_niblack, threshold_sauvola
from celldetective.utils import interpolate_nan
import scipy.ndimage as snd
import numpy as np

def gauss_filter(img, sigma, interpolate=True, *kwargs):

	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))

	return snd.gaussian_filter(img.astype(float), sigma, *kwargs)

def median_filter(img, size, interpolate=True, *kwargs):

	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))

	size = int(size)
	
	return snd.median_filter(img, size, *kwargs)

def maximum_filter(img, size, interpolate=True, *kwargs):
	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))

	return snd.maximum_filter(img.astype(float), size, *kwargs)

def minimum_filter(img, size, interpolate=True, *kwargs):
	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))

	return snd.minimum_filter(img.astype(float), size, *kwargs)

def percentile_filter(img, percentile, size, interpolate=True, *kwargs):
	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))

	return snd.percentile_filter(img.astype(float), percentile, size, *kwargs)

def subtract_filter(img, value, *kwargs):
	return img.astype(float) - value

def abs_filter(img, *kwargs):
	return np.abs(img)

def ln_filter(img, interpolate=True, *kwargs):
	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))

	img[np.where(img>0.)] = np.log(img[np.where(img>0.)])
	img[np.where(img<=0.)] = 0.

	return img

def variance_filter(img, size, interpolate=True):

	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))

	size = int(size)
	img = img.astype(float)
	win_mean = snd.uniform_filter(img, (size,size), mode='wrap')
	win_sqr_mean = snd.uniform_filter(img**2, (size,size), mode='wrap')
	img = win_sqr_mean - win_mean**2

	return img

def std_filter(img, size, interpolate=True):

	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))
	
	size = int(size)
	img = img.astype(float)
	
	win_mean = snd.uniform_filter(img, (size,size), mode='wrap')
	win_sqr_mean = snd.uniform_filter(img**2, (size, size), mode='wrap')
	win_sqr_mean[win_sqr_mean<=0.] = 0. # add this to prevent sqrt from breaking
	
	sub = np.subtract(win_sqr_mean,win_mean**2)
	sub[sub<=0.] = 0.
	img = np.sqrt(sub)

	return img

def laplace_filter(img, output=float, interpolate=True, *kwargs):
	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))
	return snd.laplace(img.astype(float), *kwargs)

def dog_filter(img, blob_size=None, sigma_low=1, sigma_high=2, interpolate=True, *kwargs):
	
	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))
	if blob_size is not None:
		sigma_low = 1.0 / (1.0 + np.sqrt(2)) * blob_size
		sigma_high = np.sqrt(2)*sigma_low
	return difference_of_gaussians(img.astype(float), sigma_low, sigma_high, *kwargs)

def otsu_filter(img, *kwargs):
	thresh = threshold_otsu(img.astype(float))
	binary = img >= thresh
	return binary.astype(float)

def local_filter(img, *kwargs):
	thresh = threshold_local(img.astype(float), *kwargs)
	binary = img >= thresh
	return binary.astype(float)

def niblack_filter(img, *kwargs):

	thresh = threshold_niblack(img, *kwargs)
	binary = img >= thresh
	return binary.astype(float)

def sauvola_filter(img, *kwargs):

	thresh = threshold_sauvola(img, *kwargs)
	binary = img >= thresh
	return binary.astype(float)

def log_filter(img, blob_size=None, sigma=1, interpolate=True, *kwargs):

	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))
	if blob_size is not None:
		sigma_low = 1.0 / (1.0 + np.sqrt(2)) * blob_size
		sigma_high = np.sqrt(2)*sigma_low

	return snd.gaussian_laplace(img.astype(float), sigma, *kwargs)

def tophat_filter(img, size, connectivity=4, interpolate=True, *kwargs):
	
	if np.any(img!=img) and interpolate:
		img = interpolate_nan(img.astype(float))
	structure = snd.generate_binary_structure(rank=2, connectivity=connectivity)
	img = snd.white_tophat(img.astype(float), structure=structure, size=size, *kwargs)
	return img

def invert_filter(img, value=65535, *kwargs):
	
	img = img.astype(float)

	image_fill = np.zeros_like(img)
	image_fill[:,:] = value

	inverted = np.subtract(image_fill, img, where=img==img)
	return inverted

