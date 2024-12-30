
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from scipy.ndimage import shift, zoom
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.config import list_physical_devices
import configparser
from sklearn.utils.class_weight import compute_class_weight
from skimage.util import random_noise
from skimage.filters import gaussian
import random
from tifffile import imread
import json
from csbdeep.utils import normalize_mi_ma
from glob import glob
from urllib.request import urlopen
import zipfile
from tqdm import tqdm
import shutil
import tempfile
from scipy.interpolate import griddata
import re
from scipy.ndimage.morphology import distance_transform_edt
from scipy import ndimage
from skimage.morphology import disk
from scipy.stats import ks_2samp
from cliffs_delta import cliffs_delta
from stardist.models import StarDist2D
from cellpose.models import CellposeModel

def _remove_invalid_cols(df):

	"""
	Removes invalid columns from a DataFrame.

	This function identifies and removes columns in the DataFrame whose names 
	start with "Unnamed", which often indicate extraneous or improperly 
	formatted columns (e.g., leftover columns from improperly read CSV files).

	Parameters
	----------
	df : pandas.DataFrame
		The input DataFrame from which invalid columns will be removed.

	Returns
	-------
	pandas.DataFrame
		A new DataFrame with the invalid columns removed. If no invalid 
		columns are found, the original DataFrame is returned unchanged.

	Notes
	-----
	- This function does not modify the original DataFrame in place; instead, 
	  it returns a new DataFrame.
	- Columns starting with "Unnamed" are commonly introduced when saving 
	  or loading data files with misaligned headers.
	"""

	invalid_cols = [c for c in list(df.columns) if c.startswith('Unnamed')]
	if len(invalid_cols)>0:
		df = df.drop(invalid_cols, axis=1)
	return df

def _extract_coordinates_from_features(features, timepoint):

	"""
	Extracts spatial coordinates and other relevant metadata from a features DataFrame.

	This function processes a DataFrame of features to extract and rename centroid 
	coordinates, assign a unique identifier to each feature, and add a frame (timepoint) 
	column. The resulting DataFrame is structured for use in trajectory analysis.

	Parameters
	----------
	features : pandas.DataFrame
		A DataFrame containing feature data, including columns for centroids 
		(`'centroid-1'` and `'centroid-0'`) and feature classes (`'class_id'`).
	timepoint : int
		The timepoint (frame) to assign to all features. This is used to populate 
		the `'FRAME'` column in the output.

	Returns
	-------
	pandas.DataFrame
		A DataFrame containing the extracted coordinates and additional metadata, 
		with the following columns:
		- `'POSITION_X'`: X-coordinate of the centroid.
		- `'POSITION_Y'`: Y-coordinate of the centroid.
		- `'class_id'`: The label associated to the cell mask.
		- `'ID'`: A unique identifier for each cell (index-based).
		- `'FRAME'`: The timepoint associated with the features.

	Notes
	-----
	- The function assumes that the input DataFrame contains columns `'centroid-1'`, 
	  `'centroid-0'`, and `'class_id'`. Missing columns will raise a KeyError.
	- The `'ID'` column is created based on the index of the input DataFrame.
	- This function renames `'centroid-1'` to `'POSITION_X'` and `'centroid-0'` 
	  to `'POSITION_Y'`.
	"""

	coords = features[['centroid-1', 'centroid-0', 'class_id']].copy()
	coords['ID'] = np.arange(len(coords))
	coords.rename(columns={'centroid-1': 'POSITION_X', 'centroid-0': 'POSITION_Y'}, inplace=True)
	coords['FRAME'] = int(timepoint)

	return coords

def _mask_intensity_measurements(df, mask_channels):

	"""
	Removes columns from a DataFrame that match specific channel name patterns.

	This function filters out intensity measurement columns in a DataFrame based on 
	specified channel names. It identifies columns containing the channel 
	names as substrings and drops them from the DataFrame.

	Parameters
	----------
	df : pandas.DataFrame
		The input DataFrame containing intensity measurement data. Column names should 
		include the mask channel names if they are to be filtered.
	mask_channels : list of str or None
		A list of channel names (as substrings) to use for identifying columns 
		to remove. If `None`, no filtering is applied, and the original DataFrame is 
		returned.

	Returns
	-------
	pandas.DataFrame
		The modified DataFrame with specified columns removed. If no columns match 
		the mask channels, the original DataFrame is returned.

	Notes
	-----
	- The function searches for mask channel substrings in column names. 
	  Partial matches are sufficient to mark a column for removal.
	- If no mask channels are specified (`mask_channels` is `None`), the function 
	  does not modify the input DataFrame.
	"""

	if mask_channels is not None:
		
		cols_to_drop = []
		columns = df.columns

		for mc in mask_channels:
			cols_to_remove = [c for c in columns if mc in c]
			cols_to_drop.extend(cols_to_remove)

		if len(cols_to_drop)>0:
			df = df.drop(cols_to_drop, axis=1)
	return df

def _rearrange_multichannel_frame(frame):

	"""
	Rearranges the axes of a multi-channel frame to ensure the channel axis is at the end.

	This function standardizes the input frame to ensure that the channel axis (if present)
	is moved to the last position. For 2D frames, it adds a singleton channel axis at the end.

	Parameters
	----------
	frame : ndarray
		The input frame to be rearranged. Can be 2D or 3D.
		- If 3D, the function identifies the channel axis (assumed to be the axis with the smallest size)
		  and moves it to the last position.
		- If 2D, the function adds a singleton channel axis to make it compatible with 3D processing.

	Returns
	-------
	ndarray
		The rearranged frame with the channel axis at the end.
		- For 3D frames, the output shape will have the channel axis as the last dimension.
		- For 2D frames, the output will have shape `(H, W, 1)` where `H` and `W` are the height and width of the frame.

	Notes
	-----
	- This function assumes that in a 3D input, the channel axis is the one with the smallest size.
	- For 2D frames, this function ensures compatibility with multi-channel processing pipelines by
	  adding a singleton dimension for the channel axis.

	Examples
	--------
	Rearranging a 3D multi-channel frame:
	>>> frame = np.zeros((10, 10, 3))  # Already channel-last
	>>> _rearrange_multichannel_frame(frame).shape
	(10, 10, 3)

	Rearranging a 3D frame with channel axis not at the end:
	>>> frame = np.zeros((3, 10, 10))  # Channel-first
	>>> _rearrange_multichannel_frame(frame).shape
	(10, 10, 3)

	Converting a 2D frame to have a channel axis:
	>>> frame = np.zeros((10, 10))  # Grayscale image
	>>> _rearrange_multichannel_frame(frame).shape
	(10, 10, 1)
	"""


	if frame.ndim == 3:
		# Systematically move channel axis to the end
		channel_axis = np.argmin(frame.shape)
		frame = np.moveaxis(frame, channel_axis, -1)

	if frame.ndim==2:
		frame = frame[:,:,np.newaxis]

	return frame

def _fix_no_contrast(frames, value=1):

	"""
	Ensures that frames with no contrast (i.e., containing only a single unique value) are adjusted.

	This function modifies frames that lack contrast by adding a small value to the first pixel in
	the affected frame. This prevents downstream issues in image processing pipelines that require
	a minimum level of contrast.

	Parameters
	----------
	frames : ndarray
		A 3D array of shape `(H, W, N)`, where:
		- `H` is the height of the frame,
		- `W` is the width of the frame,
		- `N` is the number of frames or channels.
		Each frame (or channel) is independently checked for contrast.
	value : int or float, optional
		The value to add to the first pixel (`frames[0, 0, k]`) of any frame that lacks contrast.
		Default is `1`.

	Returns
	-------
	ndarray
		The modified `frames` array, where frames with no contrast have been adjusted.

	Notes
	-----
	- A frame is determined to have "no contrast" if all its pixel values are identical.
	- Only the first pixel (`[0, 0, k]`) of a no-contrast frame is modified, leaving the rest
	  of the frame unchanged.
	"""

	for k in range(frames.shape[2]):
		unique_values = np.unique(frames[:,:,k])
		if len(unique_values)==1:
			frames[0,0,k] += value
	return frames

def zoom_multiframes(frames, zoom_factor):

	"""
	Applies zooming to each frame (channel) in a multi-frame image.

	This function resizes each channel of a multi-frame image independently using a specified zoom factor. 
	The zoom is applied using spline interpolation of the specified order, and the channels are combined 
	back into the original format.

	Parameters
	----------
	frames : ndarray
		A multi-frame image with dimensions `(height, width, channels)`. The last axis represents different 
		channels.
	zoom_factor : float
		The zoom factor to apply to each channel. Values greater than 1 increase the size, and values 
		between 0 and 1 decrease the size.

	Returns
	-------
	ndarray
		A new multi-frame image with the same number of channels as the input, but with the height and width 
		scaled by the zoom factor.

	Notes
	-----
	- The function uses spline interpolation (order 3) for resizing, which provides smooth results.
	- `prefilter=False` is used to prevent additional filtering during the zoom operation.
	- The function assumes that the input is in `height x width x channels` format, with channels along the 
	  last axis.
	"""

	frames = [zoom(frames[:,:,c].copy(), [zoom_factor,zoom_factor], order=3, prefilter=False) for c in range(frames.shape[-1])]
	frames = np.moveaxis(frames,0,-1)
	return frames

def _prep_stardist_model(model_name, path, use_gpu=False, scale=1):

	"""
	Prepares and loads a StarDist2D model for segmentation tasks.

	This function initializes a StarDist2D model with the specified parameters, sets GPU usage if desired, 
	and allows scaling to adapt the model for specific applications.

	Parameters
	----------
	model_name : str
		The name of the StarDist2D model to load. This name should match the model saved in the specified path.
	path : str
		The directory where the model is stored.
	use_gpu : bool, optional
		If `True`, the model will be configured to use GPU acceleration for computations. Default is `False`.
	scale : int or float, optional
		A scaling factor for the model. This can be used to adapt the model for specific image resolutions. 
		Default is `1`.

	Returns
	-------
	tuple
		- model : StarDist2D
			The loaded StarDist2D model configured with the specified parameters.
		- scale_model : int or float
			The scaling factor passed to the function.

	Notes
	-----
	- Ensure the StarDist2D package is installed and the model files are correctly stored in the provided path.
	- GPU support depends on the availability of compatible hardware and software setup.
	"""

	model = StarDist2D(None, name=model_name, basedir=path)
	model.config.use_gpu = use_gpu
	model.use_gpu = use_gpu
	scale_model = scale
	print(f"StarDist model {model_name} successfully loaded...")
	return model, scale_model

def _prep_cellpose_model(model_name, path, use_gpu=False, n_channels=2, scale=None):

	"""
	Prepares and loads a Cellpose model for segmentation tasks.

	This function initializes a Cellpose model with the specified parameters, configures GPU usage if available, 
	and calculates or applies a scaling factor for the model based on image resolution.

	Parameters
	----------
	model_name : str
		The name of the pretrained Cellpose model to load.
	path : str
		The directory where the model is stored.
	use_gpu : bool, optional
		If `True`, the model will use GPU acceleration for computations. Default is `False`.
	n_channels : int, optional
		The number of input channels expected by the model. Default is `2`.
	scale : float, optional
		A scaling factor to adjust the model's output to match the image resolution. If not provided, the scale is 
		automatically calculated based on the model's diameter parameters.

	Returns
	-------
	tuple
		- model : CellposeModel
			The loaded Cellpose model configured with the specified parameters.
		- scale_model : float
			The scaling factor applied to the model, calculated or provided.

	Notes
	-----
	- Ensure the Cellpose package is installed and the model files are correctly stored in the provided path.
	- GPU support depends on the availability of compatible hardware and software setup.
	- The scale is calculated as `(diam_mean / diam_labels)` if `scale` is not provided, where `diam_mean` and 
	  `diam_labels` are attributes of the model.
	"""

	import torch
	if not use_gpu:
		device = torch.device("cpu")
	else:
		device = torch.device("cuda")

	model = CellposeModel(gpu=use_gpu, device=device, pretrained_model=path+model_name, model_type=None, nchan=n_channels) #diam_mean=30.0,
	if scale is None:
		scale_model = model.diam_mean / model.diam_labels
	else:
		scale_model = scale * model.diam_mean / model.diam_labels

	print(f"Diam mean: {model.diam_mean}; Diam labels: {model.diam_labels}; Final rescaling: {scale_model}...")
	print(f'Cellpose model {model_name} successfully loaded...')
	return model, scale_model


def _get_normalize_kwargs_from_config(config):

	if isinstance(config, str):
		if os.path.exists(config):
			with open(config) as cfg:
				config = json.load(cfg)
		else:
			print('Configuration could not be loaded...')
			os.abort()

	normalization_percentile = config['normalization_percentile']
	normalization_clip = config['normalization_clip']
	normalization_values = config['normalization_values']
	normalize_kwargs = _get_normalize_kwargs(normalization_percentile, normalization_values, normalization_clip)

	return normalize_kwargs

def _get_normalize_kwargs(normalization_percentile, normalization_values, normalization_clip):

	values = []
	percentiles = []
	for k in range(len(normalization_percentile)):
		if normalization_percentile[k]:
			percentiles.append(normalization_values[k])
			values.append(None)
		else:
			percentiles.append(None)
			values.append(normalization_values[k])

	return {"percentiles": percentiles, 'values': values, 'clip': normalization_clip}

def _segment_image_with_cellpose_model(img, model=None, diameter=None, cellprob_threshold=None, flow_threshold=None, channel_axis=-1):

	"""
	Segments an input image using a Cellpose model.

	This function applies a preloaded Cellpose model to segment an input image and returns the resulting labeled mask. 
	The image is rearranged into the format expected by the Cellpose model, with the specified channel axis moved to the first dimension.

	Parameters
	----------
	img : ndarray
		The input image to be segmented. It is expected to have a channel axis specified by `channel_axis`.
	model : CellposeModel, optional
		A preloaded Cellpose model instance used for segmentation.
	diameter : float, optional
		The diameter of objects to segment. If `None`, the model's default diameter is used.
	cellprob_threshold : float, optional
		The threshold for the probability of cells used during segmentation. If `None`, the default threshold is used.
	flow_threshold : float, optional
		The threshold for flow error during segmentation. If `None`, the default threshold is used.
	channel_axis : int, optional
		The axis of the input image that represents the channels. Default is `-1` (channel-last format).

	Returns
	-------
	ndarray
		A labeled mask of the same spatial dimensions as the input image, with segmented regions assigned unique 
		integer labels. The dtype of the mask is `uint16`.

	Notes
	-----
	- The `img` array is internally rearranged to move the specified `channel_axis` to the first dimension to comply 
	  with the Cellpose model's input requirements.
	- Ensure the provided `model` is a properly initialized Cellpose model instance.
	- Parameters `diameter`, `cellprob_threshold`, and `flow_threshold` allow fine-tuning of the segmentation process.
	"""

	img = np.moveaxis(img, channel_axis, 0)
	lbl, _, _ = model.eval(img, diameter = diameter, cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold, channels=None, normalize=False)

	return lbl.astype(np.uint16)

def _segment_image_with_stardist_model(img, model=None, return_details=False, channel_axis=-1):

	"""
	Segments an input image using a StarDist model.

	This function applies a preloaded StarDist model to segment an input image and returns the resulting labeled mask. 
	Optionally, additional details about the segmentation can also be returned.

	Parameters
	----------
	img : ndarray
		The input image to be segmented. It is expected to have a channel axis specified by `channel_axis`.
	model : StarDist2D, optional
		A preloaded StarDist model instance used for segmentation.
	return_details : bool, optional
		Whether to return additional details from the model alongside the labeled mask. Default is `False`.
	channel_axis : int, optional
		The axis of the input image that represents the channels. Default is `-1` (channel-last format).

	Returns
	-------
	ndarray
		A labeled mask of the same spatial dimensions as the input image, with segmented regions assigned unique 
		integer labels. The dtype of the mask is `uint16`.
	tuple of (ndarray, dict), optional
		If `return_details` is `True`, returns a tuple where the first element is the labeled mask and the second 
		element is a dictionary containing additional details about the segmentation.

	Notes
	-----
	- The `img` array is internally rearranged to move the specified `channel_axis` to the last dimension to comply 
	  with the StarDist model's input requirements.
	- Ensure the provided `model` is a properly initialized StarDist model instance.
	- The model automatically determines the number of tiles (`n_tiles`) required for processing large images.
	"""

	if channel_axis!=-1:
		img = np.moveaxis(img, channel_axis, -1)

	lbl, details = model.predict_instances(img, n_tiles=model._guess_n_tiles(img), show_tile_progress=False, verbose=False)
	if not return_details:
		return lbl.astype(np.uint16)
	else:
		return lbl.astype(np.uint16), details

def _rescale_labels(lbl, scale_model=1):
	return zoom(lbl, [1./scale_model, 1./scale_model], order=0)

def extract_cols_from_table_list(tables, nrows=1):

	"""
	Extracts a unique list of column names from a list of CSV tables.

	Parameters
	----------
	tables : list of str
		A list of file paths to the CSV tables from which to extract column names.
	nrows : int, optional
		The number of rows to read from each table to identify the columns.
		Default is 1.

	Returns
	-------
	numpy.ndarray
		An array of unique column names found across all the tables.

	Notes
	-----
	- This function reads only the first `nrows` rows of each table to improve performance when dealing with large files.
	- The function ensures that column names are unique by consolidating them using `numpy.unique`.

	Examples
	--------
	>>> tables = ["table1.csv", "table2.csv"]
	>>> extract_cols_from_table_list(tables)
	array(['Column1', 'Column2', 'Column3'], dtype='<U8')
	"""

	all_columns = []
	for tab in tables:
		cols = pd.read_csv(tab, nrows=1).columns.tolist()
		all_columns.extend(cols)
	all_columns = np.unique(all_columns)
	return all_columns

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

def contour_of_instance_segmentation(label, distance):

	"""

	Generate an instance mask containing the contour of the segmented objects.

	Parameters
	----------
	label : ndarray
		The instance segmentation labels.
	distance : int, float, list, or tuple
		The distance or range of distances from the edge of each instance to include in the contour.
		If a single value is provided, it represents the maximum distance. If a tuple or list is provided,
		it represents the minimum and maximum distances.

	Returns
	-------
	border_label : ndarray
		An instance mask containing the contour of the segmented objects.

	Notes
	-----
	This function generates an instance mask representing the contour of the segmented instances in the label image.
	It use the distance_transform_edt function from the scipy.ndimage module to compute the Euclidean distance transform.
	The contour is defined based on the specified distance(s) from the edge of each instance.
	The resulting mask, `border_label`, contains the contour regions, while the interior regions are set to zero.

	Examples
	--------
	>>> border_label = contour_of_instance_segmentation(label, distance=3)
	# Generate a binary mask containing the contour of the segmented instances with a maximum distance of 3 pixels.

	"""
	if isinstance(distance,(list,tuple)) or distance >= 0 :

		edt = distance_transform_edt(label)

		if isinstance(distance, list) or isinstance(distance, tuple):
			min_distance = distance[0]; max_distance = distance[1]

		elif isinstance(distance, (int, float)):
			min_distance = 0
			max_distance = distance

		thresholded = (edt <= max_distance) * (edt > min_distance)
		border_label = np.copy(label)
		border_label[np.where(thresholded == 0)] = 0

	else:
		size = (2*abs(int(distance))+1, 2*abs(int(distance))+1)
		dilated_image = ndimage.grey_dilation(label, footprint=disk(int(abs(distance)))) #size=size,
		border_label=np.copy(dilated_image)
		matching_cells = np.logical_and(dilated_image != 0, label == dilated_image)
		border_label[np.where(matching_cells == True)] = 0
		border_label[label!=0] = 0.

	return border_label

def extract_identity_col(trajectories):

	"""
	Determines the identity column name in a DataFrame of trajectories.

	This function checks the provided DataFrame for the presence of a column
	that can serve as the identity column. It first looks for the column
	'TRACK_ID'. If 'TRACK_ID' exists but contains only null values, it checks
	for the column 'ID' instead. If neither column is found, the function
	returns `None` and prints a message indicating the issue.

	Parameters
	----------
	trajectories : pandas.DataFrame
		A DataFrame containing trajectory data. The function assumes that
		the identity of each trajectory might be stored in either the
		'TRACK_ID' or 'ID' column.

	Returns
	-------
	str or None
		The name of the identity column ('TRACK_ID' or 'ID') if found;
		otherwise, `None`.
	"""

	for col in ['TRACK_ID', 'ID']:
		if col in trajectories.columns and not trajectories[col].isnull().all():
			return col

	print('ID or TRACK_ID column could not be found in the table...')
	return None

def derivative(x, timeline, window, mode='bi'):
	
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
	
	if mode=='bi':
		assert window%2==1,'Please set an odd window for the bidirectional mode'
		lower_bound = window//2
		upper_bound = len(x) - window//2
	elif mode=='forward':
		lower_bound = 0
		upper_bound = len(x) - window
	elif mode=='backward':
		lower_bound = window
		upper_bound = len(x)

	for t in range(lower_bound,upper_bound):
		if mode=='bi':
			dxdt[t] = (x[t+window//2] - x[t-window//2]) / (timeline[t+window//2] - timeline[t-window//2])
		elif mode=='forward':
			dxdt[t] = (x[t+window] - x[t]) /  (timeline[t+window] - timeline[t])
		elif mode=='backward':
			dxdt[t] = (x[t] - x[t-window]) /  (timeline[t] - timeline[t-window])
	return dxdt

def differentiate_per_track(tracks, measurement, window_size=3, mode='bi'):
	
	groupby_cols = ['TRACK_ID']
	if 'position' in list(tracks.columns):
		groupby_cols = ['position']+groupby_cols

	tracks = tracks.sort_values(by=groupby_cols+['FRAME'],ignore_index=True)
	tracks = tracks.reset_index(drop=True)
	for tid, group in tracks.groupby(groupby_cols):
		indices = group.index
		timeline = group['FRAME'].values
		signal = group[measurement].values
		dsignal = derivative(signal, timeline, window_size, mode=mode)
		tracks.loc[indices, 'd/dt.'+measurement] = dsignal
	return tracks

def velocity_per_track(tracks, window_size=3, mode='bi'):
	
	groupby_cols = ['TRACK_ID']
	if 'position' in list(tracks.columns):
		groupby_cols = ['position']+groupby_cols

	tracks = tracks.sort_values(by=groupby_cols+['FRAME'],ignore_index=True)
	tracks = tracks.reset_index(drop=True)
	for tid, group in tracks.groupby(groupby_cols):
		indices = group.index
		timeline = group['FRAME'].values
		x = group['POSITION_X'].values
		y = group['POSITION_Y'].values
		v = velocity(x,y,timeline,window=window_size,mode=mode)
		v_abs = magnitude_velocity(v)
		tracks.loc[indices, 'velocity'] = v_abs
	return tracks

def velocity(x,y,timeline,window,mode='bi'):

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

	v = np.zeros((len(x),2))
	v[:,:] = np.nan
	
	v[:,0] = derivative(x, timeline, window, mode=mode)
	v[:,1] = derivative(y, timeline, window, mode=mode)

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
		if v_matrix[i,0]==v_matrix[i,0]:
			magnitude[i] = np.sqrt(v_matrix[i,0]**2 + v_matrix[i,1]**2)
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
		if v_matrix[t,0]==v_matrix[t,0]:
			orientation_array[t] = np.arctan2(v_matrix[t,0],v_matrix[t,1])
	return orientation_array


def estimate_unreliable_edge(activation_protocol=[['gauss',2],['std',4]]):

	"""
	Safely estimate the distance to the edge of an image in which the filtered image values can be artefactual.

	Parameters
	----------
	activation_protocol : list of list, optional
		A list of lists, where each sublist contains a string naming the filter function, followed by its arguments (usually a kernel size). 
		Default is [['gauss', 2], ['std', 4]].

	Returns
	-------
	int or None
		The sum of the kernel sizes in the activation protocol if the protocol 
		is not empty. Returns None if the activation protocol is empty.

	Notes
	-----
	This function assumes that the second element of each sublist in the 
	activation protocol is a kernel size.

	Examples
	--------
	>>> estimate_unreliable_edge([['gauss', 2], ['std', 4]])
	6
	>>> estimate_unreliable_edge([])
	None
	"""

	if activation_protocol==[]:
		return None
	else:
		edge=0
		for fct in activation_protocol:
			if isinstance(fct[1],(int,np.int_)) and not fct[0]=='invert':
				edge+=fct[1]
		return edge

def unpad(img, pad):
	
	"""
	Remove padding from an image.

	This function removes the specified amount of padding from the borders
	of an image. The padding is assumed to be the same on all sides.

	Parameters
	----------
	img : ndarray
		The input image from which the padding will be removed.
	pad : int
		The amount of padding to remove from each side of the image.

	Returns
	-------
	ndarray
		The image with the padding removed.

	Raises
	------
	ValueError
		If `pad` is greater than or equal to half of the smallest dimension 
		of `img`.

	See Also
	--------
	numpy.pad : Pads an array.

	Notes
	-----
	This function assumes that the input image is a 2D array.

	Examples
	--------
	>>> import numpy as np
	>>> img = np.array([[0, 0, 0, 0, 0],
	...                 [0, 1, 1, 1, 0],
	...                 [0, 1, 1, 1, 0],
	...                 [0, 1, 1, 1, 0],
	...                 [0, 0, 0, 0, 0]])
	>>> unpad(img, 1)
	array([[1, 1, 1],
		   [1, 1, 1],
		   [1, 1, 1]])
	"""

	return img[pad:-pad, pad:-pad]

def mask_edges(binary_mask, border_size):
	
	"""
	Mask the edges of a binary mask.

	This function sets the edges of a binary mask to False, effectively 
	masking out a border of the specified size.

	Parameters
	----------
	binary_mask : ndarray
		A 2D binary mask array where the edges will be masked.
	border_size : int
		The size of the border to mask (set to False) on all sides.

	Returns
	-------
	ndarray
		The binary mask with the edges masked out.

	Raises
	------
	ValueError
		If `border_size` is greater than or equal to half of the smallest 
		dimension of `binary_mask`.

	Notes
	-----
	This function assumes that the input `binary_mask` is a 2D array. The 
	input mask is converted to a boolean array before masking the edges.

	Examples
	--------
	>>> import numpy as np
	>>> binary_mask = np.array([[1, 1, 1, 1, 1],
	...                         [1, 1, 1, 1, 1],
	...                         [1, 1, 1, 1, 1],
	...                         [1, 1, 1, 1, 1],
	...                         [1, 1, 1, 1, 1]])
	>>> mask_edges(binary_mask, 1)
	array([[False, False, False, False, False],
		   [False,  True,  True,  True, False],
		   [False,  True,  True,  True, False],
		   [False,  True,  True,  True, False],
		   [False, False, False, False, False]])
	"""

	binary_mask = binary_mask.astype(bool)
	binary_mask[:border_size,:] = False
	binary_mask[(binary_mask.shape[0]-border_size):,:] = False
	binary_mask[:,:border_size] = False
	binary_mask[:,(binary_mask.shape[1]-border_size):] = False

	return binary_mask

def demangle_column_name(name):
	if name.startswith("BACKTICK_QUOTED_STRING_"):
		# Unquote backtick-quoted string.
		return name[len("BACKTICK_QUOTED_STRING_"):].replace("_DOT_", ".").replace("_SLASH_", "/")
	return name

def extract_cols_from_query(query: str):
	
	# Track variables in a dictionary to be used as a dictionary of globals. From: https://stackoverflow.com/questions/64576913/extract-pandas-dataframe-column-names-from-query-string
	
	variables = {}

	while True:
		try:
			# Try creating a Expr object with the query string and dictionary of globals.
			# This will raise an error as long as the dictionary of globals is incomplete.
			env = pd.core.computation.scope.ensure_scope(level=0, global_dict=variables)
			pd.core.computation.eval.Expr(query, env=env)

			# Exit the loop when evaluation is successful.
			break
		except pd.errors.UndefinedVariableError as e:
			# This relies on the format defined here: https://github.com/pandas-dev/pandas/blob/965ceca9fd796940050d6fc817707bba1c4f9bff/pandas/errors/__init__.py#L401
			name = re.findall("name '(.+?)' is not defined", str(e))[0]

			# Add the name to the globals dictionary with a dummy value.
			variables[name] = None

	return [demangle_column_name(name) for name in variables.keys()]

def create_patch_mask(h, w, center=None, radius=None):

	"""

	Create a circular patch mask of given dimensions.
	Adapted from alkasm on https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

	Parameters
	----------
	h : int
		Height of the mask. Prefer odd value.
	w : int
		Width of the mask. Prefer odd value.
	center : tuple, optional
		Coordinates of the center of the patch. If not provided, the middle of the image is used.
	radius : int or float or list, optional
		Radius of the circular patch. If not provided, the smallest distance between the center and image walls is used.
		If a list is provided, it should contain two elements representing the inner and outer radii of a circular annular patch.

	Returns
	-------
	numpy.ndarray
		Boolean mask where True values represent pixels within the circular patch or annular patch, and False values represent pixels outside.

	Notes
	-----
	The function creates a circular patch mask of the given dimensions by determining which pixels fall within the circular patch or annular patch.
	The circular patch or annular patch is centered at the specified coordinates or at the middle of the image if coordinates are not provided.
	The radius of the circular patch or annular patch is determined by the provided radius parameter or by the minimum distance between the center and image walls.
	If an annular patch is desired, the radius parameter should be a list containing the inner and outer radii respectively.

	Examples
	--------
	>>> mask = create_patch_mask(100, 100, center=(50, 50), radius=30)
	>>> print(mask)

	"""

	if center is None: # use the middle of the image
		center = (int(w/2), int(h/2))
	if radius is None: # use the smallest distance between the center and image walls
		radius = min(center[0], center[1], w-center[0], h-center[1])

	Y, X = np.ogrid[:h, :w]
	dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

	if isinstance(radius,int) or isinstance(radius,float):
		mask = dist_from_center <= radius
	elif isinstance(radius,list):
		mask = (dist_from_center <= radius[1])*(dist_from_center >= radius[0])
	else:
		print("Please provide a proper format for the radius")
		return None
		
	return mask

def rename_intensity_column(df, channels):

	"""

	Rename intensity columns in a DataFrame based on the provided channel names.

	Parameters
	----------
	df : pandas DataFrame
		The DataFrame containing the intensity columns.
	channels : list
		A list of channel names corresponding to the intensity columns.

	Returns
	-------
	pandas DataFrame
		The DataFrame with renamed intensity columns.

	Notes
	-----
	This function renames the intensity columns in a DataFrame based on the provided channel names.
	It searches for columns containing the substring 'intensity' in their names and replaces it with
	the respective channel name. The renaming is performed according to the order of the channels
	provided in the `channels` list. If multiple channels are provided, the function assumes that the
	intensity columns have a naming pattern that includes a numerical index indicating the channel.
	If only one channel is provided, the function replaces 'intensity' with the single channel name.

	Examples
	--------
	>>> data = {'intensity_0': [1, 2, 3], 'intensity_1': [4, 5, 6]}
	>>> df = pd.DataFrame(data)
	>>> channels = ['channel1', 'channel2']
	>>> renamed_df = rename_intensity_column(df, channels)
	# Rename the intensity columns in the DataFrame based on the provided channel names.

	"""

	channel_names = np.array(channels)
	channel_indices = np.arange(len(channel_names),dtype=int)

	if np.any(['intensity' in c for c in list(df.columns)]):

		intensity_indices = [s.startswith('intensity') for s in df.columns]
		intensity_columns = df.columns[intensity_indices]

		if len(channel_names) >= 1:
			to_rename = {}
			for k in range(len(intensity_columns)):

				sections = np.array(re.split('-|_', intensity_columns[k]))
				test_digit = np.array([s.isdigit() for s in sections])
				index = int(sections[np.where(test_digit)[0]][-1])

				channel_name = channel_names[np.where(channel_indices==index)[0]][0]
				new_name = np.delete(sections, np.where(test_digit)[0]) #np.where(test_digit)[0]
				new_name = '_'.join(list(new_name))
				new_name = new_name.replace('intensity', channel_name)
				new_name = new_name.replace('-','_')
				new_name = new_name.replace('_nanmean','_mean')

				to_rename.update({intensity_columns[k]: new_name})

				if 'centre' in intensity_columns[k]:
					# sections = np.array(re.split('-|_', intensity_columns[k]))
					measure = np.array(re.split('-|_', new_name))
					if sections[-2] == "0":
						new_name = np.delete(measure, -1)
						new_name = '_'.join(list(new_name))
						if 'edge' in intensity_columns[k]:
							new_name = new_name.replace('centre_of_mass_displacement', "edge_centre_of_mass_displacement_in_px")
						else:
							new_name = new_name.replace('centre_of_mass', "centre_of_mass_displacement_in_px")
						to_rename.update({intensity_columns[k]: new_name.replace('-', '_')})
					elif sections[-2] == "1":
						new_name = np.delete(measure, -1)
						new_name = '_'.join(list(new_name))
						if 'edge' in intensity_columns[k]:
							new_name = new_name.replace('centre_of_mass_displacement', "edge_centre_of_mass_orientation")
						else:
							new_name = new_name.replace('centre_of_mass', "centre_of_mass_orientation")
						to_rename.update({intensity_columns[k]: new_name.replace('-', '_')})
					elif sections[-2] == "2":
						new_name = np.delete(measure, -1)
						new_name = '_'.join(list(new_name))
						if 'edge' in intensity_columns[k]:
							new_name = new_name.replace('centre_of_mass_displacement', "edge_centre_of_mass_x")
						else:
							new_name = new_name.replace('centre_of_mass', "centre_of_mass_x")
						to_rename.update({intensity_columns[k]: new_name.replace('-', '_')})
					elif sections[-2] == "3":
						new_name = np.delete(measure, -1)
						new_name = '_'.join(list(new_name))
						if 'edge' in intensity_columns[k]:
							new_name = new_name.replace('centre_of_mass_displacement', "edge_centre_of_mass_y")
						else:
							new_name = new_name.replace('centre_of_mass', "centre_of_mass_y")
						to_rename.update({intensity_columns[k]: new_name.replace('-', '_')})
				if 'radial_gradient' in intensity_columns[k]:
					# sections = np.array(re.split('-|_', intensity_columns[k]))
					measure = np.array(re.split('-|_', new_name))
					if sections[-2] == "0":
						new_name = np.delete(measure, -1)
						new_name = '_'.join(list(measure))
						new_name = new_name.replace('radial_gradient', "radial_gradient")
						to_rename.update({intensity_columns[k]: new_name.replace('-', '_')})
					elif sections[-2] == "1":
						new_name = np.delete(measure, -1)
						new_name = '_'.join(list(measure))
						new_name = new_name.replace('radial_gradient', "radial_intercept")
						to_rename.update({intensity_columns[k]: new_name.replace('-', '_')})
		else:
			to_rename = {}
			for k in range(len(intensity_columns)):
				sections = np.array(re.split('_|-', intensity_columns[k]))
				channel_name = channel_names[0]
				test_digit = np.array([s.isdigit() for s in sections])
				new_name = np.delete(sections, np.where(test_digit)[0])
				new_name = '_'.join(list(new_name))
				new_name = new_name.replace('intensity', channel_name)
				to_rename.update({intensity_columns[k]: new_name.replace('-','_')})
				if 'centre' in intensity_columns[k]:
					measure = np.array(re.split('-|_', new_name))
					if sections[-2] == "0":
						new_name = np.delete(measure, -1)
						new_name = '_'.join(list(new_name))
						if 'edge' in intensity_columns[k]:
							new_name = new_name.replace('centre_of_mass_displacement', "edge_centre_of_mass_displacement_in_px")
						else:
							new_name = new_name.replace('centre_of_mass', "centre_of_mass_displacement_in_px")
						to_rename.update({intensity_columns[k]: new_name.replace('-', '_')})
					if sections[-2] == "1":
						new_name = np.delete(measure, -1)
						new_name = '_'.join(list(new_name))
						if 'edge' in intensity_columns[k]:
							new_name = new_name.replace('centre_of_mass_displacement', "edge_centre_of_mass_orientation")
						else:
							new_name = new_name.replace('centre_of_mass', "centre_of_mass_orientation")
						to_rename.update({intensity_columns[k]: new_name.replace('-', '_')})
				if 'radial_gradient' in intensity_columns[k]:
					# sections = np.array(re.split('-|_', intensity_columns[k]))
					measure = np.array(re.split('-|_', new_name))
					if sections[-2] == "0":
						#new_name = np.delete(measure, -1)
						new_name = '_'.join(list(measure))
						new_name = new_name.replace('radial_gradient', "radial_gradient")
						to_rename.update({intensity_columns[k]: new_name.replace('-', '_')})
					elif sections[-2] == "1":
						#new_name = np.delete(measure, -1)
						new_name = '_'.join(list(measure))
						new_name = new_name.replace('radial_gradient', "radial_intercept")
						to_rename.update({intensity_columns[k]: new_name.replace('-', '_')})

		df = df.rename(columns=to_rename)

	return df


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

	fig,ax = plt.subplots(1,1,figsize=(4,3))
	ax.scatter(y_pred, y_true)
	ax.set_xlabel("prediction")
	ax.set_ylabel("ground truth")
	line = np.linspace(np.amin([y_pred,y_true]),np.amax([y_pred,y_true]),1000)
	ax.plot(line,line,linestyle="--",c="k",alpha=0.7)
	plt.tight_layout()
	if savepath is not None:
		plt.savefig(savepath,bbox_inches="tight",dpi=300)
	plt.pause(2)
	plt.close()

def split_by_ratio(arr, *ratios):

	"""

	Split an array into multiple chunks based on given ratios.

	Parameters
	----------
	arr : array-like
		The input array to be split.
	*ratios : float
		Ratios specifying the proportions of each chunk. The sum of ratios should be less than or equal to 1.

	Returns
	-------
	list
		A list of arrays containing the splits/chunks of the input array.

	Notes
	-----
	This function randomly permutes the input array (`arr`) and then splits it into multiple chunks based on the provided ratios.
	The ratios determine the relative sizes of the resulting chunks. The sum of the ratios should be less than or equal to 1.
	The function uses the accumulated ratios to determine the split indices.

	The function returns a list of arrays representing the splits of the input array. The number of splits is equal to the number
	of provided ratios. If there are more ratios than splits, the extra ratios are ignored.

	Examples
	--------
	>>> arr = np.arange(10)
	>>> splits = split_by_ratio(arr, 0.6, 0.2, 0.2)
	>>> print(len(splits))
	3
	# Split the array into 3 chunks with ratios 0.6, 0.2, and 0.2.

	>>> arr = np.arange(100)
	>>> splits = split_by_ratio(arr, 0.5, 0.25)
	>>> print([len(split) for split in splits])
	[50, 25]
	# Split the array into 2 chunks with ratios 0.5 and 0.25.
	
	"""

	arr = np.random.permutation(arr)
	ind = np.add.accumulate(np.array(ratios) * len(arr)).astype(int)
	return [x.tolist() for x in np.split(arr, ind)][:len(ratios)]

def compute_weights(y):

	"""

	Compute class weights based on the input labels.

	Parameters
	----------
	y : array-like
		Array of labels.

	Returns
	-------
	dict
		A dictionary containing the computed class weights.

	Notes
	-----
	This function calculates the class weights based on the input labels (`y`) using the "balanced" method.
	The class weights are computed to address the class imbalance problem, where the weights are inversely
	proportional to the class frequencies.

	The function returns a dictionary (`class_weights`) where the keys represent the unique classes in `y`
	and the values represent the computed weights for each class.

	Examples
	--------
	>>> labels = np.array([0, 1, 0, 1, 1])
	>>> weights = compute_weights(labels)
	>>> print(weights)
	{0: 1.5, 1: 0.75}
	# Compute class weights for the binary labels.

	>>> labels = np.array([0, 1, 2, 0, 1, 2, 2])
	>>> weights = compute_weights(labels)
	>>> print(weights)
	{0: 1.1666666666666667, 1: 1.1666666666666667, 2: 0.5833333333333334}
	# Compute class weights for the multi-class labels.
	
	"""

	class_weights = compute_class_weight(
											class_weight = "balanced",
											classes = np.unique(y),
											y = y,                                                  
								)
	class_weights = dict(zip(np.unique(y), class_weights))

	return class_weights

def train_test_split(data_x, data_y1, data_class=None, validation_size=0.25, test_size=0, n_iterations=10):

	"""

	Split the dataset into training, validation, and test sets.

	Parameters
	----------
	data_x : array-like
		Input features or independent variables.
	data_y1 : array-like
		Target variable 1.
	data_y2 : array-like
		Target variable 2.
	validation_size : float, optional
		Proportion of the dataset to include in the validation set. Default is 0.25.
	test_size : float, optional
		Proportion of the dataset to include in the test set. Default is 0.

	Returns
	-------
	dict
		A dictionary containing the split datasets.
		Keys: "x_train", "x_val", "y1_train", "y1_val", "y2_train", "y2_val".
		If test_size > 0, additional keys: "x_test", "y1_test", "y2_test".

	Notes
	-----
	This function divides the dataset into training, validation, and test sets based on the specified proportions.
	It shuffles the data and splits it according to the proportions defined by `validation_size` and `test_size`.

	The input features (`data_x`) and target variables (`data_y1`, `data_y2`) should be arrays or array-like objects
	with compatible dimensions.

	The function returns a dictionary containing the split datasets. The training set is assigned to "x_train",
	"y1_train", and "y2_train". The validation set is assigned to "x_val", "y1_val", and "y2_val". If `test_size` is
	greater than 0, the test set is assigned to "x_test", "y1_test", and "y2_test".

	"""

	if data_class is not None:
		print(f"Unique classes: {np.sort(np.argmax(np.unique(data_class,axis=0),axis=1))}")

	for i in range(n_iterations):

		n_values = len(data_x)
		randomize = np.arange(n_values)
		np.random.shuffle(randomize)

		train_percentage = 1 - validation_size - test_size

		chunks = split_by_ratio(randomize, train_percentage, validation_size, test_size)

		x_train = data_x[chunks[0]]
		y1_train = data_y1[chunks[0]]
		if data_class is not None:
			y2_train = data_class[chunks[0]]

		x_val = data_x[chunks[1]]
		y1_val = data_y1[chunks[1]]
		if data_class is not None:
			y2_val = data_class[chunks[1]]

		if data_class is not None:
			print(f"classes in train set: {np.sort(np.argmax(np.unique(y2_train,axis=0),axis=1))}; classes in validation set: {np.sort(np.argmax(np.unique(y2_val,axis=0),axis=1))}")
			same_class_test = np.array_equal(np.sort(np.argmax(np.unique(y2_train,axis=0),axis=1)), np.sort(np.argmax(np.unique(y2_val,axis=0),axis=1)))
			print(f"Check that classes are found in all sets: {same_class_test}...")
		else:
			same_class_test = True

		if same_class_test:

			ds = {"x_train": x_train, "x_val": x_val,
				 "y1_train": y1_train, "y1_val": y1_val}
			if data_class is not None:
				ds.update({"y2_train": y2_train, "y2_val": y2_val})

			if test_size>0:
				x_test = data_x[chunks[2]]
				y1_test = data_y1[chunks[2]]
				ds.update({"x_test": x_test, "y1_test": y1_test})
				if data_class is not None:
					y2_test = data_class[chunks[2]]
					ds.update({"y2_test": y2_test})
			return ds
		else:
			continue

	raise Exception("Some classes are missing from the train or validation set... Abort.")


def remove_redundant_features(features, reference_features, channel_names=None):

	"""
	
	Remove redundant features from a list of features based on a reference feature list.

	Parameters
	----------
	features : list
		The list of features to be filtered.
	reference_features : list
		The reference list of features.
	channel_names : list or None, optional
		The list of channel names. If provided, it is used to identify and remove redundant intensity features. 
		Default is None.

	Returns
	-------
	list
		The filtered list of features without redundant entries.

	Notes
	-----
	This function removes redundant features from the input list based on a reference list of features. Features that 
	appear in the reference list are removed from the input list. Additionally, if the channel_names parameter is provided,
	it is used to identify and remove redundant intensity features. Intensity features that have the same mode (e.g., 'mean',
	'min', 'max') as any of the channel names in the reference list are also removed.

	Examples
	--------
	>>> features = ['area', 'intensity_mean', 'intensity_max', 'eccentricity']
	>>> reference_features = ['area', 'eccentricity']
	>>> filtered_features = remove_redundant_features(features, reference_features)
	>>> filtered_features
	['intensity_mean', 'intensity_max']

	>>> channel_names = ['brightfield', 'channel1', 'channel2']
	>>> filtered_features = remove_redundant_features(features, reference_features, channel_names)
	>>> filtered_features
	['area', 'eccentricity']

	"""

	new_features = features[:]

	for f in features:

		if f in reference_features:
			new_features.remove(f)

		if ('intensity' in f) and (channel_names is not None):

			mode = f.split('_')[-1]
			pattern = [a+'_'+mode for a in channel_names]

			for p in pattern:
				if p in reference_features:
					try:
						new_features.remove(f)
					except:
						pass
	return new_features

def _estimate_scale_factor(spatial_calibration, required_spatial_calibration):

	"""
	Estimates the scale factor needed to adjust spatial calibration to a required value.

	This function calculates the scale factor by which spatial dimensions (e.g., in microscopy images)
	should be adjusted to align with a specified calibration standard. This is particularly useful when
	preparing data for analysis with models trained on data of a specific spatial calibration.

	Parameters
	----------
	spatial_calibration : float or None
		The current spatial calibration factor of the data, expressed as units per pixel (e.g., micrometers per pixel).
		If None, indicates that the current spatial calibration is unknown or unspecified.
	required_spatial_calibration : float or None
		The spatial calibration factor required for compatibility with the model or analysis standard, expressed
		in the same units as `spatial_calibration`. If None, indicates no adjustment is required.

	Returns
	-------
	float or None
		The scale factor by which the current data should be rescaled to match the required spatial calibration,
		or None if no scaling is necessary or if insufficient information is provided.

	Notes
	-----
	- A scale factor close to 1 (within a tolerance defined by `epsilon`) indicates that no significant rescaling
	  is needed, and the function returns None.
	- The function issues a warning if a significant rescaling is necessary, indicating the scale factor to be applied.

	Examples
	--------
	>>> scale_factor = _estimate_scale_factor(spatial_calibration=0.5, required_spatial_calibration=0.25)
	# Each frame will be rescaled by a factor 2.0 to match with the model training data...

	>>> scale_factor = _estimate_scale_factor(spatial_calibration=None, required_spatial_calibration=0.25)
	# Returns None due to insufficient information about current spatial calibration.
	"""

	if (required_spatial_calibration is not None)*(spatial_calibration is not None):
		scale = spatial_calibration / required_spatial_calibration
	else:
		scale = None

	epsilon = 0.05
	if scale is not None:
		if not np.all([scale >= (1-epsilon), scale <= (1+epsilon)]):
			print(f"Each frame will be rescaled by a factor {scale} to match with the model training data...")
		else:
			scale = None
	return scale

def auto_find_gpu():

	"""
	Automatically detects the presence of GPU devices in the system.

	This function checks if any GPU devices are available for use by querying the system's physical devices.
	It is a utility function to simplify the process of determining whether GPU-accelerated computing can be
	leveraged in data processing or model training tasks.

	Returns
	-------
	bool
		True if one or more GPU devices are detected, False otherwise.

	Notes
	-----
	- The function uses TensorFlow's `list_physical_devices` method to query available devices, specifically
	  looking for 'GPU' devices.
	- This function is useful for dynamically adjusting computation strategies based on available hardware resources.
	
	Examples
	--------
	>>> has_gpu = auto_find_gpu()
	>>> print(f"GPU available: {has_gpu}")
	# GPU available: True or False based on the system's hardware configuration.
	"""

	gpus = list_physical_devices('GPU')
	if len(gpus)>0:
		use_gpu = True
	else:
		use_gpu = False

	return use_gpu

def _extract_channel_indices(channels, required_channels):

	"""
	Extracts the indices of required channels from a list of available channels.

	This function is designed to match the channels required by a model or analysis process with the channels
	present in the dataset. It returns the indices of the required channels within the list of available channels.
	If the required channels are not found among the available channels, the function prints an error message and
	returns None.

	Parameters
	----------
	channels : list of str or None
		A list containing the names of the channels available in the dataset. If None, it is assumed that the
		dataset channels are in the same order as the required channels.
	required_channels : list of str
		A list containing the names of the channels required by the model or analysis process.

	Returns
	-------
	ndarray or None
		An array of indices indicating the positions of the required channels within the list of available
		channels. Returns None if there is a mismatch between required and available channels.

	Notes
	-----
	- The function is useful for preprocessing steps where specific channels of multi-channel data are needed
	  for further analysis or model input.
	- In cases where `channels` is None, indicating that the dataset does not specify channel names, the function
	  assumes that the dataset's channel order matches the order of `required_channels` and returns an array of
	  indices based on this assumption.

	Examples
	--------
	>>> available_channels = ['DAPI', 'GFP', 'RFP']
	>>> required_channels = ['GFP', 'RFP']
	>>> indices = _extract_channel_indices(available_channels, required_channels)
	>>> print(indices)
	# [1, 2]

	>>> indices = _extract_channel_indices(None, required_channels)
	>>> print(indices)
	# [0, 1]
	"""

	channel_indices = []
	for c in required_channels:
		if c!='None' and c is not None:
			try:
				ch_idx = channels.index(c)
				channel_indices.append(ch_idx)
			except Exception as e:
				channel_indices.append(None)
		else:
			channel_indices.append(None)

	return channel_indices

def ConfigSectionMap(path,section):

	"""
	Parse the config file to extract experiment parameters
	following https://wiki.python.org/moin/ConfigParserExamples

	Parameters
	----------

	path: str
			path to the config.ini file

	section: str
			name of the section that contains the parameter

	Returns
	-------

	dict1: dictionary

	"""

	Config = configparser.ConfigParser()
	Config.read(path)
	dict1 = {}
	try:
		options = Config.options(section)
	except:
		return None
	for option in options:
		try:
			dict1[option] = Config.get(section, option)
			if dict1[option] == -1:
				DebugPrint("skip: %s" % option)
		except:
			print("exception on %s!" % option)
			dict1[option] = None
	return dict1

def _extract_channel_indices_from_config(config, channels_to_extract):

	"""
	Extracts the indices of specified channels from a configuration object.

	This function attempts to map required channel names to their respective indices as specified in a
	configuration file. It supports two versions of configuration parsing: a primary method (V2) and a
	fallback legacy method. If the required channels are not found using the primary method, the function
	attempts to find them using the legacy configuration settings.

	Parameters
	----------
	config : ConfigParser object
		The configuration object parsed from a .ini or similar configuration file that includes channel settings.
	channels_to_extract : list of str
		A list of channel names for which indices are to be extracted from the configuration settings.

	Returns
	-------
	list of int or None
		A list containing the indices of the specified channels as found in the configuration settings.
		If a channel cannot be found, None is appended in its place. If an error occurs during the extraction
		process, the function returns None.

	Notes
	-----
	- This function is designed to be flexible, accommodating changes in configuration file structure by
	  checking multiple sections for the required information.
	- The configuration file is expected to contain either "Channels" or "MovieSettings" sections with mappings
	  from channel names to indices.
	- An error message is printed if a required channel cannot be found, advising the user to check the
	  configuration file.

	Examples
	--------
	>>> config = ConfigParser()
	>>> config.read('example_config.ini')
	>>> channels_to_extract = ['GFP', 'RFP']
	>>> channel_indices = _extract_channel_indices_from_config(config, channels_to_extract)
	>>> print(channel_indices)
	# [1, 2] or None if an error occurs or the channels are not found.
	"""

	channels = []
	for c in channels_to_extract:
		try:
			c1 = int(ConfigSectionMap(config,"Channels")[c])
			channels.append(c1)
		except Exception as e:
			print(f"Warning... The channel {c} required by the model is not available in your data...")
			channels.append(None)
	if np.all([c is None for c in channels]):
		channels = None

	return channels

def _extract_nbr_channels_from_config(config, return_names=False):

	"""
	Extracts the indices of specified channels from a configuration object.

	This function attempts to map required channel names to their respective indices as specified in a
	configuration file. It supports two versions of configuration parsing: a primary method (V2) and a
	fallback legacy method. If the required channels are not found using the primary method, the function
	attempts to find them using the legacy configuration settings.

	Parameters
	----------
	config : ConfigParser object
		The configuration object parsed from a .ini or similar configuration file that includes channel settings.
	channels_to_extract : list of str
		A list of channel names for which indices are to be extracted from the configuration settings.

	Returns
	-------
	list of int or None
		A list containing the indices of the specified channels as found in the configuration settings.
		If a channel cannot be found, None is appended in its place. If an error occurs during the extraction
		process, the function returns None.

	Notes
	-----
	- This function is designed to be flexible, accommodating changes in configuration file structure by
	  checking multiple sections for the required information.
	- The configuration file is expected to contain either "Channels" or "MovieSettings" sections with mappings
	  from channel names to indices.
	- An error message is printed if a required channel cannot be found, advising the user to check the
	  configuration file.

	Examples
	--------
	>>> config = ConfigParser()
	>>> config.read('example_config.ini')
	>>> channels_to_extract = ['GFP', 'RFP']
	>>> channel_indices = _extract_channel_indices_from_config(config, channels_to_extract)
	>>> print(channel_indices)
	# [1, 2] or None if an error occurs or the channels are not found.
	"""

	# V2
	nbr_channels = 0
	channels = []
	try:
		fields = ConfigSectionMap(config,"Channels")
		for c in fields:
			try:
				channel = int(ConfigSectionMap(config, "Channels")[c])
				nbr_channels += 1
				channels.append(c)
			except:
				pass
	except:
		pass

	if nbr_channels==0:	

		# Read channels LEGACY
		nbr_channels = 0
		channels = []
		try:
			brightfield_channel = int(ConfigSectionMap(config,"MovieSettings")["brightfield_channel"])
			nbr_channels += 1
			channels.append('brightfield_channel')
		except:
			brightfield_channel = None

		try:
			live_nuclei_channel = int(ConfigSectionMap(config,"MovieSettings")["live_nuclei_channel"])
			nbr_channels += 1
			channels.append('live_nuclei_channel')
		except:
			live_nuclei_channel = None

		try:
			dead_nuclei_channel = int(ConfigSectionMap(config,"MovieSettings")["dead_nuclei_channel"])
			nbr_channels +=1
			channels.append('dead_nuclei_channel')
		except:
			dead_nuclei_channel = None

		try:
			effector_fluo_channel = int(ConfigSectionMap(config,"MovieSettings")["effector_fluo_channel"])
			nbr_channels +=1
			channels.append('effector_fluo_channel')
		except:
			effector_fluo_channel = None

		try:
			adhesion_channel = int(ConfigSectionMap(config,"MovieSettings")["adhesion_channel"])
			nbr_channels += 1
			channels.append('adhesion_channel')
		except:
			adhesion_channel = None

		try:
			fluo_channel_1 = int(ConfigSectionMap(config,"MovieSettings")["fluo_channel_1"])
			nbr_channels += 1
			channels.append('fluo_channel_1')
		except:
			fluo_channel_1 = None	

		try:
			fluo_channel_2 = int(ConfigSectionMap(config,"MovieSettings")["fluo_channel_2"])
			nbr_channels += 1
			channels.append('fluo_channel_2')
		except:
			fluo_channel_2 = None

	if return_names:
		return nbr_channels,channels
	else:
		return nbr_channels

def _get_img_num_per_channel(channels_indices, len_movie, nbr_channels):

	"""
	Calculates the image frame numbers for each specified channel in a multi-channel movie.

	Given the indices of channels of interest, the total length of the movie, and the number of channels,
	this function computes the frame numbers corresponding to each channel throughout the movie. If a
	channel index is specified as None, it assigns a placeholder value to indicate no frames for that channel.

	Parameters
	----------
	channels_indices : list of int or None
		A list containing the indices of channels for which to calculate frame numbers. If an index is None,
		it is interpreted as a channel with no frames to be processed.
	len_movie : int
		The total number of frames in the movie across all channels.
	nbr_channels : int
		The total number of channels in the movie.

	Returns
	-------
	ndarray
		A 2D numpy array where each row corresponds to a channel specified in `channels_indices` and contains
		the frame numbers for that channel throughout the movie. If a channel index is None, the corresponding
		row contains placeholder values (-1).

	Notes
	-----
	- The function assumes that frames in the movie are interleaved by channel, with frames for each channel
	  appearing in a regular sequence throughout the movie.
	- This utility is particularly useful for multi-channel time-lapse movies where analysis or processing
	  needs to be performed on a per-channel basis.

	Examples
	--------
	>>> channels_indices = [0, 2, None]  # Indices for channels 1, 3, and a non-existing channel
	>>> len_movie = 10  # Total frames for each channel
	>>> nbr_channels = 3  # Total channels in the movie
	>>> img_num_per_channel = _get_img_num_per_channel(channels_indices, len_movie, nbr_channels)
	>>> print(img_num_per_channel)
	# [[ 0  3  6  9 12 15 18 21 24 27]
	#  [ 2  5  8 11 14 17 20 23 26 29]
	#  [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1]]
	"""

	len_movie = int(len_movie)
	nbr_channels = int(nbr_channels)
	
	img_num_all_channels = []
	for c in channels_indices:
		if c is not None:
			indices = np.arange(len_movie*nbr_channels)[c::nbr_channels]
		else:
			indices = [-1]*len_movie
		img_num_all_channels.append(indices)
	img_num_all_channels = np.array(img_num_all_channels, dtype=int)	
	return img_num_all_channels

def _extract_labels_from_config(config,number_of_wells):

	"""

	Extract each well's biological condition from the configuration file

	Parameters
	----------

	config: str,
			path to the configuration file

	number_of_wells: int,
			total number of wells in the experiment

	Returns
	-------

	labels: string of the biological condition for each well

	"""
	
	try:
		concentrations = ConfigSectionMap(config,"Labels")["concentrations"].split(",")
		cell_types = ConfigSectionMap(config,"Labels")["cell_types"].split(",")
		antibodies = ConfigSectionMap(config,"Labels")["antibodies"].split(",")
		pharmaceutical_agents = ConfigSectionMap(config,"Labels")["pharmaceutical_agents"].split(",")
		index = np.arange(len(concentrations)).astype(int) + 1
		if not np.all(pharmaceutical_agents=="None"):
			labels = [f"W{idx}: [CT] "+a+"; [Ab] "+b+" @ "+c+" pM "+d for idx,a,b,c,d in zip(index,cell_types,antibodies,concentrations,pharmaceutical_agents)]
		else:
			labels = [f"W{idx}: [CT] "+a+"; [Ab] "+b+" @ "+c+" pM " for idx,a,b,c in zip(index,cell_types,antibodies,concentrations)]


	except Exception as e:
		print(f"{e}: the well labels cannot be read from the concentration and cell_type fields")
		labels = np.linspace(0,number_of_wells-1,number_of_wells,dtype=str)

	return(labels)

def extract_experiment_channels(config):

	"""
	Extracts channel names and their indices from an experiment configuration.

	This function attempts to parse channel information from a given configuration object, supporting
	both a newer (V2) and a legacy format. It first tries to extract channel names and indices according
	to the V2 format from the "Channels" section. If no channels are found or if the section does not
	exist, it falls back to extracting specific channel information from the "MovieSettings" section
	based on predefined channel names.

	Parameters
	----------
	config : ConfigParser object
		The configuration object parsed from an experiment's .ini or similar configuration file.

	Returns
	-------
	tuple
		A tuple containing two numpy arrays: `channel_names` and `channel_indices`. `channel_names` includes
		the names of the channels as specified in the configuration, and `channel_indices` includes their
		corresponding indices. Both arrays are ordered according to the channel indices.

	Notes
	-----
	- The function supports extracting a variety of channel types, including brightfield, live and dead nuclei
	  channels, effector fluorescence channels, adhesion channels, and generic fluorescence channels.
	- If channel information cannot be parsed or if required fields are missing, the function returns empty arrays.
	- This utility is particularly useful for preprocessing steps where specific channels of multi-channel
	  experimental data are needed for further analysis or model input.

	Examples
	--------
	>>> config = ConfigParser()
	>>> config.read('experiment_config.ini')
	>>> channel_names, channel_indices = extract_experiment_channels(config)
	# Extracts and sorts channel information based on indices from the experiment configuration.
	"""

	# V2
	channel_names = []
	channel_indices = []
	try:
		fields = ConfigSectionMap(config,"Channels")
		for c in fields:
			try:
				idx = int(ConfigSectionMap(config, "Channels")[c])
				channel_names.append(c)
				channel_indices.append(idx)
			except:
				pass
	except:
		pass


	if not channel_names:
		# LEGACY
		# Remap intensities to channel:
		channel_names = []
		channel_indices = []

		try:
			brightfield_channel = int(ConfigSectionMap(config,"MovieSettings")["brightfield_channel"])
			channel_names.append("brightfield_channel")
			channel_indices.append(brightfield_channel)
			#exp_channels.update({"brightfield_channel": brightfield_channel})
		except:
			pass
		try:
			live_nuclei_channel = int(ConfigSectionMap(config,"MovieSettings")["live_nuclei_channel"])
			channel_names.append("live_nuclei_channel")
			channel_indices.append(live_nuclei_channel)
			#exp_channels.update({"live_nuclei_channel": live_nuclei_channel})
		except:
			pass
		try:
			dead_nuclei_channel = int(ConfigSectionMap(config,"MovieSettings")["dead_nuclei_channel"])
			channel_names.append("dead_nuclei_channel")
			channel_indices.append(dead_nuclei_channel)
			#exp_channels.update({"dead_nuclei_channel": dead_nuclei_channel})
		except:
			pass
		try:
			effector_fluo_channel = int(ConfigSectionMap(config,"MovieSettings")["effector_fluo_channel"])
			channel_names.append("effector_fluo_channel")
			channel_indices.append(effector_fluo_channel)
			#exp_channels.update({"effector_fluo_channel": effector_fluo_channel})
		except:
			pass
		try:
			adhesion_channel = int(ConfigSectionMap(config,"MovieSettings")["adhesion_channel"])
			channel_names.append("adhesion_channel")
			channel_indices.append(adhesion_channel)
			#exp_channels.update({"adhesion_channel": adhesion_channel})
		except:
			pass
		try:
			fluo_channel_1 = int(ConfigSectionMap(config,"MovieSettings")["fluo_channel_1"])
			channel_names.append("fluo_channel_1")
			channel_indices.append(fluo_channel_1)
			#exp_channels.update({"fluo_channel_1": fluo_channel_1})
		except:
			pass
		try:
			fluo_channel_2 = int(ConfigSectionMap(config,"MovieSettings")["fluo_channel_2"])
			channel_names.append("fluo_channel_2")
			channel_indices.append(fluo_channel_2)
			#exp_channels.update({"fluo_channel_2": fluo_channel_2})
		except:
			pass

	channel_indices = np.array(channel_indices)
	channel_names = np.array(channel_names)
	reorder = np.argsort(channel_indices)
	channel_indices = channel_indices[reorder]
	channel_names = channel_names[reorder]

	return channel_names, channel_indices

def get_software_location():
	return rf"{os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]}"

def remove_trajectory_measurements(trajectories, column_labels):

	"""
	Filters a DataFrame of trajectory measurements to retain only essential tracking and classification columns.

	Given a DataFrame containing detailed trajectory measurements and metadata for tracked objects, this
	function reduces the DataFrame to include only a predefined set of essential columns necessary for
	further analysis or visualization. The set of columns to retain includes basic tracking information,
	spatial coordinates, classification results, and certain metadata.

	Parameters
	----------
	trajectories : pandas.DataFrame
		The DataFrame containing trajectory measurements and metadata for each tracked object.
	column_labels : dict
		A dictionary mapping standard column names to their corresponding column names in the `trajectories` DataFrame.
		Expected keys include 'track', 'time', 'x', 'y', among others.

	Returns
	-------
	pandas.DataFrame
		A filtered DataFrame containing only the essential columns as defined by `columns_to_keep` and present
		in `trajectories`.

	Notes
	-----
	- The function dynamically adjusts the list of columns to retain based on their presence in the input DataFrame,
	  ensuring compatibility with DataFrames containing varying sets of measurements.
	- Essential columns include tracking identifiers, time points, spatial coordinates (both pixel and physical units),
	  classification labels, state information, lineage metadata, and visualization attributes.

	Examples
	--------
	>>> column_labels = {
	...     'track': 'TRACK_ID', 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'
	... }
	>>> trajectories_df = pd.DataFrame({
	...     'TRACK_ID': [1, 1, 2],
	...     'FRAME': [0, 1, 0],
	...     'POSITION_X': [100, 105, 200],
	...     'POSITION_Y': [150, 155, 250],
	...     'velocity': [0.5, 0.5, 0.2],  # Additional column to be removed
	... })
	>>> filtered_df = remove_trajectory_measurements(trajectories_df, column_labels)
	# `filtered_df` will contain only the essential columns as per `column_labels` and predefined essential columns.
	"""

	tracks = trajectories.copy()

	columns_to_keep = [column_labels['track'], column_labels['time'], column_labels['x'], column_labels['y'],column_labels['x']+'_um', column_labels['y']+'_um', 'class_id', 
					  't', 'state', 'generation', 'root', 'parent', 'ID', 't0', 'class', 'status', 'class_color', 'status_color', 'class_firstdetection', 't_firstdetection', 'velocity']
	cols = tracks.columns
	for c in columns_to_keep:
		if c not in cols:
			columns_to_keep.remove(c)

	keep = [x for x in columns_to_keep if x in cols]
	tracks = tracks[keep]	

	return tracks


def color_from_status(status, recently_modified=False):
	
	if not recently_modified:
		if status==0:
			return 'tab:blue'
		elif status==1:
			return 'tab:red'
		elif status==2:
			return 'yellow'
		else:
			return 'k'
	else:
		if status==0:
			return 'tab:cyan'
		elif status==1:
			return 'tab:orange'
		elif status==2:
			return 'tab:olive'
		else:
			return 'k'

def color_from_class(cclass, recently_modified=False):

	if not recently_modified:
		if cclass==0:
			return 'tab:red'
		elif cclass==1:
			return 'tab:blue'
		elif cclass==2:
			return 'yellow'
		else:
			return 'k'
	else:
		if cclass==0:
			return 'tab:orange'
		elif cclass==1:
			return 'tab:cyan'
		elif cclass==2:
			return 'tab:olive'
		else:
			return 'k'

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

# def random_intensity_change(img):
#     img[img!=0.] = img[img!=0.]*np.random.uniform(0.3,2)
#     img[img!=0.] += np.random.uniform(-0.2,0.2)
#     return img

def random_shift(image,mask, max_shift_amplitude=0.1):

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
	max_shift = input_shape*max_shift_amplitude
	
	shift_value_x = random.choice(np.arange(max_shift))
	if np.random.random() > 0.5:
		shift_value_x*=-1

	shift_value_y = random.choice(np.arange(max_shift))
	if np.random.random() > 0.5:
		shift_value_y*=-1

	image = shift(image,[shift_value_x, shift_value_y, 0], output=np.float32, order=3, mode="constant",cval=0.0)
	mask = shift(mask,[shift_value_x,shift_value_y],order=0,mode="constant",cval=0.0)
	
	return image,mask


def blur(x,max_sigma=4.0):

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

	sigma = np.random.random()*max_sigma
	loc_i,loc_j,loc_c = np.where(x==0.)
	x = gaussian(x, sigma, channel_axis=-1, preserve_range=True)
	x[loc_i,loc_j,loc_c] = 0.

	return x

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
	loc_i,loc_j,loc_c = np.where(x_noise==0.)
	options =  ['gaussian', 'localvar', 'poisson', 'speckle']

	for k in range(x_noise.shape[-1]):
		mode_order = random.sample(options, len(options))
		for m in mode_order:
			p = np.random.random()
			if p <= apply_probability:
				try:
					x_noise[:,:,k] = random_noise(x_noise[:,:,k], mode=m, clip=clip_option)
				except:
					pass

	x_noise[loc_i,loc_j,loc_c] = 0.

	return x_noise



def augmenter(x, y, flip=True, gauss_blur=True, noise_option=True, shift=True, 
	channel_extinction=True, extinction_probability=0.1, clip=False, max_sigma_blur=4, 
	apply_noise_probability=0.5, augment_probability=0.9):

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
	if r<= augment_probability:

		if flip:
			x, y = random_fliprot(x, y)

		if gauss_blur:
			x = blur(x, max_sigma=max_sigma_blur)

		if noise_option:
			x = noise(x, apply_probability=apply_noise_probability, clip_option=clip)

		if shift:
			x,y = random_shift(x,y)
	  
		if channel_extinction:
			assert extinction_probability <= 1.,'The extinction probability must be a number between 0 and 1.'
			channel_off = [np.random.random() < extinction_probability for i in range(x.shape[-1])]
			channel_off[0] = False
			x[:,:,np.array(channel_off, dtype=bool)] = 0.

	return x, y

def normalize_per_channel(X, normalization_percentile_mode=True, normalization_values=[0.1,99.99],normalization_clipping=False):
	
	"""
	Applies per-channel normalization to a list of multi-channel images.

	This function normalizes each channel of every image in the list `X` based on either percentile values
	or fixed min-max values. Optionally, it can also clip the normalized values to stay within the [0, 1] range.
	The normalization can be applied in a percentile mode, where the lower and upper bounds for normalization
	are determined based on the specified percentiles of the non-zero values in each channel.

	Parameters
	----------
	X : list of ndarray
		A list of 3D numpy arrays, where each array represents a multi-channel image with dimensions
		(height, width, channels).
	normalization_percentile_mode : bool or list of bool, optional
		If True (or a list of True values), normalization bounds are determined by percentiles specified
		in `normalization_values` for each channel. If False, fixed `normalization_values` are used directly.
		Default is True.
	normalization_values : list of two floats or list of lists of two floats, optional
		The percentile values [lower, upper] used for normalization in percentile mode, or the fixed
		min-max values [min, max] for direct normalization. Default is [0.1, 99.99].
	normalization_clipping : bool or list of bool, optional
		Determines whether to clip the normalized values to the [0, 1] range for each channel. Default is False.

	Returns
	-------
	list of ndarray
		The list of normalized multi-channel images.

	Raises
	------
	AssertionError
		If the input images do not have a channel dimension, or if the lengths of `normalization_values`,
		`normalization_clipping`, and `normalization_percentile_mode` do not match the number of channels.

	Notes
	-----
	- The normalization is applied in-place, modifying the input list `X`.
	- This function is designed to handle multi-channel images commonly used in image processing and
	  computer vision tasks, particularly when different channels require separate normalization strategies.

	Examples
	--------
	>>> X = [np.random.rand(100, 100, 3) for _ in range(5)]  # Example list of 5 RGB images
	>>> normalized_X = normalize_per_channel(X)
	# Normalizes each channel of each image based on the default percentile values [0.1, 99.99].
	"""

	assert X[0].ndim==3,'Channel axis does not exist. Abort.'
	n_channels = X[0].shape[-1]
	if isinstance(normalization_percentile_mode, bool):
		normalization_percentile_mode = [normalization_percentile_mode]*n_channels
	if isinstance(normalization_clipping, bool):
		normalization_clipping = [normalization_clipping]*n_channels
	if len(normalization_values)==2 and not isinstance(normalization_values[0], list):
		normalization_values = [normalization_values]*n_channels

	assert len(normalization_values)==n_channels
	assert len(normalization_clipping)==n_channels
	assert len(normalization_percentile_mode)==n_channels
	
	X_normalized = []
	for i in range(len(X)):
		x = X[i].copy()
		loc_i,loc_j,loc_c = np.where(x==0.)
		norm_x = np.zeros_like(x, dtype=np.float32)
		for k in range(x.shape[-1]):
			chan = x[:,:,k].copy()
			if not np.all(chan.flatten()==0):
				if normalization_percentile_mode[k]:
					min_val = np.nanpercentile(chan[chan!=0.].flatten(), normalization_values[k][0])
					max_val = np.nanpercentile(chan[chan!=0.].flatten(), normalization_values[k][1])
				else:
					min_val = normalization_values[k][0]
					max_val = normalization_values[k][1]

				clip_option = normalization_clipping[k]
				norm_x[:,:,k] = normalize_mi_ma(chan.astype(np.float32).copy(), min_val, max_val, clip=clip_option, eps=1e-20, dtype=np.float32)
			else:
				norm_x[:,:,k] = 0.
		norm_x[loc_i,loc_j,loc_c] = 0.
		X_normalized.append(norm_x.copy())

	return X_normalized

def load_image_dataset(datasets, channels, train_spatial_calibration=None, mask_suffix='labelled'):

	"""
	Loads image and corresponding mask datasets, optionally applying spatial calibration adjustments.

	This function iterates over specified datasets, loading image and mask pairs based on provided channels
	and adjusting images according to a specified spatial calibration factor. It supports loading images with
	multiple channels and applies necessary transformations to match the training spatial calibration.

	Parameters
	----------
	datasets : list of str
		A list of paths to the datasets containing the images and masks.
	channels : str or list of str
		The channel(s) to be loaded from the images. If a string is provided, it is converted into a list.
	train_spatial_calibration : float, optional
		The spatial calibration (e.g., micrometers per pixel) used during model training. If provided, images
		will be rescaled to match this calibration. Default is None, indicating no rescaling is applied.
	mask_suffix : str, optional
		The suffix used to identify mask files corresponding to the images. Default is 'labelled'.

	Returns
	-------
	tuple of lists
		A tuple containing two lists: `X` for images and `Y` for corresponding masks. Both lists contain
		numpy arrays of loaded and optionally transformed images and masks.

	Raises
	------
	AssertionError
		If the provided `channels` argument is not a list or if the number of loaded images does not match
		the number of loaded masks.

	Notes
	-----
	- The function assumes that mask filenames are derived from image filenames by appending a `mask_suffix`
	  before the file extension.
	- Spatial calibration adjustment involves rescaling the images and masks to match the `train_spatial_calibration`.
	- Only images with a corresponding mask and a valid configuration file specifying channel indices and
	  spatial calibration are loaded.
	- The image samples must have at least one channel in common with the required channels to be accepted. The missing
	  channels are passed as black frames.

	Examples
	--------
	>>> datasets = ['/path/to/dataset1', '/path/to/dataset2']
	>>> channels = ['DAPI', 'GFP']
	>>> X, Y = load_image_dataset(datasets, channels, train_spatial_calibration=0.65)
	# Loads DAPI and GFP channels from specified datasets, rescaling images to match a spatial calibration of 0.65.
	"""

	if isinstance(channels, str):
		channels = [channels]
		
	assert isinstance(channels, list),'Please provide a list of channels. Abort.'

	X = []; Y = []; files = [];

	for ds in datasets:
		print(f'Loading data from dataset {ds}...')
		if not ds.endswith(os.sep):
			ds+=os.sep
		img_paths = list(set(glob(ds+'*.tif')) - set(glob(ds+f'*_{mask_suffix}.tif')))
		for im in img_paths:
			print(f'{im=}')
			mask_path = os.sep.join([os.path.split(im)[0],os.path.split(im)[-1].replace('.tif', f'_{mask_suffix}.tif')])
			if os.path.exists(mask_path):
				# load image and mask
				image = imread(im)
				if image.ndim==2:
					image = image[np.newaxis]
				if image.ndim>3:
					print('Invalid image shape, skipping')
					continue
				mask = imread(mask_path)
				config_path = im.replace('.tif','.json')
				if os.path.exists(config_path):
					# Load config
					with open(config_path, 'r') as f:
						config = json.load(f)

					existing_channels = config['channels']
					intersection = list(set(list(channels)) & set(list(existing_channels)))
					print(f'{existing_channels=} {intersection=}')
					if len(intersection)==0:
						print('Channels could not be found in the config... Skipping image.')
						continue
					else:
						ch_idx = []
						for c in channels:
							if c in existing_channels:
								idx = existing_channels.index(c)
								ch_idx.append(idx)
							else:
								# For None or missing channel pass black frame
								ch_idx.append(np.nan)
						im_calib = config['spatial_calibration']
				
				ch_idx = np.array(ch_idx)
				ch_idx_safe = np.copy(ch_idx)
				ch_idx_safe[ch_idx_safe!=ch_idx_safe] = 0
				ch_idx_safe = ch_idx_safe.astype(int)
				
				image = image[ch_idx_safe]
				image[np.where(ch_idx!=ch_idx)[0],:,:] = 0

				image = np.moveaxis(image,0,-1)
				assert image.ndim==3,'The image has a wrong number of dimensions. Abort.'
	
				if im_calib != train_spatial_calibration:
					factor = im_calib / train_spatial_calibration
					image = np.moveaxis([zoom(image[:,:,c].astype(float).copy(), [factor,factor], order=3, prefilter=False) for c in range(image.shape[-1])],0,-1) #zoom(image, [factor,factor,1], order=3)
					mask = zoom(mask, [factor,factor], order=0)        
					
			X.append(image)
			Y.append(mask)

			# fig,ax = plt.subplots(1,image.shape[-1]+1)
			# for k in range(image.shape[-1]):
			# 	ax[k].imshow(image[:,:,k],cmap='gray')
			# ax[image.shape[-1]].imshow(mask)
			# plt.pause(1)
			# plt.close()

			files.append(im)

	assert len(X)==len(Y),'The number of images does not match with the number of masks... Abort.'
	return X,Y,files


def download_url_to_file(url, dst, progress=True):
	r"""Download object at the given URL to a local path.
			Thanks to torch, slightly modified, from Cellpose
	Args:
		url (string): URL of the object to download
		dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
		progress (bool, optional): whether or not to display a progress bar to stderr
			Default: True
	"""
	file_size = None
	import ssl
	ssl._create_default_https_context = ssl._create_unverified_context
	u = urlopen(url)
	meta = u.info()
	if hasattr(meta, 'getheaders'):
		content_length = meta.getheaders("Content-Length")
	else:
		content_length = meta.get_all("Content-Length")
	if content_length is not None and len(content_length) > 0:
		file_size = int(content_length[0])
	# We deliberately save it in a temp file and move it after
	dst = os.path.expanduser(dst)
	dst_dir = os.path.dirname(dst)
	f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
	try:
		with tqdm(total=file_size, disable=not progress,
				  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
			while True:
				buffer = u.read(8192) #8192
				if len(buffer) == 0:
					break
				f.write(buffer)
				pbar.update(len(buffer))
		f.close()
		shutil.move(f.name, dst)
	finally:
		f.close()
		if os.path.exists(f.name):
			os.remove(f.name)

def get_zenodo_files(cat=None):


	zenodo_json = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective", "links", "zenodo.json"])
	with open(zenodo_json,"r") as f:
		zenodo_json = json.load(f)
	all_files = list(zenodo_json['files']['entries'].keys())
	all_files_short = [f.replace(".zip","") for f in all_files]
	
	categories = []
	for f in all_files_short:
		if f.startswith('CP') or f.startswith('SD'):
			category = os.sep.join(['models','segmentation_generic'])
		elif f.startswith('MCF7') or f.startswith('mcf7'):
			category = os.sep.join(['models','segmentation_targets'])
		elif f.startswith('primNK') or f.startswith('lymphocytes'):
			category = os.sep.join(['models','segmentation_effectors'])
		elif f.startswith('demo'):
			category = 'demos'
		elif f.startswith('db-si'):
			category = os.sep.join(['datasets','signal_annotations'])
		elif f.startswith('db'):
			category = os.sep.join(['datasets','segmentation_annotations'])
		else:
			category = os.sep.join(['models','signal_detection'])
		categories.append(category)
	
	if cat is not None:
		assert cat in [os.sep.join(['models','segmentation_generic']), os.sep.join(['models','segmentation_targets']), os.sep.join(['models','segmentation_effectors']), \
						   'demos', os.sep.join(['datasets','signal_annotations']), os.sep.join(['datasets','segmentation_annotations']),  os.sep.join(['models','signal_detection'])]
		categories = np.array(categories)
		all_files_short = np.array(all_files_short)
		return list(all_files_short[np.where(categories==cat)[0]])
	else:
		return all_files_short,categories

def download_zenodo_file(file, output_dir):
	
	zenodo_json = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective", "links", "zenodo.json"])
	with open(zenodo_json,"r") as f:
		zenodo_json = json.load(f)
	all_files = list(zenodo_json['files']['entries'].keys())
	all_files_short = [f.replace(".zip","") for f in all_files]
	zenodo_url = zenodo_json['links']['files'].replace('api/','')
	full_links = ["/".join([zenodo_url, f]) for f in all_files]
	index = all_files_short.index(file)
	zip_url = full_links[index]
	
	path_to_zip_file = os.sep.join([output_dir, 'temp.zip'])
	download_url_to_file(fr"{zip_url}",path_to_zip_file)
	with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
		zip_ref.extractall(output_dir)
	
	file_to_rename = glob(os.sep.join([output_dir,file,"*[!.json][!.png][!.h5][!.csv][!.npy][!.tif][!.ini]"]))
	if len(file_to_rename)>0 and not file_to_rename[0].endswith(os.sep) and not file.startswith('demo'):
		os.rename(file_to_rename[0], os.sep.join([output_dir,file,file]))

	os.remove(path_to_zip_file)

def interpolate_nan(img, method='nearest'):

	"""
	Interpolate NaN on single channel array 2D
	"""

	if np.all(img==0):
		return img

	if np.any(img.flatten()!=img.flatten()):
		# then need to interpolate
		x_grid, y_grid = np.meshgrid(np.arange(img.shape[1]),np.arange(img.shape[0]))
		mask = [~np.isnan(img)][0]
		x = x_grid[mask].reshape(-1)
		y = y_grid[mask].reshape(-1)
		points = np.array([x,y]).T
		values = img[mask].reshape(-1)
		interp_grid = griddata(points, values, (x_grid, y_grid), method=method)
		return interp_grid
	else:
		return img


def interpolate_nan_multichannel(frames):
	frames = np.moveaxis([interpolate_nan(frames[:,:,c].copy()) for c in range(frames.shape[-1])],0,-1)
	return frames

def collapse_trajectories_by_status(df, status=None, projection='mean', population='effectors', groupby_columns=['position','TRACK_ID']):
	
	static_columns = ['well_index', 'well_name', 'pos_name', 'position', 'well', 'status', 't0', 'class','cell_type','concentration', 'antibody', 'pharmaceutical_agent','TRACK_ID','position', 'neighbor_population', 'reference_population', 'NEIGHBOR_ID', 'REFERENCE_ID', 'FRAME']

	if status is None or status not in list(df.columns):
		print('invalid status selection...')
		return None

	df = df.dropna(subset=status,ignore_index=True)
	unique_statuses = np.unique(df[status].to_numpy())

	df_sections = []
	for s in unique_statuses:
		subtab = df.loc[df[status]==s,:]
		op = getattr(subtab.groupby(groupby_columns), projection)
		subtab_projected = op(subtab.groupby(groupby_columns))
		frame_duration = subtab.groupby(groupby_columns).size().to_numpy()
		for c in static_columns:
			try:
				subtab_projected[c] = subtab.groupby(groupby_columns)[c].apply(lambda x: x.unique()[0])
			except Exception as e:
				print(e)
				pass
		subtab_projected['duration_in_state'] = frame_duration
		df_sections.append(subtab_projected)

	group_table = pd.concat(df_sections,axis=0,ignore_index=True)
	if population=='pairs':
		for col in ['duration_in_state',status, 'neighbor_population', 'reference_population', 'NEIGHBOR_ID', 'REFERENCE_ID']:
			first_column = group_table.pop(col) 
			group_table.insert(0, col, first_column)				
	else:
		for col in ['duration_in_state',status,'TRACK_ID']:
			first_column = group_table.pop(col) 
			group_table.insert(0, col, first_column)

	group_table.pop('FRAME')
	group_table = group_table.sort_values(by=groupby_columns + [status],ignore_index=True)
	group_table = group_table.reset_index(drop=True)

	return group_table

def step_function(t, t_shift, dt):

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

	return 1/(1+np.exp(-(t-t_shift)/dt))


def test_2samp_generic(data, feature=None, groupby_cols=None, method="ks_2samp", *args, **kwargs):

	"""
	Performs pairwise statistical tests between groups of data, comparing a specified feature using a chosen method.
	
	The function applies two-sample statistical tests, such as the Kolmogorov-Smirnov (KS) test or Cliff's Delta,
	to compare distributions of a given feature across groups defined by `groupby_cols`. It returns the test results
	in a pivot table format with each group's pairwise comparison.

	Parameters
	----------
	data : pandas.DataFrame
		The input dataset containing the feature to be tested.
	feature : str
		The name of the column representing the feature to compare between groups.
	groupby_cols : list or str
		The column(s) used to group the data. These columns define the groups that will be compared pairwise.
	method : str, optional, default="ks_2samp"
		The statistical test to use. Options:
		- "ks_2samp": Two-sample Kolmogorov-Smirnov test (default).
		- "cliffs_delta": Cliff's Delta for effect size between two distributions.
	*args, **kwargs : 
		Additional arguments and keyword arguments for the selected test method.

	Returns
	-------
	pivot : pandas.DataFrame
		A pivot table containing the pairwise test results (p-values or effect sizes). 
		The rows and columns represent the unique groups defined by `groupby_cols`, 
		and the values represent the test result (e.g., p-values or effect sizes) between each group.

	Notes
	-----
	- The function compares all unique pairwise combinations of the groups based on `groupby_cols`.
	- For the "ks_2samp" method, the test compares the distributions using the Kolmogorov-Smirnov test.
	- For the "cliffs_delta" method, the function calculates the effect size between two distributions.
	- The results are returned in a symmetric pivot table where each cell represents the test result for the corresponding group pair.
	
	"""


	assert groupby_cols is not None,"Please set a valid groupby_cols..."
	assert feature is not None,"Please set a feature to test..."

	results = []

	for lbl1,group1 in data.dropna(subset=feature).groupby(groupby_cols):
		for lbl2,group2 in data.dropna(subset=feature).groupby(groupby_cols):

			dist1 = group1[feature].values
			dist2 = group2[feature].values
			if method=="ks_2samp":
				test = ks_2samp(list(dist1),list(dist2), alternative='less', mode='auto', *args, **kwargs)
				val = test.pvalue
			elif method=="cliffs_delta":
				test = cliffs_delta(list(dist1),list(dist2), *args, **kwargs)
				val = test[0]

			results.append({"cdt1": lbl1, "cdt2": lbl2, "value": val})
	
	results = pd.DataFrame(results)
	results['cdt1'] = results['cdt1'].astype(str)
	results['cdt2'] = results['cdt2'].astype(str)

	pivot = results.pivot(index='cdt1', columns='cdt2', values='value')
	pivot.reset_index(inplace=True)
	pivot.columns.name = None
	pivot.set_index("cdt1",drop=True, inplace=True)
	pivot.index.name = None

	return pivot