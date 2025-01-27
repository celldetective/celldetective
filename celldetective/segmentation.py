"""
Segmentation module
"""
import json
import os
from .io import locate_segmentation_model, normalize_multichannel
from .utils import _estimate_scale_factor, _extract_channel_indices
from pathlib import Path
from tqdm import tqdm
import numpy as np
from celldetective.io import _view_on_napari, locate_labels, locate_stack, _view_on_napari, _check_label_dims, auto_correct_masks
from celldetective.filters import * #rework this to give a name
from celldetective.utils import interpolate_nan_multichannel,_rearrange_multichannel_frame, _fix_no_contrast, zoom_multiframes, _rescale_labels, rename_intensity_column, mask_edges, _prep_stardist_model, _prep_cellpose_model, estimate_unreliable_edge,_get_normalize_kwargs_from_config, _segment_image_with_stardist_model, _segment_image_with_cellpose_model
from stardist import fill_label_holes
from stardist.matching import matching

import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table
from skimage.exposure import match_histograms
from scipy.ndimage import zoom
import pandas as pd
import subprocess


abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],'celldetective'])

def segment(stack, model_name, channels=None, spatial_calibration=None, view_on_napari=False,
			use_gpu=True, channel_axis=-1, cellprob_threshold=None, flow_threshold=None):
	
	"""
	
	Segment objects in a stack using a pre-trained segmentation model.

	Parameters
	----------
	stack : ndarray
		The input stack to be segmented, with shape (frames, height, width, channels).
	model_name : str
		The name of the pre-trained segmentation model to use.
	channels : list or None, optional
		The names of the channels in the stack. If None, assumes the channels are indexed from 0 to `stack.shape[-1] - 1`.
		Default is None.
	spatial_calibration : float or None, optional
		The spatial calibration factor of the stack. If None, the calibration factor from the model configuration will be used.
		Default is None.
	view_on_napari : bool, optional
		Whether to visualize the segmentation results using Napari. Default is False.
	use_gpu : bool, optional
		Whether to use GPU acceleration if available. Default is True.

	Returns
	-------
	ndarray
		The segmented labels with shape (frames, height, width).

	Notes
	-----
	This function applies object segmentation to a stack of images using a pre-trained segmentation model. The stack is first
	preprocessed by normalizing the intensity values, rescaling the spatial dimensions, and applying the segmentation model.
	The resulting labels are returned as an ndarray with the same number of frames as the input stack.

	Examples
	--------
	>>> stack = np.random.rand(10, 256, 256, 3)
	>>> labels = segment(stack, 'model_name', channels=['channel_1', 'channel_2', 'channel_3'], spatial_calibration=0.5)

	"""

	model_path = locate_segmentation_model(model_name)
	input_config = model_path+'config_input.json'
	if os.path.exists(input_config):
		with open(input_config) as config:
			print("Loading input configuration from 'config_input.json'.")
			input_config = json.load(config)
	else:
		print('Model input configuration could not be located...')
		return None

	if not use_gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	else:
		os.environ['CUDA_VISIBLE_DEVICES'] = '0'

	if channel_axis != -1:
		stack = np.moveaxis(stack, channel_axis, -1)

	if channels is not None:
		assert len(channels)==stack.shape[-1],f'The channel names provided do not match with the expected number of channels in the stack: {stack.shape[-1]}.'

	required_channels = input_config['channels']
	channel_intersection = [ch for ch in channels if ch in required_channels]
	assert len(channel_intersection)>0,'None of the channels required by the model can be found in the images to segment... Abort.'

	channel_indices = _extract_channel_indices(channels, required_channels)

	required_spatial_calibration = input_config['spatial_calibration']
	model_type = input_config['model_type']

	normalize_kwargs = _get_normalize_kwargs_from_config(input_config)

	if model_type=='cellpose':
		diameter = input_config['diameter']
		# if diameter!=30:
		# 	required_spatial_calibration = None
		if cellprob_threshold is None:
			cellprob_threshold = input_config['cellprob_threshold']
		if flow_threshold is None:
			flow_threshold = input_config['flow_threshold']

	scale = _estimate_scale_factor(spatial_calibration, required_spatial_calibration)
	print(f"{spatial_calibration=} {required_spatial_calibration=} Scale = {scale}...")

	if model_type=='stardist':
		model, scale_model = _prep_stardist_model(model_name, Path(model_path).parent, use_gpu=use_gpu, scale=scale)

	elif model_type=='cellpose':
		model, scale_model = _prep_cellpose_model(model_path.split('/')[-2], model_path, use_gpu=use_gpu, n_channels=len(required_channels), scale=scale)		

	labels = []

	for t in tqdm(range(len(stack)),desc="frame"):

		# normalize
		channel_indices = np.array(channel_indices)
		none_channel_indices = np.where(channel_indices==None)[0]
		channel_indices[channel_indices==None] = 0

		frame = stack[t]
		frame = _rearrange_multichannel_frame(frame).astype(float)

		frame_to_segment = np.zeros((frame.shape[0], frame.shape[1], len(required_channels))).astype(float)
		for ch in channel_intersection:
			idx = required_channels.index(ch)
			frame_to_segment[:,:,idx] = frame[:,:,channels.index(ch)]
		frame = frame_to_segment
		template = frame.copy()

		frame = normalize_multichannel(frame, **normalize_kwargs)
		
		if scale_model is not None:
			frame = zoom_multiframes(frame, scale_model)
		
		frame = _fix_no_contrast(frame)
		frame = interpolate_nan_multichannel(frame)
		frame[:,:,none_channel_indices] = 0.

		if model_type=="stardist":
			Y_pred = _segment_image_with_stardist_model(frame, model=model, return_details=False)

		elif model_type=="cellpose":
			Y_pred = _segment_image_with_cellpose_model(frame, model=model, diameter=diameter, cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold)

		if Y_pred.shape != stack[0].shape[:2]:
			Y_pred = _rescale_labels(Y_pred, scale_model)
		
		Y_pred = _check_label_dims(Y_pred, template=template)

		labels.append(Y_pred)

	labels = np.array(labels,dtype=int)

	if view_on_napari:
		_view_on_napari(tracks=None, stack=stack, labels=labels)

	return labels


def segment_from_thresholds(stack, target_channel=0, thresholds=None, view_on_napari=False, equalize_reference=None,
							filters=None, marker_min_distance=30, marker_footprint_size=20, marker_footprint=None, feature_queries=None):

	"""
	Segments objects from a stack of images based on provided thresholds and optional image processing steps.

	This function applies instance segmentation to each frame in a stack of images. Segmentation is based on intensity 
	thresholds, optionally preceded by image equalization and filtering. Identified objects can 
	be distinguished by applying distance-based marker detection. The segmentation results can be optionally viewed in Napari.

	Parameters
	----------
	stack : ndarray
		A 4D numpy array representing the image stack with dimensions (T, Y, X, C) where T is the 
		time dimension and C the channel dimension.
	target_channel : int, optional
		The channel index to be used for segmentation (default is 0).
	thresholds : list of tuples, optional
		A list of tuples specifying intensity thresholds for segmentation. Each tuple corresponds to a frame in the stack, 
		with values (lower_threshold, upper_threshold). If None, global thresholds are determined automatically (default is None).
	view_on_napari : bool, optional
		If True, displays the original stack and segmentation results in Napari (default is False).
	equalize_reference : int or None, optional
		The index of a reference frame used for histogram equalization. If None, equalization is not performed (default is None).
	filters : list of dict, optional
		A list of dictionaries specifying filters to be applied pre-segmentation. Each dictionary should 
		contain filter parameters (default is None).
	marker_min_distance : int, optional
		The minimum distance between markers used for distinguishing separate objects (default is 30).
	marker_footprint_size : int, optional
		The size of the footprint used for local maxima detection when generating markers (default is 20).
	marker_footprint : ndarray or None, optional
		An array specifying the footprint used for local maxima detection. Overrides `marker_footprint_size` if provided 
		(default is None).
	feature_queries : list of str or None, optional
		A list of query strings used to select features of interest from the segmented objects (default is None).

	Returns
	-------
	ndarray
		A 3D numpy array (T, Y, X) of type int16, where each element represents the segmented object label at each pixel.

	Notes
	-----
	- The segmentation process can be customized extensively via the parameters, allowing for complex segmentation tasks.

	"""


	masks = []
	for t in tqdm(range(len(stack))):
		instance_seg = segment_frame_from_thresholds(stack[t], target_channel=target_channel, thresholds=thresholds, equalize_reference=equalize_reference,
													filters=filters, marker_min_distance=marker_min_distance, marker_footprint_size=marker_footprint_size,
													marker_footprint=marker_footprint, feature_queries=feature_queries)
		masks.append(instance_seg)

	masks = np.array(masks, dtype=np.int16)
	if view_on_napari:
		_view_on_napari(tracks=None, stack=stack, labels=masks)
	return masks

def segment_frame_from_thresholds(frame, target_channel=0, thresholds=None, equalize_reference=None,
								  filters=None, marker_min_distance=30, marker_footprint_size=20, marker_footprint=None, feature_queries=None, channel_names=None, do_watershed=True, edge_exclusion=True, fill_holes=True):
	
	"""
	Segments objects within a single frame based on intensity thresholds and optional image processing steps.

	This function performs instance segmentation on a single frame using intensity thresholds, with optional steps
	including histogram equalization, filtering, and marker-based watershed segmentation. The segmented
	objects can be further filtered based on specified features.

	Parameters
	----------
	frame : ndarray
		A 3D numpy array representing a single frame with dimensions (Y, X, C).
	target_channel : int, optional
		The channel index to be used for segmentation (default is 0).
	thresholds : tuple of int, optional
		A tuple specifying the intensity thresholds for segmentation, in the form (lower_threshold, upper_threshold).
	equalize_reference : ndarray or None, optional
		A 2D numpy array used as a reference for histogram equalization. If None, equalization is not performed (default is None).
	filters : list of dict, optional
		A list of dictionaries specifying filters to be applied to the image before segmentation. Each dictionary
		should contain filter parameters (default is None).
	marker_min_distance : int, optional
		The minimum distance between markers used for distinguishing separate objects during watershed segmentation (default is 30).
	marker_footprint_size : int, optional
		The size of the footprint used for local maxima detection when generating markers for watershed segmentation (default is 20).
	marker_footprint : ndarray or None, optional
		An array specifying the footprint used for local maxima detection. Overrides `marker_footprint_size` if provided (default is None).
	feature_queries : list of str or None, optional
		A list of query strings used to select features of interest from the segmented objects for further filtering (default is None).
	channel_names : list of str or None, optional
		A list of channel names corresponding to the dimensions in `frame`, used in conjunction with `feature_queries` for feature selection (default is None).

	Returns
	-------
	ndarray
		A 2D numpy array of type int, where each element represents the segmented object label at each pixel.

	"""

	if frame.ndim==2:
		frame = frame[:,:,np.newaxis]
	img = frame[:,:,target_channel]

	if np.any(img!=img):
		img = interpolate_nan(img)

	if equalize_reference is not None:
		img = match_histograms(img, equalize_reference)

	img_mc = frame.copy()
	img = filter_image(img, filters=filters)
	if edge_exclusion:
		edge = estimate_unreliable_edge(filters)
	else:
		edge = None

	binary_image = threshold_image(img, thresholds[0], thresholds[1], fill_holes=fill_holes, edge_exclusion=edge)
	
	if do_watershed:
		coords,distance = identify_markers_from_binary(binary_image, marker_min_distance, footprint_size=marker_footprint_size, footprint=marker_footprint, return_edt=True)
		instance_seg = apply_watershed(binary_image, coords, distance)
	else:
		instance_seg, _ = ndi.label(binary_image.astype(int).copy())

	instance_seg = filter_on_property(instance_seg, intensity_image=img_mc, queries=feature_queries, channel_names=channel_names)

	return instance_seg


def filter_on_property(labels, intensity_image=None, queries=None, channel_names=None):

	"""
	Filters segmented objects in a label image based on specified properties and queries.

	This function evaluates each segmented object (label) in the input label image against a set of queries related to its 
	morphological and intensity properties. Objects not meeting the criteria defined in the queries are removed from the label 
	image. This allows for the exclusion of objects based on size, shape, intensity, or custom-defined properties.

	Parameters
	----------
	labels : ndarray
		A 2D numpy array where each unique non-zero integer represents a segmented object (label).
	intensity_image : ndarray, optional
		A 2D numpy array of the same shape as `labels`, providing intensity values for each pixel. This is used to calculate
		intensity-related properties of the segmented objects if provided (default is None).
	queries : str or list of str, optional
		One or more query strings used to filter the segmented objects based on their properties. Each query should be a 
		valid pandas query string (default is None).
	channel_names : list of str or None, optional
		A list of channel names corresponding to the dimensions in the `intensity_image`. This is used to rename intensity
		property columns appropriately (default is None).

	Returns
	-------
	ndarray
		A 2D numpy array of the same shape as `labels`, with objects not meeting the query criteria removed.

	Notes
	-----
	- The function computes a set of predefined morphological properties and, if `intensity_image` is provided, intensity properties.
	- Queries should be structured according to pandas DataFrame query syntax and can reference any of the computed properties.
	- If `channel_names` is provided, intensity property column names are renamed to reflect the corresponding channel.

	"""

	if queries is None:
		return labels
	else:
		if isinstance(queries, str):
			queries = [queries]

	props = ['label','area', 'area_bbox', 'area_convex', 'area_filled', 'axis_major_length',
						'axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 
						'euler_number', 'feret_diameter_max', 'orientation', 'perimeter',
						'perimeter_crofton', 'solidity', 'centroid']

	intensity_props = ['intensity_mean', 'intensity_max', 'intensity_min']

	if intensity_image is not None:
		props.extend(intensity_props)

	properties = pd.DataFrame(regionprops_table(labels, intensity_image=intensity_image, properties=props))
	if channel_names is not None:
		properties = rename_intensity_column(properties, channel_names)
	properties['radial_distance'] = np.sqrt((properties['centroid-1'] - labels.shape[0]/2)**2 + (properties['centroid-0'] - labels.shape[1]/2)**2)

	for query in queries:
		if query!='':
			try:
				properties = properties.query(f'not ({query})')
			except Exception as e:
				print(f'Query {query} could not be applied. Ensure that the feature exists. {e}')
		else:
			pass

	cell_ids = list(np.unique(labels)[1:])
	leftover_cells = list(properties['label'].unique())
	to_remove = [value for value in cell_ids if value not in leftover_cells]

	for c in to_remove:
		labels[np.where(labels==c)] = 0.

	return labels


def apply_watershed(binary_image, coords, distance, fill_holes=True):

	"""
	Applies the watershed algorithm to segment objects in a binary image using given markers and distance map.

	This function uses the watershed segmentation algorithm to delineate objects in a binary image. Markers for watershed
	are determined by the coordinates of local maxima, and the segmentation is guided by a distance map to separate objects
	that are close to each other.

	Parameters
	----------
	binary_image : ndarray
		A 2D numpy array of type bool, where True represents the foreground objects to be segmented and False represents the background.
	coords : ndarray
		An array of shape (N, 2) containing the (row, column) coordinates of local maxima points that will be used as markers for the
		watershed algorithm. N is the number of local maxima.
	distance : ndarray
		A 2D numpy array of the same shape as `binary_image`, containing the distance transform of the binary image. This map is used
		to guide the watershed segmentation.

	Returns
	-------
	ndarray
		A 2D numpy array of type int, where each unique non-zero integer represents a segmented object (label).

	Notes
	-----
	- The function assumes that `coords` are derived from the distance map of `binary_image`, typically obtained using
	  peak local max detection on the distance transform.
	- The watershed algorithm treats each local maximum as a separate object and segments the image by "flooding" from these points.
	- This implementation uses the `skimage.morphology.watershed` function under the hood.

	Examples
	--------
	>>> from skimage import measure, morphology
	>>> binary_image = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0]], dtype=bool)
	>>> distance = morphology.distance_transform_edt(binary_image)
	>>> coords = measure.peak_local_max(distance, indices=True)
	>>> labels = apply_watershed(binary_image, coords, distance)
	# Segments the objects in `binary_image` using the watershed algorithm.
	
	"""

	mask = np.zeros(binary_image.shape, dtype=bool)
	mask[tuple(coords.T)] = True
	markers, _ = ndi.label(mask)
	labels = watershed(-distance, markers, mask=binary_image)
	if fill_holes:
		labels = fill_label_holes(labels)
	return labels

def identify_markers_from_binary(binary_image, min_distance, footprint_size=20, footprint=None, return_edt=False):

	"""

	Identify markers from a binary image using distance transform and peak detection.

	Parameters
	----------
	binary_image : ndarray
		The binary image from which to identify markers.
	min_distance : int
		The minimum distance between markers. Only the markers with a minimum distance greater than or equal to
		`min_distance` will be identified.
	footprint_size : int, optional
		The size of the footprint or structuring element used for peak detection. Default is 20.
	footprint : ndarray, optional
		The footprint or structuring element used for peak detection. If None, a square footprint of size
		`footprint_size` will be used. Default is None.
	return_edt : bool, optional
		Whether to return the Euclidean distance transform image along with the identified marker coordinates.
		If True, the function will return the marker coordinates and the distance transform image as a tuple.
		If False, only the marker coordinates will be returned. Default is False.

	Returns
	-------
	ndarray or tuple
		If `return_edt` is False, returns the identified marker coordinates as an ndarray of shape (N, 2), where N is
		the number of identified markers. If `return_edt` is True, returns a tuple containing the marker coordinates
		and the distance transform image.

	Notes
	-----
	This function uses the distance transform of the binary image to identify markers by detecting local maxima. The
	distance transform assigns each pixel a value representing the Euclidean distance to the nearest background pixel.
	By finding peaks in the distance transform, we can identify the markers in the original binary image. The `min_distance`
	parameter controls the minimum distance between markers to avoid clustering.

	"""
	
	distance = ndi.distance_transform_edt(binary_image.astype(float))
	if footprint is None:
		footprint = np.ones((footprint_size, footprint_size))
	coords = peak_local_max(distance, footprint=footprint,
		labels=binary_image.astype(int), min_distance=min_distance)
	if return_edt:
		return coords, distance
	else:
		return coords


def threshold_image(img, min_threshold, max_threshold, foreground_value=255., fill_holes=True, edge_exclusion=None):

	"""
	
	Threshold the input image to create a binary mask.

	Parameters
	----------
	img : ndarray
		The input image to be thresholded.
	min_threshold : float
		The minimum threshold value.
	max_threshold : float
		The maximum threshold value.
	foreground_value : float, optional
		The value assigned to foreground pixels in the binary mask. Default is 255.
	fill_holes : bool, optional
		Whether to fill holes in the binary mask. If True, the binary mask will be processed to fill any holes.
		If False, the binary mask will not be modified. Default is True.

	Returns
	-------
	ndarray
		The binary mask after thresholding.

	Notes
	-----
	This function applies a threshold to the input image to create a binary mask. Pixels with values within the specified
	threshold range are considered as foreground and assigned the `foreground_value`, while pixels outside the range are
	considered as background and assigned 0. If `fill_holes` is True, the binary mask will be processed to fill any holes
	using morphological operations.

	Examples
	--------
	>>> image = np.random.rand(256, 256)
	>>> binary_mask = threshold_image(image, 0.2, 0.8, foreground_value=1., fill_holes=True)

	"""

	binary = np.zeros_like(img).astype(bool)
	binary[img==img] = (img[img==img]>=min_threshold)*(img[img==img]<=max_threshold) * foreground_value
	if isinstance(edge_exclusion, (int,np.int_)):
		binary = mask_edges(binary, edge_exclusion)
	if fill_holes:
		binary = ndi.binary_fill_holes(binary.astype(int))
	return binary

def filter_image(img, filters=None):

	"""

	Apply one or more image filters to the input image.

	Parameters
	----------
	img : ndarray
		The input image to be filtered.
	filters : list or None, optional
		A list of filters to be applied to the image. Each filter is represented as a tuple or list with the first element being
		the filter function name (minus the '_filter' extension, as listed in software.filters) and the subsequent elements being 
		the arguments for that filter function. If None, the original image is returned without any filtering applied. Default is None.

	Returns
	-------
	ndarray
		The filtered image.

	Notes
	-----
	This function applies a series of image filters to the input image. The filters are specified as a list of tuples,
	where each tuple contains the name of the filter function and its corresponding arguments. The filters are applied
	sequentially to the image. If no filters are provided, the original image is returned unchanged.

	Examples
	--------
	>>> image = np.random.rand(256, 256)
	>>> filtered_image = filter_image(image, filters=[('gaussian', 3), ('median', 5)])

	"""

	if filters is None:
		return img

	if img.ndim==3:
		img = np.squeeze(img)

	for f in filters:
		func = eval(f[0]+'_filter')
		img = func(img, *f[1:])	
	return img


def segment_at_position(pos, mode, model_name, stack_prefix=None, use_gpu=True, return_labels=False, view_on_napari=False, threads=1):

	"""
	Perform image segmentation at the specified position using a pre-trained model.

	Parameters
	----------
	pos : str
		The path to the position directory containing the input images to be segmented.
	mode : str
		The segmentation mode. This determines the type of objects to be segmented ('target' or 'effector').
	model_name : str
		The name of the pre-trained segmentation model to be used.
	stack_prefix : str or None, optional
		The prefix of the stack file name. Defaults to None.
	use_gpu : bool, optional
		Whether to use the GPU for segmentation if available. Defaults to True.
	return_labels : bool, optional
		If True, the function returns the segmentation labels as an output. Defaults to False.
	view_on_napari : bool, optional
		If True, the segmented labels are displayed in a Napari viewer. Defaults to False.

	Returns
	-------
	numpy.ndarray or None
		If `return_labels` is True, the function returns the segmentation labels as a NumPy array. Otherwise, it returns None. The subprocess writes the
		segmentation labels in the position directory.

	Examples
	--------
	>>> labels = segment_at_position('ExperimentFolder/W1/100/', 'effector', 'mice_t_cell_RICM', return_labels=True)

	"""
	
	pos = pos.replace('\\','/')
	pos = rf'{pos}'
	assert os.path.exists(pos),f'Position {pos} is not a valid path.'

	name_path = locate_segmentation_model(model_name)
	
	script_path = os.sep.join([abs_path, 'scripts', 'segment_cells.py'])
	cmd = f'python "{script_path}" --pos "{pos}" --model "{model_name}" --mode "{mode}" --use_gpu "{use_gpu}" --threads "{threads}"'
	subprocess.call(cmd, shell=True)

	if return_labels or view_on_napari:
		labels = locate_labels(pos, population=mode)
	if view_on_napari:
		if stack_prefix is None:
			stack_prefix = ''
		stack = locate_stack(pos, prefix=stack_prefix)
		_view_on_napari(tracks=None, stack=stack, labels=labels)
	if return_labels:
		return labels
	else:
		return None

def segment_from_threshold_at_position(pos, mode, config, threads=1):

	"""
	Executes a segmentation script on a specified position directory using a given configuration and mode.

	This function calls an external Python script designed to segment images at a specified position directory.
	The segmentation is configured through a JSON file and can operate in different modes specified by the user.
	The function can leverage multiple threads to potentially speed up the processing.

	Parameters
	----------
	pos : str
		The file path to the position directory where images to be segmented are stored. The path must be valid.
	mode : str
		The operation mode for the segmentation script. The mode determines how the segmentation is performed and 
		which algorithm or parameters are used.
	config : str
		The file path to the JSON configuration file that specifies parameters for the segmentation process. The 
		path must be valid.
	threads : int, optional
		The number of threads to use for processing. Using more than one thread can speed up segmentation on 
		systems with multiple CPU cores (default is 1).

	Raises
	------
	AssertionError
		If either the `pos` or `config` paths do not exist.

	Notes
	-----
	- The external segmentation script (`segment_cells_thresholds.py`) is expected to be located in a specific 
	  directory relative to this function.
	- The segmentation process and its parameters, including modes and thread usage, are defined by the external 
	  script and the configuration file.

	Examples
	--------
	>>> pos = '/path/to/position'
	>>> mode = 'default'
	>>> config = '/path/to/config.json'
	>>> segment_from_threshold_at_position(pos, mode, config, threads=2)
	# This will execute the segmentation script on the specified position directory with the given mode and 
	# configuration, utilizing 2 threads.
	
	"""


	pos = pos.replace('\\','/')
	pos = rf"{pos}"
	assert os.path.exists(pos),f'Position {pos} is not a valid path.'

	config = config.replace('\\','/')
	config = rf"{config}"
	assert os.path.exists(config),f'Config {config} is not a valid path.'

	script_path = os.sep.join([abs_path, 'scripts', 'segment_cells_thresholds.py'])
	cmd = f'python "{script_path}" --pos "{pos}" --config "{config}" --mode "{mode}" --threads "{threads}"'
	subprocess.call(cmd, shell=True)


def train_segmentation_model(config, use_gpu=True):

	"""
	Trains a segmentation model based on a specified configuration file.

	This function initiates the training of a segmentation model by calling an external Python script, 
	which reads the training parameters and dataset information from a given JSON configuration file. 
	The training process, including model architecture, training data, and hyperparameters, is defined 
	by the contents of the configuration file.

	Parameters
	----------
	config : str
		The file path to the JSON configuration file that specifies training parameters and dataset 
		information for the segmentation model. The path must be valid.

	Raises
	------
	AssertionError
		If the `config` path does not exist.

	Notes
	-----
	- The external training script (`train_segmentation_model.py`) is assumed to be located in a specific 
	  directory relative to this function.
	- The segmentation model and training process are highly dependent on the details specified in the 
	  configuration file, including the model architecture, loss functions, optimizer settings, and 
	  training/validation data paths.

	Examples
	--------
	>>> config = '/path/to/training_config.json'
	>>> train_segmentation_model(config)
	# Initiates the training of a segmentation model using the parameters specified in the given configuration file.
	
	"""

	config = config.replace('\\','/')
	config = rf"{config}"
	assert os.path.exists(config),f'Config {config} is not a valid path.'

	script_path = os.sep.join([abs_path, 'scripts', 'train_segmentation_model.py'])
	cmd = f'python "{script_path}" --config "{config}" --use_gpu "{use_gpu}"'
	subprocess.call(cmd, shell=True)


def merge_instance_segmentation(labels, iou_matching_threshold=0.05, mode='OR'):

	label_reference = labels[0]
	for i in range(1,len(labels)):

		label_to_merge = labels[i]
		pairs = matching(label_reference,label_to_merge, thresh=0.5, criterion='iou', report_matches=True).matched_pairs
		scores = matching(label_reference,label_to_merge, thresh=0.5, criterion='iou', report_matches=True).matched_scores
		
		accepted_pairs = []
		for k,p in enumerate(pairs):
			s = scores[k]
			if s > iou_matching_threshold:
				accepted_pairs.append(p)
		
		merge = np.copy(label_reference)

		for p in accepted_pairs:
			merge[np.where(merge==p[0])] = 0.
			cdt1 = label_reference==p[0]
			cdt2 = label_to_merge==p[1]
			if mode=='OR':
				cdt = np.logical_or(cdt1, cdt2)
			elif mode=='AND':
				cdt = np.logical_and(cdt1, cdt2)
			elif mode=='XOR':
				cdt = np.logical_xor(cdt1,cdt2)
			loc_i, loc_j = np.where(cdt)
			merge[loc_i, loc_j] = p[0]

		cells_to_ignore = [p[1] for p in accepted_pairs]
		for c in cells_to_ignore:
			label_to_merge[label_to_merge==c] = 0

		label_to_merge[label_to_merge!=0] = label_to_merge[label_to_merge!=0] + int(np.amax(label_reference))
		merge[label_to_merge!=0] = label_to_merge[label_to_merge!=0]

		label_reference = merge

	merge = auto_correct_masks(merge)

	return merge


if __name__ == "__main__":
	print(segment(None,'test'))
