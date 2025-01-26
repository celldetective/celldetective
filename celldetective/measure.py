import math

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy import ndimage
from tqdm import tqdm
#from skimage.measure import regionprops_table
from functools import reduce
from mahotas.features import haralick
from scipy.ndimage import zoom
import os
import subprocess
from math import ceil

from skimage.draw import disk as dsk
from skimage.feature import blob_dog, blob_log

from celldetective.utils import rename_intensity_column, create_patch_mask, remove_redundant_features, \
	remove_trajectory_measurements, contour_of_instance_segmentation, extract_cols_from_query, step_function, interpolate_nan, _remove_invalid_cols
from celldetective.preprocessing import field_correction
from celldetective.extra_properties import *
from inspect import getmembers, isfunction
from skimage.morphology import disk
from scipy.signal import find_peaks, peak_widths

from celldetective.segmentation import filter_image
from celldetective.regionprops import regionprops_table

abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'celldetective'])



def measure(stack=None, labels=None, trajectories=None, channel_names=None,
			features=None, intensity_measurement_radii=None, isotropic_operations=['mean'], border_distances=None,
			haralick_options=None, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}, clear_previous=False):

	"""

	Perform measurements on a stack of images or labels.

	Parameters
	----------
	stack : numpy array, optional
		Stack of images with shape (T, Y, X, C), where T is the number of frames, Y and X are the spatial dimensions,
		and C is the number of channels. Default is None.
	labels : numpy array, optional
		Label stack with shape (T, Y, X) representing cell segmentations. Default is None.
	trajectories : pandas DataFrame, optional
		DataFrame of cell trajectories with columns specified in `column_labels`. Default is None.
	channel_names : list, optional
		List of channel names corresponding to the image stack. Default is None.
	features : list, optional
		List of features to measure using the `measure_features` function. Default is None.
	intensity_measurement_radii : int, float, or list, optional
		Radius or list of radii specifying the size of the isotropic measurement area for intensity measurements.
		If a single value is provided, a circular measurement area is used. If a list of values is provided, multiple
		measurements are performed using ring-shaped measurement areas. Default is None.
	isotropic_operations : list, optional
		List of operations to perform on the isotropic intensity values. Default is ['mean'].
	border_distances : int, float, or list, optional
		Distance or list of distances specifying the size of the border region for intensity measurements.
		If a single value is provided, measurements are performed at a fixed distance from the cell borders.
		If a list of values is provided, measurements are performed at multiple border distances. Default is None.
	haralick_options : dict, optional
		Dictionary of options for Haralick feature measurements. Default is None.
	column_labels : dict, optional
		Dictionary containing the column labels for the DataFrame. Default is {'track': "TRACK_ID",
		'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	pandas DataFrame
		DataFrame containing the measured features and intensities.

	Notes
	-----
	This function performs measurements on a stack of images or labels. If both `stack` and `labels` are provided,
	measurements are performed on each frame of the stack. The measurements include isotropic intensity values, computed
	using the `measure_isotropic_intensity` function, and additional features, computed using the `measure_features` function.
	The intensity measurements are performed at the positions specified in the `trajectories` DataFrame, using the
	specified `intensity_measurement_radii` and `border_distances`. The resulting measurements are combined into a single
	DataFrame and returned.

	Examples
	--------
	>>> stack = np.random.rand(10, 100, 100, 3)
	>>> labels = np.random.randint(0, 2, (10, 100, 100))
	>>> trajectories = pd.DataFrame({'TRACK_ID': [1, 2, 3], 'FRAME': [1, 1, 1],
	...							'POSITION_X': [10, 20, 30], 'POSITION_Y': [15, 25, 35]})
	>>> channel_names = ['channel1', 'channel2', 'channel3']
	>>> features = ['area', 'intensity_mean']
	>>> intensity_measurement_radii = [5, 10]
	>>> border_distances = 2
	>>> measurements = measure(stack=stack, labels=labels, trajectories=trajectories, channel_names=channel_names,
	...							features=features, intensity_measurement_radii=intensity_measurement_radii,
	...							border_distances=border_distances)
	# Perform measurements on the stack, labels, and trajectories, computing isotropic intensities and additional features.

	"""


	do_iso_intensities = True
	do_features = True


	# Check that conditions are satisfied to perform measurements
	assert (labels is not None) or (stack is not None),'Please pass a stack and/or labels... Abort.'
	if (labels is not None)*(stack is not None):
		assert labels.shape==stack.shape[:-1],f"Shape mismatch between the stack of shape {stack.shape} and the segmentation {labels.shape}..."

	# Condition to compute features
	if labels is None:
		do_features = False
		nbr_frames = len(stack)
		print('No labels were provided... Features will not be computed...')
	else:
		nbr_frames = len(labels)

	# Condition to compute isotropic intensities
	if (stack is None) or (trajectories is None) or (intensity_measurement_radii is None):
		do_iso_intensities = False
		print('Either no image, no positions or no radii were provided... Isotropic intensities will not be computed...')

	# Compensate for non provided channel names
	if (stack is not None)*(channel_names is None):
		nbr_channels = stack.shape[-1]
		channel_names = [f'intensity-{k}' for k in range(nbr_channels)]

	if isinstance(intensity_measurement_radii, int) or isinstance(intensity_measurement_radii, float):
		intensity_measurement_radii = [intensity_measurement_radii]

	if isinstance(border_distances, int) or isinstance(border_distances, float):
		border_distances = [border_distances]

	if features is not None:
		features = remove_redundant_features(features, trajectories.columns,
											channel_names=channel_names)

	if features is None:
		features = []

	# Prep for the case where no trajectory is provided but still want to measure isotropic intensities...
	if (trajectories is None):
		do_features = True
		features += ['centroid']
	else:
		if clear_previous:
			trajectories = remove_trajectory_measurements(trajectories, column_labels)

	timestep_dataframes = []

	for t in tqdm(range(nbr_frames),desc='frame'):

		if stack is not None:
			img = stack[t]
		else:
			img = None
		if labels is not None:
			lbl = labels[t]
		else:
			lbl = None

		if trajectories is not None:
			positions_at_t = trajectories.loc[trajectories[column_labels['time']]==t].copy()

		if do_features:
			feature_table = measure_features(img, lbl, features = features, border_dist=border_distances,
											channels=channel_names, haralick_options=haralick_options, verbose=False)
			if trajectories is None:
				# Use the centroids as estimate for the location of the cells, to be passed to the measure_isotropic_intensity function.
				positions_at_t = feature_table[['centroid-1', 'centroid-0','class_id']].copy()
				positions_at_t['ID'] = np.arange(len(positions_at_t))	# temporary ID for the cells, that will be reset at the end since they are not tracked
				positions_at_t.rename(columns={'centroid-1': 'POSITION_X', 'centroid-0': 'POSITION_Y'},inplace=True)
				positions_at_t['FRAME'] = int(t)
				column_labels = {'track': "ID", 'time': column_labels['time'], 'x': column_labels['x'], 'y': column_labels['y']}

		center_of_mass_x_cols = [c for c in list(positions_at_t.columns) if c.endswith('centre_of_mass_x')]
		center_of_mass_y_cols = [c for c in list(positions_at_t.columns) if c.endswith('centre_of_mass_y')]
		for c in center_of_mass_x_cols:
			positions_at_t.loc[:,c.replace('_x','_POSITION_X')] = positions_at_t[c] + positions_at_t['POSITION_X']
		for c in center_of_mass_y_cols:
			positions_at_t.loc[:,c.replace('_y','_POSITION_Y')] = positions_at_t[c] + positions_at_t['POSITION_Y']
		positions_at_t = positions_at_t.drop(columns = center_of_mass_x_cols+center_of_mass_y_cols)

		# Isotropic measurements (circle, ring)
		if do_iso_intensities:
			iso_table = measure_isotropic_intensity(positions_at_t, img, channels=channel_names, intensity_measurement_radii=intensity_measurement_radii,
													column_labels=column_labels, operations=isotropic_operations, verbose=False)

		if do_iso_intensities*do_features:
			measurements_at_t = iso_table.merge(feature_table, how='outer', on='class_id')
		elif do_iso_intensities*(not do_features):
			measurements_at_t = iso_table
		elif do_features*(trajectories is not None):
			measurements_at_t = positions_at_t.merge(feature_table, how='outer', on='class_id')
		elif do_features*(trajectories is None):
			measurements_at_t = positions_at_t

		try:
			measurements_at_t['radial_distance'] = np.sqrt((measurements_at_t[column_labels['x']] - img.shape[0] / 2) ** 2 + (
					measurements_at_t[column_labels['y']] - img.shape[1] / 2) ** 2)
		except Exception as e:
			print(f"{e=}")

		timestep_dataframes.append(measurements_at_t)

	measurements = pd.concat(timestep_dataframes)
	if trajectories is not None:
		measurements = measurements.sort_values(by=[column_labels['track'],column_labels['time']])
		measurements = measurements.dropna(subset=[column_labels['track']])
	else:
		measurements['ID'] = np.arange(len(df))

	measurements = measurements.reset_index(drop=True)
	measurements = _remove_invalid_cols(measurements)

	return measurements

def write_first_detection_class(tab, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):

	tab = tab.sort_values(by=[column_labels['track'],column_labels['time']])
	if 'area' in tab.columns:
		for tid,track_group in tab.groupby(column_labels['track']):
			indices = track_group.index
			area = track_group['area'].values
			timeline = track_group[column_labels['time']].values
			if np.any(area==area):
				t_first = timeline[area==area][0]
				cclass = 1
				if t_first==0:
					t_first = 0
					cclass = 2
			else:
				t_first = -1
				cclass = 2

			tab.loc[indices, 'class_firstdetection'] = cclass
			tab.loc[indices, 't_firstdetection'] = t_first
	return tab


def drop_tonal_features(features):

	"""
	Removes features related to intensity from a list of feature names.

	This function iterates over a list of feature names and removes any feature that includes the term 'intensity' in its name.
	The operation is performed in-place, meaning the original list of features is modified directly.

	Parameters
	----------
	features : list of str
		A list of feature names from which intensity-related features are to be removed.

	Returns
	-------
	list of str
		The modified list of feature names with intensity-related features removed. Note that this operation modifies the
		input list in-place, so the return value is the same list object with some elements removed.

	"""

	feat2 = features[:]
	for f in features:
		if 'intensity' in f:
			feat2.remove(f)
	return feat2

def measure_features(img, label, features=['area', 'intensity_mean'], channels=None,
					 border_dist=None, haralick_options=None, verbose=True, normalisation_list=None,
					 radial_intensity=None,
					 radial_channel=None, spot_detection=None):
	"""

	Measure features within segmented regions of an image.

	Parameters
	----------
	img : ndarray
		The input image as a NumPy array.
	label : ndarray
		The segmentation labels corresponding to the image regions.
	features : list, optional
		The list of features to measure within the segmented regions. The default is ['area', 'intensity_mean'].
	channels : list, optional
		The list of channel names in the image. The default is ["brightfield_channel", "dead_nuclei_channel", "live_nuclei_channel"].
	border_dist : int, float, or list, optional
		The distance(s) in pixels from the edge of each segmented region to measure features. The default is None.
	haralick_options : dict, optional
		The options for computing Haralick features. The default is None.

	Returns
	-------
	df_props : DataFrame
		A pandas DataFrame containing the measured features for each segmented region.

	Notes
	-----
	This function measures features within segmented regions of an image.
	It utilizes the regionprops_table function from the skimage.measure module for feature extraction.
	The features to measure can be specified using the 'features' parameter.
	Optional parameters such as 'channels' and 'border_dist' allow for additional measurements.
	If provided, Haralick features can be computed using the 'haralick_options' parameter.
	The results are returned as a pandas DataFrame.

	Examples
	--------
	>>> df_props = measure_features(img, label, features=['area', 'intensity_mean'], channels=["brightfield_channel", "dead_nuclei_channel", "live_nuclei_channel"])
	# Measure area and mean intensity within segmented regions of the image.

	"""

	if isinstance(features, list):
		features = features.copy()

	if features is None:
		features = []

	measure_mean_intensities = False
	if img is None:
		if verbose:
			print('No image was provided... Skip intensity measurements.')
		border_dist = None;
		haralick_options = None;
		features = drop_tonal_features(features)

	if 'intensity_mean' in features:
		measure_mean_intensities = True
		features.remove('intensity_mean')

	# Add label to have identity of mask
	if 'label' not in features:
		features.append('label')

	if img is not None:
		if img.ndim == 2:
			img = img[:, :, np.newaxis]
		if channels is None:
			channels = [f'intensity-{k}' for k in range(img.shape[-1])]
		if (channels is not None) * (img.ndim == 3):
			assert len(channels) == img.shape[
				-1], "Mismatch between the provided channel names and the shape of the image"

		if spot_detection is not None:
			for index, channel in enumerate(channels):
				if channel == spot_detection['channel']:
					ind = index
					if "image_preprocessing" not in spot_detection:
						spot_detection.update({'image_preprocessing': None})
					df_spots = blob_detection(img, label, diameter=spot_detection['diameter'],threshold=spot_detection['threshold'], channel_name=spot_detection['channel'], target_channel=ind, image_preprocessing=spot_detection['image_preprocessing'])
		
		if normalisation_list:
			for norm in normalisation_list:
				for index, channel in enumerate(channels):
					if channel == norm['target_channel']:
						ind = index
				if norm['correction_type'] == 'local':
					normalised_image = normalise_by_cell(img[:, :, ind].copy(), label,
														 distance=int(norm['distance']), model=norm['model'],
														 operation=norm['operation'], clip=norm['clip'])
					img[:, :, ind] = normalised_image
				else:
					corrected_image = field_correction(img[:,:,ind].copy(), threshold_on_std=norm['threshold_on_std'], operation=norm['operation'], model=norm['model'], clip=norm['clip'])
					img[:, :, ind] = corrected_image
	
	import celldetective.extra_properties as extra_props

	extra = getmembers(extra_props, isfunction)
	extra = [extra[i][0] for i in range(len(extra))]

	extra_props_list = []
	feats = features.copy()
	for f in features:
		if f in extra:
			feats.remove(f)
			extra_props_list.append(getattr(extra_props, f))

	# Add intensity nan mean if need to measure mean intensities
	if measure_mean_intensities:
		extra_props_list.append(getattr(extra_props, 'intensity_nanmean'))

	if len(extra_props_list) == 0:
		extra_props_list = None
	else:
		extra_props_list = tuple(extra_props_list)

	props = regionprops_table(label, intensity_image=img, properties=feats, extra_properties=extra_props_list, channel_names=channels)
	df_props = pd.DataFrame(props)
	if spot_detection is not None:
		if df_spots is not None:
			df_props = df_props.merge(df_spots, how='outer', on='label',suffixes=('_delme', ''))
			df_props = df_props[[c for c in df_props.columns if not c.endswith('_delme')]]

	if border_dist is not None:
		# automatically drop all non intensity features
		intensity_features_test = [('intensity' in s and 'centroid' not in s and 'peripheral' not in s) for s in
								   features]
		intensity_features = list(np.array(features)[np.array(intensity_features_test)])
		intensity_extra = []
		for s in intensity_features:
			if s in extra:
				intensity_extra.append(getattr(extra_props, s))
				intensity_features.remove(s)

		if len(intensity_features) == 0:
			if verbose:
				print('No intensity feature was passed... Adding mean intensity for edge measurement...')
			intensity_features = np.append(intensity_features, 'intensity_mean')
		intensity_features = list(np.append(intensity_features, 'label'))

		new_intensity_features = intensity_features.copy()
		for int_feat in intensity_features:
			if int_feat in extra:
				new_intensity_features.remove(int_feat)
		intensity_features = new_intensity_features

		if (isinstance(border_dist, int) or isinstance(border_dist, float)):
			border_label = contour_of_instance_segmentation(label, border_dist)
			props_border = regionprops_table(border_label, intensity_image=img, properties=intensity_features, channel_names=channels)
			df_props_border = pd.DataFrame(props_border)
			for c in df_props_border.columns:
				if 'intensity' in c:
					df_props_border = df_props_border.rename({c: c+f'_edge_{border_dist}px'},axis=1)

		if isinstance(border_dist, list):
			df_props_border_list = []
			for d in border_dist:
				border_label = contour_of_instance_segmentation(label, d)
				props_border = regionprops_table(border_label, intensity_image=img, properties=intensity_features, channel_names=channels)
				df_props_border_d = pd.DataFrame(props_border)
				for c in df_props_border_d.columns:
					if 'intensity' in c:
						if '-' in str(d):
							df_props_border_d = df_props_border_d.rename({c: c + f'_outer_edge_{d}px'}, axis=1)
						else:
							df_props_border_d = df_props_border_d.rename({c: c + f'_edge_{d}px'}, axis=1)
				df_props_border_list.append(df_props_border_d)

			df_props_border = reduce(lambda  left,right: pd.merge(left,right,on=['label'],
											how='outer'), df_props_border_list)

		df_props = df_props.merge(df_props_border, how='outer', on='label')

	if haralick_options is not None:
		try:
			df_haralick = compute_haralick_features(img, label, channels=channels, **haralick_options)
			if df_haralick is not None:
				df_haralick = df_haralick.rename(columns={"cell_id": "label"})
				df_props = df_props.merge(df_haralick, how='outer', on='label', suffixes=('_delme', ''))
				df_props = df_props[[c for c in df_props.columns if not c.endswith('_delme')]]
		except Exception as e:
			print(e)
			pass

	if channels is not None:
		df_props = rename_intensity_column(df_props, channels)
	df_props.rename(columns={"label": "class_id"},inplace=True)
	df_props['class_id'] = df_props['class_id'].astype(float)

	return df_props

def compute_haralick_features(img, labels, channels=None, target_channel=0, scale_factor=1, percentiles=(0.01,99.99), clip_values=None,
								n_intensity_bins=256, ignore_zero=True, return_mean=True, return_mean_ptp=False, distance=1, disable_progress_bar=False, return_norm_image_only=False, return_digit_image_only=False):

	"""

	Compute Haralick texture features on each segmented region of an image.

	Parameters
	----------
	img : ndarray
		The input image as a NumPy array.
	labels : ndarray
		The segmentation labels corresponding to the image regions.
	target_channel : int, optional
		The target channel index of the image. The default is 0.
	modality : str, optional
		The modality or channel type of the image. The default is 'brightfield_channel'.
	scale_factor : float, optional
		The scale factor for resampling the image and labels. The default is 1.
	percentiles : tuple of float, optional
		The percentiles to use for image normalization. The default is (0.01, 99.99).
	clip_values : tuple of float, optional
		The minimum and maximum values to clip the image. If None, percentiles are used. The default is None.
	n_intensity_bins : int, optional
		The number of intensity bins for image normalization. The default is 255.
	ignore_zero : bool, optional
		Flag indicating whether to ignore zero values during feature computation. The default is True.
	return_mean : bool, optional
		Flag indicating whether to return the mean value of each Haralick feature. The default is True.
	return_mean_ptp : bool, optional
		Flag indicating whether to return the mean and peak-to-peak values of each Haralick feature. The default is False.
	distance : int, optional
		The distance parameter for Haralick feature computation. The default is 1.

	Returns
	-------
	features : DataFrame
		A pandas DataFrame containing the computed Haralick features for each segmented region.

	Notes
	-----
	This function computes Haralick features on an image within segmented regions.
	It uses the mahotas library for feature extraction and pandas DataFrame for storage.
	The image is rescaled, normalized and digitized based on the specified parameters.
	Haralick features are computed for each segmented region, and the results are returned as a DataFrame.

	Examples
	--------
	>>> features = compute_haralick_features(img, labels, target_channel=0, modality="brightfield_channel")
	# Compute Haralick features on the image within segmented regions.

	"""

	assert ((img.ndim==2)|(img.ndim==3)),f'Invalid image shape to compute the Haralick features. Expected YXC, got {img.shape}...'
	assert img.shape[:2]==labels.shape,f'Mismatch between image shape {img.shape} and labels shape {labels.shape}'

	if img.ndim==2:
		img = img[:,:,np.newaxis]
		target_channel = 0
		if isinstance(channels, list):
			modality = channels[0]
		elif isinstance(channels, str):
			modality = channels
		else:
			print('Channel name unrecognized...')
			modality=''
	elif img.ndim==3:
		assert target_channel is not None,"The image is multichannel. Please provide a target channel to compute the Haralick features. Abort."
		modality = channels[target_channel]

	haralick_labels = ["angular_second_moment",
					   "contrast",
					   "correlation",
					   "sum_of_square_variance",
					   "inverse_difference_moment",
					   "sum_average",
					   "sum_variance",
					   "sum_entropy",
					   "entropy",
					   "difference_variance",
					   "difference_entropy",
					   "information_measure_of_correlation_1",
					   "information_measure_of_correlation_2",
					   "maximal_correlation_coefficient"]

	haralick_labels = ['haralick_'+h+"_"+modality for h in haralick_labels]
	if len(img.shape)==3:
		img = img[:,:,target_channel]

	# Routine to skip black frames
	if np.percentile(img.flatten(),99.9)==0.0:
		return None

	img = interpolate_nan(img)

	# Rescale image and mask
	img = zoom(img,[scale_factor,scale_factor],order=3).astype(float)
	labels = zoom(labels, [scale_factor,scale_factor],order=0)

	# Normalize image
	if clip_values is None:
		min_value = np.nanpercentile(img[img!=0.].flatten(), percentiles[0])
		max_value = np.nanpercentile(img[img!=0.].flatten(), percentiles[1])
	else:
		min_value = clip_values[0]; max_value = clip_values[1]

	img -= min_value
	img /= (max_value-min_value) / n_intensity_bins
	img[img<=0.] = 0.
	img[img>=n_intensity_bins] = n_intensity_bins

	if return_norm_image_only:
		return img

	hist,bins = np.histogram(img.flatten(),bins=n_intensity_bins)
	centered_bins = [bins[0]] + [bins[i] + (bins[i+1] - bins[i])/2. for i in range(len(bins)-1)]

	digitized = np.digitize(img, bins)
	img_binned = np.zeros_like(img)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			img_binned[i,j] = centered_bins[digitized[i,j] - 1]

	img = img_binned.astype(int)
	if return_digit_image_only:
		return img

	haralick_properties = []

	for cell in tqdm(np.unique(labels)[1:],disable=disable_progress_bar):

		mask = labels==cell
		f = img*mask
		features = haralick(f, ignore_zeros=ignore_zero,return_mean=return_mean,distance=distance)

		dictionary = {'cell_id': cell}
		for k in range(len(features)):
			dictionary.update({haralick_labels[k]: features[k]})
		haralick_properties.append(dictionary)

	assert len(haralick_properties)==(len(np.unique(labels))-1),'Some cells have not been measured...'

	return pd.DataFrame(haralick_properties)


def measure_isotropic_intensity(positions, # Dataframe of cell positions @ t
								 img,  # multichannel frame (YXC) @ t
								 channels=None, #channels, need labels to name measurements
								 intensity_measurement_radii=None, #list of radii, single value is circle, tuple is ring?
								 operations = ['mean'],
								 measurement_kernel = None,
								 pbar=None,
								 column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'},
								 verbose=True,
								):

	"""

	Measure isotropic intensity values around cell positions in an image.

	Parameters
	----------
	positions : pandas DataFrame
		DataFrame of cell positions at time 't' containing columns specified in `column_labels`.
	img : numpy array
		Multichannel frame (YXC) at time 't' used for intensity measurement.
	channels : list or str, optional
		List of channel names corresponding to the image channels. Default is None.
	intensity_measurement_radii : int, list, or tuple
		Radius or list of radii specifying the size of the isotropic measurement area.
		If a single value is provided, a circular measurement area is used. If a list or tuple of two values
		is provided, a ring-shaped measurement area is used. Default is None.
	operations : list, optional
		List of operations to perform on the intensity values. Default is ['mean'].
	measurement_kernel : numpy array, optional
		Kernel used for intensity measurement. If None, a circular or ring-shaped kernel is generated
		based on the provided `intensity_measurement_radii`. Default is None.
	pbar : tqdm progress bar, optional
		Progress bar for tracking the measurement process. Default is None.
	column_labels : dict, optional
		Dictionary containing the column labels for the DataFrame. Default is {'track': "TRACK_ID",
		'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.
	verbose : bool, optional
		If True, enables verbose output. Default is True.

	Returns
	-------
	pandas DataFrame
		The updated DataFrame `positions` with additional columns representing the measured intensity values.

	Notes
	-----
	This function measures the isotropic intensity values around the cell positions specified in the `positions`
	DataFrame using the provided image `img`. The intensity measurements are performed using circular or ring-shaped
	measurement areas defined by the `intensity_measurement_radii`. The measurements are calculated for each channel
	specified in the `channels` list. The resulting intensity values are stored in additional columns of the `positions`
	DataFrame. The `operations` parameter allows specifying different operations to be performed on the intensity
	values, such as 'mean', 'median', etc. The measurement kernel can be customized by providing the `measurement_kernel`
	parameter. If not provided, the measurement kernel is automatically generated based on the `intensity_measurement_radii`.
	The progress bar `pbar` can be used to track the measurement process. The `column_labels` dictionary is used to
	specify the column labels for the DataFrame.

	Examples
	--------
	>>> positions = pd.DataFrame({'TRACK_ID': [1, 2, 3], 'FRAME': [1, 1, 1],
	...							'POSITION_X': [10, 20, 30], 'POSITION_Y': [15, 25, 35]})
	>>> img = np.random.rand(100, 100, 3)
	>>> channels = ['channel1', 'channel2', 'channel3']
	>>> intensity_measurement_radii = 5
	>>> positions = measure_isotropic_intensity(positions, img, channels=channels,
	...											intensity_measurement_radii=intensity_measurement_radii)
	# Measure isotropic intensity values around cell positions in the image.

	"""

	epsilon = -10000
	assert ((img.ndim==2)|(img.ndim==3)),f'Invalid image shape to compute the Haralick features. Expected YXC, got {img.shape}...'

	if img.ndim==2:
		img = img[:,:,np.newaxis]
		if isinstance(channels, str):
			channels = [channels]
		else:
			if verbose:
				print('Channel name unrecognized...')
			channels=['intensity']
	elif img.ndim==3:
		assert channels is not None,"The image is multichannel. Please provide the list of channel names. Abort."

	if isinstance(intensity_measurement_radii, int) or isinstance(intensity_measurement_radii, float):
		intensity_measurement_radii = [intensity_measurement_radii]

	if (measurement_kernel is None)*(intensity_measurement_radii is not None):

		for r in intensity_measurement_radii:

			if isinstance(r,list):
				mask = create_patch_mask(2*max(r)+1,2*max(r)+1,((2*max(r))//2,(2*max(r))//2),radius=r)
			else:
				mask = create_patch_mask(2*r+1,2*r+1,((2*r)//2,(2*r)//2),r)

			pad_value_x = mask.shape[0]//2 + 1
			pad_value_y = mask.shape[1]//2 + 1
			frame_padded = np.pad(img.astype(float), [(pad_value_x,pad_value_x),(pad_value_y,pad_value_y),(0,0)], constant_values=[(epsilon,epsilon),(epsilon,epsilon),(0,0)])

			# Find a way to measure intensity in mask
			for tid,group in positions.groupby(column_labels['track']):

				x = group[column_labels['x']].to_numpy()[0]
				y = group[column_labels['y']].to_numpy()[0]

				xmin = int(x)
				xmax = int(x) + 2*pad_value_y - 1
				ymin = int(y)
				ymax = int(y) + 2*pad_value_x - 1

				assert frame_padded[ymin:ymax,xmin:xmax,0].shape == mask.shape,"Shape mismatch between the measurement kernel and the image..."

				expanded_mask = np.expand_dims(mask, axis=-1)  # shape: (X, Y, 1)
				crop = frame_padded[ymin:ymax,xmin:xmax]

				crop_temp = crop.copy()
				crop_temp[crop_temp==epsilon] = 0.
				projection = np.multiply(crop_temp, expanded_mask)

				projection[crop==epsilon] = epsilon
				projection[expanded_mask[:,:,0]==0.,:] = epsilon

				for op in operations:
					func = eval('np.'+op)
					intensity_values = func(projection, axis=(0,1), where=projection>epsilon)
					for k in range(crop.shape[-1]):
						if isinstance(r,list):
							positions.loc[group.index, f'{channels[k]}_ring_{min(r)}_{max(r)}_{op}'] = intensity_values[k]
						else:
							positions.loc[group.index, f'{channels[k]}_circle_{r}_{op}'] = intensity_values[k]

	elif (measurement_kernel is not None):
		# do something like this
		mask = measurement_kernel
		pad_value_x = mask.shape[0]//2 + 1
		pad_value_y = mask.shape[1]//2 + 1
		frame_padded = np.pad(img, [(pad_value_x,pad_value_x),(pad_value_y,pad_value_y),(0,0)])

		for tid,group in positions.groupby(column_labels['track']):

			x = group[column_labels['x']].to_numpy()[0]
			y = group[column_labels['y']].to_numpy()[0]

			xmin = int(x)
			xmax = int(x) + 2*pad_value_y - 1
			ymin = int(y)
			ymax = int(y) + 2*pad_value_x - 1

			assert frame_padded[ymin:ymax,xmin:xmax,0].shape == mask.shape,"Shape mismatch between the measurement kernel and the image..."

			expanded_mask = np.expand_dims(mask, axis=-1)  # shape: (X, Y, 1)
			crop = frame_padded[ymin:ymax,xmin:xmax]
			projection = np.multiply(crop, expanded_mask)

			for op in operations:
				func = eval('np.'+op)
				intensity_values = func(projection, axis=(0,1), where=projection==projection)
				for k in range(crop.shape[-1]):
					positions.loc[group.index, f'{channels[k]}_custom_kernel_{op}'] = intensity_values[k]

	if pbar is not None:
		pbar.update(1)
	positions['class_id'] = positions['class_id'].astype(float)
	return positions

def measure_at_position(pos, mode, return_measurements=False, threads=1):

	"""
	Executes a measurement script at a specified position directory, optionally returning the measured data.

	This function calls an external Python script to perform measurements on data
	located in a specified position directory. The measurement mode determines the type of analysis performed by the script.
	The function can either return the path to the resulting measurements table or load and return the measurements as a
	pandas DataFrame.

	Parameters
	----------
	pos : str
		The path to the position directory where the measurements should be performed. The path should be a valid directory.
	mode : str
		The measurement mode to be used by the script. This determines the type of analysis performed (e.g., 'tracking',
		'feature_extraction').
	return_measurements : bool, optional
		If True, the function loads the resulting measurements from a CSV file into a pandas DataFrame and returns it. If
		False, the function returns None (default is False).

	Returns
	-------
	pandas.DataFrame or None
		If `return_measurements` is True, returns a pandas DataFrame containing the measurements. Otherwise, returns None.

	"""

	pos = pos.replace('\\','/')
	pos = rf"{pos}"
	assert os.path.exists(pos),f'Position {pos} is not a valid path.'
	if not pos.endswith('/'):
		pos += '/'
	script_path = os.sep.join([abs_path, 'scripts', 'measure_cells.py'])
	cmd = f'python "{script_path}" --pos "{pos}" --mode "{mode}" --threads "{threads}"'
	subprocess.call(cmd, shell=True)

	table = pos + os.sep.join(["output","tables",f"trajectories_{mode}.csv"])
	if return_measurements:
		df = pd.read_csv(table)
		return df
	else:
		return None


def local_normalisation(image, labels, background_intensity, measurement='intensity_median', operation='subtract', clip=False):
	
	
	for index, cell in enumerate(np.unique(labels)):
		if cell == 0:
			continue
		if operation == 'subtract':
			image[np.where(labels == cell)] = image[np.where(labels == cell)].astype(float) - \
											  background_intensity[measurement][index-1].astype(float)
		elif operation == 'divide':
			image[np.where(labels == cell)] = image[np.where(labels == cell)].astype(float) / \
											  background_intensity[measurement][index-1].astype(float)
	if clip:
		image[image<=0.] = 0.

	return image.astype(float)


def normalise_by_cell(image, labels, distance=5, model='median', operation='subtract', clip=False):

	import celldetective.extra_properties as extra_props

	border = contour_of_instance_segmentation(label=labels, distance=distance * (-1))
	if model == 'mean':
		measurement = 'intensity_nanmean'
		extra_props = [getattr(extra_props, measurement)]
		background_intensity = regionprops_table(intensity_image=image, label_image=border,
												 extra_properties=extra_props)
	elif model == 'median':
		measurement = 'intensity_median'
		extra_props = [getattr(extra_props, measurement)]
		background_intensity = regionprops_table(intensity_image=image, label_image=border,
												 extra_properties=extra_props)

	normalised_frame = local_normalisation(image=image.astype(float).copy(),
										   labels=labels, background_intensity=background_intensity, measurement=measurement,
										   operation=operation, clip=clip)

	return normalised_frame


def extract_blobs_in_image(image, label, diameter, threshold=0., method="log", image_preprocessing=None):
	
	if np.percentile(image.flatten(),99.9)==0.0:
		return None

	if isinstance(image_preprocessing, (list, np.ndarray)):
		image = filter_image(image.copy(),filters=image_preprocessing) # apply prefiltering to images before spot detection

	dilated_image = ndimage.grey_dilation(label, footprint=disk(int(1.2*diameter))) # dilation larger than spot diameter to be safe

	masked_image = image.copy()
	masked_image[np.where((dilated_image == 0)|(image!=image))] = 0
	min_sigma = (1 / (1 + math.sqrt(2))) * diameter
	max_sigma = math.sqrt(2) * min_sigma
	if method=="dog":
		blobs = blob_dog(masked_image, threshold=threshold, min_sigma=min_sigma, max_sigma=max_sigma, overlap=0.75)
	elif method=="log":
		blobs = blob_log(masked_image, threshold=threshold, min_sigma=min_sigma, max_sigma=max_sigma, overlap=0.75)

	# Exclude spots outside of cell masks
	mask = np.array([label[int(y), int(x)] != 0 for y, x, _ in blobs])
	if np.any(mask):
		blobs_filtered = blobs[mask]
	else:
		blobs_filtered=[]

	return blobs_filtered


def blob_detection(image, label, diameter, threshold=0., channel_name=None, target_channel=0, method="log", image_preprocessing=None):
	

	image = image[:, :, target_channel].copy()
	if np.percentile(image.flatten(),99.9)==0.0:
		return None

	detections = []
	blobs_filtered = extract_blobs_in_image(image, label, diameter, method=method, threshold=threshold, image_preprocessing=image_preprocessing)

	for lbl in np.unique(label):
		if lbl>0:
			
			blob_selection = np.array([label[int(y), int(x)] == lbl for y, x, _ in blobs_filtered])
			if np.any(blob_selection):
				# if any spot
				blobs_in_cell = blobs_filtered[blob_selection]
				n_spots = len(blobs_in_cell)
				binary_blobs = np.zeros_like(label)
				for blob in blobs_in_cell:
					y, x, sig = blob
					r = np.sqrt(2)*sig
					rr, cc = dsk((y, x), r, shape=binary_blobs.shape)
					binary_blobs[rr, cc] = 1    
				intensity_mean = np.nanmean(image[binary_blobs==1].flatten())
			else:
				n_spots = 0
				intensity_mean = np.nan
			detections.append({'label': lbl, f'{channel_name}_spot_count': n_spots, f'{channel_name}_mean_spot_intensity': intensity_mean})
	detections = pd.DataFrame(detections)

	return detections


def estimate_time(df, class_attr, model='step_function', class_of_interest=[2], r2_threshold=0.5):

	"""
	Estimate the timing of an event for cells based on classification status and fit a model to the observed status signal.

	Parameters
	----------
	df : pandas.DataFrame
		DataFrame containing tracked data with classification and status columns.
	class_attr : str
		Column name for the classification attribute (e.g., 'class_event').
	model : str, optional
		Name of the model function used to fit the status signal (default is 'step_function').
	class_of_interest : list, optional
		List of class values that define the cells of interest for analysis (default is [2]).
	r2_threshold : float, optional
		R-squared threshold for determining if the model fit is acceptable (default is 0.5).

	Returns
	-------
	pandas.DataFrame
		Updated DataFrame with estimated event timing added in a column replacing 'class' with 't', 
		and reclassification of cells based on the model fit.

	Notes
	-----
	- The function assumes that cells are grouped by a unique identifier ('TRACK_ID') and sorted by time ('FRAME').
	- If the model provides a poor fit (R² < r2_threshold), the class of interest is set to 2.0 and timing (-1).
	- The function supports different models that can be passed as the `model` parameter, which are evaluated using `eval()`.

	Example
	-------
	>>> df = estimate_time(df, 'class', model='step_function', class_of_interest=[2], r2_threshold=0.6)

	"""

	cols = list(df.columns)
	assert 'TRACK_ID' in cols,'Please provide tracked data...'
	if 'position' in cols:
		sort_cols = ['position', 'TRACK_ID']
	else:
		sort_cols = ['TRACK_ID']

	df = df.sort_values(by=sort_cols,ignore_index=True)
	df = df.reset_index(drop=True)
	max_time = df['FRAME'].max()


	for tid,group in df.loc[df[class_attr].isin(class_of_interest)].groupby(sort_cols):
		
		indices = group.index
		status_col = class_attr.replace('class','status')

		group_clean = group.dropna(subset=status_col)
		status_signal = group_clean[status_col].values
		if np.all(np.array(status_signal)==1):
			continue

		timeline = group_clean['FRAME'].values
		frames = group_clean['FRAME'].to_numpy()
		status_values = group_clean[status_col].to_numpy()
		t_first = group['t_firstdetection'].to_numpy()[0]

		try:
			popt, pcov = curve_fit(eval(model), timeline.astype(int), status_signal, p0=[max(timeline)//2, 0.8],maxfev=100000)
			values = [eval(model)(t, *popt) for t in timeline]
			r2 = r2_score(status_signal,values)
		except Exception:
			df.loc[indices, class_attr] = 2.0
			df.loc[indices, class_attr.replace('class','t')] = -1
			continue

		if r2 > float(r2_threshold):
			t0 = popt[0]
			if t0>=max_time:
				t0 = max_time - 1
			df.loc[indices, class_attr.replace('class','t')] = t0
			df.loc[indices, class_attr] = 0.0
		else:
			df.loc[indices, class_attr.replace('class','t')] = -1
			df.loc[indices, class_attr] = 2.0

	return df


def interpret_track_classification(df, class_attr, irreversible_event=False, unique_state=False, transient_event=False, r2_threshold=0.5, percentile_recovery=50, pre_event=None):

	"""
	Interpret and classify tracked cells based on their status signals.

	Parameters
	----------
	df : pandas.DataFrame
		DataFrame containing tracked cell data, including a classification attribute column and other necessary columns.
	class_attr : str
		Column name for the classification attribute (e.g., 'class') used to determine the state of cells.
	irreversible_event : bool, optional
		If True, classifies irreversible events in the dataset (default is False).
		When set to True, `unique_state` is ignored.
	unique_state : bool, optional
		If True, classifies unique states of cells in the dataset based on a percentile threshold (default is False).
		This option is ignored if `irreversible_event` is set to True.
	r2_threshold : float, optional
		R-squared threshold used when fitting the model during the classification of irreversible events (default is 0.5).

	Returns
	-------
	pandas.DataFrame
		DataFrame with updated classifications for cell trajectories:
		- If `irreversible_event` is True, it classifies irreversible events using the `classify_irreversible_events` function.
		- If `unique_state` is True, it classifies unique states using the `classify_unique_states` function.
	
	Raises
	------
	AssertionError
		If the 'TRACK_ID' column is missing in the input DataFrame.
	
	Notes
	-----
	- The function assumes that the input DataFrame contains a column for tracking cells (`TRACK_ID`) and possibly a 'position' column.
	- The classification behavior depends on the `irreversible_event` and `unique_state` flags:
		- When `irreversible_event` is True, the function classifies events that are considered irreversible.
		- When `unique_state` is True (and `irreversible_event` is False), it classifies unique states using a 50th percentile threshold.
	
	Example
	-------
	>>> df = interpret_track_classification(df, 'class', irreversible_event=True, r2_threshold=0.7)

	"""

	cols = list(df.columns)

	assert 'TRACK_ID' in cols,'Please provide tracked data...'
	if 'position' in cols:
		sort_cols = ['position', 'TRACK_ID']
	else:
		sort_cols = ['TRACK_ID']
	if class_attr.replace('class','status') not in cols:
		df.loc[:,class_attr.replace('class','status')] = df.loc[:,class_attr]

	if irreversible_event:
		unique_state = False

	if irreversible_event:

		df = classify_irreversible_events(df, class_attr, r2_threshold=r2_threshold, percentile_recovery=percentile_recovery, pre_event=pre_event)
	
	elif unique_state:
		
		df = classify_unique_states(df, class_attr, percentile=50, pre_event=pre_event)

	elif transient_event:

		df = classify_transient_events(df, class_attr, pre_event=pre_event)

	return df


def classify_transient_events(data, class_attr, pre_event=None):

	df = data.copy()
	cols = list(df.columns)

	# Control input
	assert 'TRACK_ID' in cols,'Please provide tracked data...'
	if 'position' in cols:
		sort_cols = ['position', 'TRACK_ID']
		df = df.sort_values(by=sort_cols+['FRAME'])
	else:
		sort_cols = ['TRACK_ID']
		df = df.sort_values(by=sort_cols+['FRAME'])
	if pre_event is not None:
		assert 't_'+pre_event in cols,"Pre-event time does not seem to be a valid column in the DataFrame..."
		assert 'class_'+pre_event in cols,"Pre-event class does not seem to be a valid column in the DataFrame..."

	stat_col = class_attr.replace('class','status')
	continuous_stat_col = stat_col.replace('status_','smooth_status_')
	df[continuous_stat_col] = df[stat_col].copy()

	for tid,track in df.groupby(sort_cols):

		indices = track[class_attr].index

		if pre_event is not None:

			if track['class_'+pre_event].values[0]==1:
				df.loc[indices, class_attr] = np.nan
				df.loc[indices, stat_col] = np.nan
				continue
			else:
				# pre-event took place (if left-censored took place at time -1)
				t_pre_event = track['t_'+pre_event].values[0]
				indices_pre = track.loc[track['FRAME']<=t_pre_event,class_attr].index
				df.loc[indices_pre, stat_col] = np.nan # set to NaN all statuses before pre-event
				track.loc[track['FRAME']<=t_pre_event, stat_col] = np.nan
				track.loc[track['FRAME']<=t_pre_event, continuous_stat_col] = np.nan

		status = track[stat_col].to_numpy()
		timeline = track['FRAME'].to_numpy()
		timeline_safe = timeline[status==status]
		status_safe = list(status[status==status])

		peaks, _ = find_peaks(status_safe)
		widths, _, left, right = peak_widths(status_safe, peaks, rel_height=1)
		minimum_weight = 0
		
		if len(peaks)>0:
			idx = np.argmax(widths)
			peak = peaks[idx]; width = widths[idx];
			if width >= minimum_weight:
				left = left[idx]; right = right[idx];
				left = timeline_safe[int(left)]; right = timeline_safe[int(right)];

				df.loc[indices, class_attr] = 0
				t0 = left #take onset + (right - left)/2.0
				df.loc[indices, class_attr.replace('class_','t_')] = t0
				df.loc[track.loc[track[stat_col].isnull(),class_attr].index, continuous_stat_col] = np.nan
				df.loc[track.loc[track['FRAME']<t0,class_attr].index, continuous_stat_col] = 0
				df.loc[track.loc[track['FRAME']>=t0,class_attr].index, continuous_stat_col] = 1
			else:
				df.loc[indices, class_attr] = 1
				df.loc[indices, class_attr.replace('class_','t_')] = -1
				df.loc[indices, continuous_stat_col] = 0
		else:
			df.loc[indices, class_attr] = 1
			df.loc[indices, class_attr.replace('class_','t_')] = -1
			df.loc[indices, continuous_stat_col] = 0

	# restate NaN for out of scope timepoints
	df.loc[df[stat_col].isnull(),continuous_stat_col] = np.nan
	if 'inst_'+stat_col in list(df.columns):
		df = df.drop(columns=['inst_'+stat_col])
	df = df.rename(columns={stat_col: 'inst_'+stat_col})
	df = df.rename(columns={continuous_stat_col: stat_col})
	print("Classes: ",df.loc[df['FRAME']==0,class_attr].value_counts())

	return df


def classify_irreversible_events(data, class_attr, r2_threshold=0.5, percentile_recovery=50, pre_event=None):

	"""
	Classify irreversible events in a tracked dataset based on the status of cells and transitions.

	Parameters
	----------
	df : pandas.DataFrame
		DataFrame containing tracked cell data, including classification and status columns.
	class_attr : str
		Column name for the classification attribute (e.g., 'class') used to update the classification of cell states.
	r2_threshold : float, optional
		R-squared threshold for fitting the model (default is 0.5). Used when estimating the time of transition.

	Returns
	-------
	pandas.DataFrame
		DataFrame with updated classifications for irreversible events, with the following outcomes:
		- Cells with all 0s in the status column are classified as 1 (no event).
		- Cells with all 1s are classified as 2 (event already occurred).
		- Cells with a mix of 0s and 1s are classified as 2 (ambiguous, possible transition).
		- For cells classified as 2, the time of the event is estimated using the `estimate_time` function. If successful they are reclassified as 0 (event).
		- The classification for cells still classified as 2 is revisited using a 95th percentile threshold.

	Notes
	-----
	- The function assumes that cells are grouped by a unique identifier ('TRACK_ID') and sorted by position or ID.
	- The classification is based on the `stat_col` derived from `class_attr` (status column).
	- Cells with no event (all 0s in the status column) are assigned a class value of 1.
	- Cells with irreversible events (all 1s in the status column) are assigned a class value of 2.
	- Cells with transitions (a mix of 0s and 1s) are classified as 2 and their event times are estimated. When successful they are reclassified as 0.
	- After event classification, the function reclassifies leftover ambiguous cases (class 2) using the `classify_unique_states` function.

	Example
	-------
	>>> df = classify_irreversible_events(df, 'class', r2_threshold=0.7)

	"""

	df = data.copy()
	cols = list(df.columns)

	# Control input
	assert 'TRACK_ID' in cols,'Please provide tracked data...'
	if 'position' in cols:
		sort_cols = ['position', 'TRACK_ID']
	else:
		sort_cols = ['TRACK_ID']
	if pre_event is not None:
		assert 't_'+pre_event in cols,"Pre-event time does not seem to be a valid column in the DataFrame..."
		assert 'class_'+pre_event in cols,"Pre-event class does not seem to be a valid column in the DataFrame..."

	stat_col = class_attr.replace('class','status')

	for tid,track in df.groupby(sort_cols):

		indices = track[class_attr].index

		if pre_event is not None:
			if track['class_'+pre_event].values[0]==1:
				df.loc[indices, class_attr] = np.nan
				df.loc[indices, stat_col] = np.nan
				continue
			else:
				# pre-event took place (if left-censored took place at time -1)
				t_pre_event = track['t_'+pre_event].values[0]
				indices_pre = track.loc[track['FRAME']<=t_pre_event,class_attr].index
				df.loc[indices_pre, stat_col] = np.nan # set to NaN all statuses before pre-event
				track.loc[track['FRAME']<=t_pre_event, stat_col] = np.nan
		else:
			# set state to 0 before first detection
			t_firstdetection = track['t_firstdetection'].values[0]
			indices_pre_detection = track.loc[track['FRAME']<=t_firstdetection,class_attr].index
			track.loc[indices_pre_detection,stat_col] = 0.0
			df.loc[indices_pre_detection,stat_col] = 0.0

		# The non-NaN part of track (post pre-event)
		track_valid = track.dropna(subset=stat_col, inplace=False)
		status_values = track_valid[stat_col].to_numpy()

		if np.all([s==0 for s in status_values]):
			# all negative to condition, event not observed
			df.loc[indices, class_attr] = 1
		elif np.all([s==1 for s in status_values]):
			# all positive, event already observed (left-censored)
			df.loc[indices, class_attr] = 2
		else:
			# ambiguity, possible transition, use `unique_state` technique after
			df.loc[indices, class_attr] = 2

	print("Classes after initial pass: ",df.loc[df['FRAME']==0,class_attr].value_counts())

	df.loc[df[class_attr]!=2, class_attr.replace('class', 't')] = -1
	# Try to fit time on class 2 cells (ambiguous)
	df = estimate_time(df, class_attr, model='step_function', class_of_interest=[2], r2_threshold=r2_threshold)
	print("Classes after fit: ", df.loc[df['FRAME']==0,class_attr].value_counts())

	# Revisit class 2 cells to classify as neg/pos with percentile tolerance
	df.loc[df[class_attr]==2,:] = classify_unique_states(df.loc[df[class_attr]==2,:].copy(), class_attr, percentile_recovery)
	print("Classes after unique state recovery: ",df.loc[df['FRAME']==0,class_attr].value_counts())
	
	return df


def classify_unique_states(df, class_attr, percentile=50, pre_event=None):

	"""
	Classify unique cell states based on percentile values of a status attribute in a tracked dataset.

	Parameters
	----------
	df : pandas.DataFrame
		DataFrame containing tracked cell data, including classification and status columns.
	class_attr : str
		Column name for the classification attribute (e.g., 'class') used to update the classification of cell states.
	percentile : int, optional
		Percentile value used to classify the status attribute within the valid frames (default is median).

	Returns
	-------
	pandas.DataFrame
		DataFrame with updated classification for each track and corresponding time (if applicable). 
		The classification is updated based on the calculated percentile:
		- Cells with percentile values that round to 0 (negative to classification) are classified as 1.
		- Cells with percentile values that round to 1 (positive to classification) are classified as 2.
		- If classification is not applicable (NaN), time (`class_attr.replace('class', 't')`) is set to -1.

	Notes
	-----
	- The function assumes that cells are grouped by a unique identifier ('TRACK_ID') and sorted by position or ID.
	- The classification is based on the `stat_col` derived from `class_attr` (status column).
	- NaN values in the status column are excluded from the percentile calculation.
	- For each track, the classification is assigned according to the rounded percentile value.
	- Time (`class_attr.replace('class', 't')`) is set to -1 when the cell state is classified.

	Example
	-------
	>>> df = classify_unique_states(df, 'class', percentile=75)

	"""

	cols = list(df.columns)
	assert 'TRACK_ID' in cols,'Please provide tracked data...'
	if 'position' in cols:
		sort_cols = ['position', 'TRACK_ID']
	else:
		sort_cols = ['TRACK_ID']

	if pre_event is not None:
		assert 't_'+pre_event in cols,"Pre-event time does not seem to be a valid column in the DataFrame..."
		assert 'class_'+pre_event in cols,"Pre-event class does not seem to be a valid column in the DataFrame..."

	stat_col = class_attr.replace('class','status')

	for tid, track in df.groupby(sort_cols):

		indices = track[class_attr].index

		if pre_event is not None:
			if track['class_'+pre_event].values[0]==1:
				df.loc[indices, class_attr] = np.nan
				df.loc[indices, stat_col] = np.nan
				df.loc[indices, stat_col.replace('status_','t_')] = -1
				continue
			else:
				t_pre_event = track['t_'+pre_event].values[0]
				indices_pre = track.loc[track['FRAME']<=t_pre_event, class_attr].index
				df.loc[indices_pre, stat_col] = np.nan
				track.loc[track['FRAME']<=t_pre_event, stat_col] = np.nan

		# Post pre-event track
		track_valid = track.dropna(subset=stat_col, inplace=False)
		status_values = track_valid[stat_col].to_numpy()
		frames = track_valid['FRAME'].to_numpy()
		t_first = track['t_firstdetection'].to_numpy()[0]
		perc_status = np.nanpercentile(status_values[frames>=t_first], percentile)
		
		if perc_status==perc_status:
			c = ceil(perc_status)
			if c==0:
				df.loc[indices, class_attr] = 1
				df.loc[indices, class_attr.replace('class','t')] = -1
			elif c==1:
				df.loc[indices, class_attr] = 2
				df.loc[indices, class_attr.replace('class','t')] = -1
	return df

def classify_cells_from_query(df, status_attr, query):
	
	"""
	Classify cells in a DataFrame based on a query string, assigning classifications to a specified column.

	Parameters
	----------
	df : pandas.DataFrame
		The DataFrame containing cell data to be classified.
	status_attr : str
		The name of the column where the classification results will be stored. 
		- Initially, all cells are assigned a value of 0.
	query : str
		A string representing the condition for classifying the cells. The query is applied to the DataFrame using pandas `.query()`.

	Returns
	-------
	pandas.DataFrame
		The DataFrame with an updated `status_attr` column:
		- Cells matching the query are classified with a value of 1.
		- Cells that have `NaN` values in any of the columns involved in the query are classified as `NaN`.
		- Cells that do not match the query are classified with a value of 0.

	Notes
	-----
	- If the `query` string is empty, a message is printed and no classification is performed.
	- If the query contains columns that are not found in `df`, the entire `class_attr` column is set to `NaN`.
	- Any errors encountered during query evaluation will prevent changes from being applied and will print a message.

	Examples
	--------
	>>> data = {'cell_type': ['A', 'B', 'A', 'B'], 'size': [10, 20, np.nan, 15]}
	>>> df = pd.DataFrame(data)
	>>> classify_cells_from_query(df, 'selected_cells', 'size > 15')
	cell_type  size  selected_cells
	0         A   10.0            0.0
	1         B   20.0            1.0
	2         A    NaN            NaN
	3         B   15.0            0.0

	- If the query string is empty, the function prints a message and returns the DataFrame unchanged.
	- If any of the columns in the query don't exist in the DataFrame, the classification column is set to `NaN`.

	Raises
	------
	Exception
		If the query is invalid or if there are issues with the DataFrame or query syntax, an error message is printed, and `None` is returned.
	
	"""


	# Initialize all states to 0
	if not status_attr.startswith('status_'):
		status_attr = 'status_'+status_attr

	df = df.copy()
	df.loc[:,status_attr] = 0
	df[status_attr] = df[status_attr].astype(float)

	cols = extract_cols_from_query(query)
	print(f"{cols=}")

	cols_in_df = np.all([c in list(df.columns) for c in cols], axis=0)
	if query=='':
		print('The provided query is empty...')
	else:
		try:
			if cols_in_df:
				selection = df.dropna(subset=cols).query(query).index
				null_selection = df[df.loc[:,cols].isna().any(axis=1)].index
				# Set NaN to invalid cells, 1 otherwise
				df.loc[null_selection, status_attr] = np.nan
				df.loc[selection, status_attr] = 1
			else:
				df.loc[:, status_attr] = np.nan
		except Exception as e:
			print(f"The query could not be understood. No filtering was applied. {e}...")
			return None
	return df.copy()

def classify_tracks_from_query(df, event_name, query, irreversible_event=True, unique_state=False, r2_threshold=0.5, percentile_recovery=50):
	
	status_attr = "status_"+event_name
	df = classify_cells_from_query(df, status_attr, query)
	class_attr = "class_"+event_name

	name_map = {status_attr: class_attr}
	df = df.drop(list(set(name_map.values()) & set(df.columns)), axis=1).rename(columns=name_map)
	df.reset_index(inplace=True, drop=True)

	df = interpret_track_classification(df, class_attr, irreversible_event=irreversible_event, unique_state=unique_state, r2_threshold=r2_threshold, percentile_recovery=percentile_recovery)

	return df

def measure_radial_distance_to_center(df, volume, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):

	try:
		df['radial_distance'] = np.sqrt((df[column_labels['x']] - volume[0] / 2) ** 2 + (df[column_labels['y']] - volume[1] / 2) ** 2)
	except Exception as e:
		print(f"{e=}")

	return df

def center_of_mass_to_abs_coordinates(df):
	
	center_of_mass_x_cols = [c for c in list(df.columns) if c.endswith('centre_of_mass_x')]
	center_of_mass_y_cols = [c for c in list(df.columns) if c.endswith('centre_of_mass_y')]
	for c in center_of_mass_x_cols:
		df.loc[:,c.replace('_x','_POSITION_X')] = df[c] + df['POSITION_X']
	for c in center_of_mass_y_cols:
		df.loc[:,c.replace('_y','_POSITION_Y')] = df[c] + df['POSITION_Y']
	df = df.drop(columns = center_of_mass_x_cols+center_of_mass_y_cols)

	return df
