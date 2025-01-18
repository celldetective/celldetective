import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from btrack.io.utils import localizations_to_objects
from btrack import BayesianTracker

from celldetective.measure import measure_features
from celldetective.utils import rename_intensity_column, velocity_per_track
from celldetective.io import interpret_tracking_configuration

import os
import subprocess
import trackpy as tp

abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],'celldetective'])

def track(labels, configuration=None, stack=None, spatial_calibration=1, features=None, channel_names=None,
		  haralick_options=None, return_napari_data=False, view_on_napari=False, mask_timepoints=None, mask_channels=None, volume=(2048,2048),
		  optimizer_options = {'tm_lim': int(12e4)}, track_kwargs={'step_size': 100}, objects=None,
		  clean_trajectories_kwargs=None, btrack_option=True, search_range=None, memory=None,column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'},
		  ):

	"""

	Perform cell tracking on segmented labels using the bTrack library.

	Parameters
	----------
	labels : ndarray
		The segmented labels representing cell objects.
	configuration : Configuration or None
		The bTrack configuration object. If None, a default configuration is used.
	stack : ndarray or None, optional
		The image stack corresponding to the labels. Default is None.
	spatial_calibration : float, optional
		The spatial calibration factor to convert pixel coordinates to physical units. Default is 1.
	features : list or None, optional
		The list of features to extract from the objects. If None, no additional features are extracted. Default is None.
	channel_names : list or None, optional
		The list of channel names corresponding to the image stack. Used for renaming intensity columns in the output DataFrame.
		Default is None.
	haralick_options : dict or None, optional
		The options for Haralick feature extraction. If None, no Haralick features are extracted. Default is None.
	return_napari_data : bool, optional
		Whether to return the napari data dictionary along with the DataFrame. Default is False.
	view_on_napari : bool, optional
		Whether to view the tracking results on napari. Default is False.
	optimizer_options : dict, optional
		The options for the optimizer. Default is {'tm_lim': int(12e4)}.
	track_kwargs : dict, optional
		Additional keyword arguments for the bTrack tracker. Default is {'step_size': 100}.
	clean_trajectories_kwargs : dict or None, optional
		Keyword arguments for the clean_trajectories function to post-process the tracking trajectories. If None, no post-processing is performed.
		Default is None.
	column_labels : dict, optional
		The column labels to use in the output DataFrame. Default is {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	DataFrame or tuple
		If return_napari_data is False, returns the DataFrame containing the tracking results. If return_napari_data is True, returns a tuple
		containing the DataFrame and the napari data dictionary.

	Notes
	-----
	This function performs cell tracking on the segmented labels using the bTrack library. It extracts features from the objects, normalizes
	the features, tracks the objects, and generates a DataFrame with the tracking results. The DataFrame can be post-processed using the
	clean_trajectories function. If specified, the tracking results can be visualized on napari.

	Examples
	--------
	>>> labels = np.array([[1, 1, 2, 2, 0, 0],
						   [1, 1, 1, 2, 2, 0],
						   [0, 0, 1, 2, 0, 0]])
	>>> configuration = cell_config()
	>>> stack = np.random.rand(3, 6)
	>>> df = track(labels, configuration, stack=stack, spatial_calibration=0.5)
	>>> df.head()

	   TRACK_ID  FRAME  POSITION_Y  POSITION_X
	0         0      0         0.0         0.0
	1         0      1         0.0         0.0
	2         0      2         0.0         0.0
	3         1      0         0.5         0.5
	4         1      1         0.5         0.5

	"""

	configuration = interpret_tracking_configuration(configuration)

	if objects is None:
		
		if not btrack_option:
			features = []
			channel_names = None
			haralick_options = None

		objects = extract_objects_and_features(labels, stack, features, 
										   channel_names=channel_names,
										   haralick_options=haralick_options,
										   mask_timepoints=mask_timepoints,
										   mask_channels=mask_channels,
										   )

	if btrack_option:
		columns = list(objects.columns)
		to_remove = ['x','y','class_id','t']
		for tr in to_remove:
			try:
				columns.remove(tr)
			except:
				print(f'column {tr} could not be found...')

		scaler = StandardScaler()
		if columns:
			x = objects[columns].values
			x_scaled = scaler.fit_transform(x)
			df_temp = pd.DataFrame(x_scaled, columns=columns, index = objects.index)
			objects[columns] = df_temp
		else:
			print('Warning: no features were passed to bTrack...')

		# 2) track the objects
		new_btrack_objects = localizations_to_objects(objects)

		with BayesianTracker() as tracker:

			tracker.configure(configuration)

			if columns:
				tracking_updates = ["motion","visual"]
				#tracker.tracking_updates = ["motion","visual"]
				tracker.features = columns
			else:
				tracking_updates = ["motion"]
			
			tracker.append(new_btrack_objects)
			tracker.volume = ((0,volume[0]), (0,volume[1]), (-1e5, 1e5)) #(-1e5, 1e5)
			#print(tracker.volume)
			tracker.track(tracking_updates=tracking_updates, **track_kwargs)
			tracker.optimize(options=optimizer_options)

			data, properties, graph = tracker.to_napari() #ndim=2
		# do the table post processing and napari options
		if data.shape[1]==4:
			df = pd.DataFrame(data, columns=[column_labels['track'],column_labels['time'],column_labels['y'],column_labels['x']])
		elif data.shape[1]==5:
			df = pd.DataFrame(data, columns=[column_labels['track'],column_labels['time'],"z",column_labels['y'],column_labels['x']])
			df = df.drop(columns=['z'])	
		df[column_labels['x']+'_um'] = df[column_labels['x']]*spatial_calibration
		df[column_labels['y']+'_um'] = df[column_labels['y']]*spatial_calibration

	else:
		properties = None
		graph = {}
		print(f"{objects=} {objects.columns=}")
		objects = objects.rename(columns={"t": "frame"})
		if search_range is not None and memory is not None:
			data = tp.link(objects, search_range, memory=memory,link_strategy='auto')
		else:
			print('Please provide a valid search range and memory value...')
			return None
		data['particle'] = data['particle'] + 1 # force track id to start at 1
		df = data.rename(columns={'frame': column_labels['time'], 'x': column_labels['x'], 'y': column_labels['y'], 'particle': column_labels['track']})
		df['state'] = 5.0; df['generation'] = 0.0; df['root'] = 1.0; df['parent'] = 1.0; df['dummy'] = False; df['z'] = 0.0;
		data = df[[column_labels['track'],column_labels['time'],"z",column_labels['y'],column_labels['x']]].to_numpy()
		print(f"{df=}")

	if btrack_option:
		df = df.merge(pd.DataFrame(properties),left_index=True, right_index=True)
		if columns:
			x = df[columns].values
			x_scaled = scaler.inverse_transform(x)
			df_temp = pd.DataFrame(x_scaled, columns=columns, index = df.index)
			df[columns] = df_temp

		# set dummy features to NaN
		df.loc[df['dummy'],['class_id']+columns] = np.nan

	df = df.sort_values(by=[column_labels['track'],column_labels['time']])
	df = velocity_per_track(df, window_size=3, mode='bi')

	if channel_names is not None:
		df = rename_intensity_column(df, channel_names)

	df = write_first_detection_class(df, img_shape=volume, column_labels=column_labels)

	if clean_trajectories_kwargs is not None:
		df = clean_trajectories(df.copy(),**clean_trajectories_kwargs)

	df['ID'] = np.arange(len(df)).astype(int)

	invalid_cols = [c for c in list(df.columns) if c.startswith('Unnamed')]
	if len(invalid_cols)>0:
		df = df.drop(invalid_cols, axis=1)	

	# if view_on_napari:
	# 	view_on_napari_btrack(data,properties,graph,stack=stack,labels=labels,relabel=True)

	if return_napari_data:
		napari_data = {"data": data, "properties": properties, "graph": graph}
		return df, napari_data
	else:
		return df

def extract_objects_and_features(labels, stack, features, channel_names=None, haralick_options=None, mask_timepoints=None, mask_channels=None):

	"""

	Extract objects and features from segmented labels and image stack.

	Parameters
	----------
	labels : ndarray
		The segmented labels representing cell objects.
	stack : ndarray
		The image stack corresponding to the labels.
	features : list or None
		The list of features to extract from the objects. If None, no additional features are extracted.
	channel_names : list or None, optional
		The list of channel names corresponding to the image stack. Used for extracting Haralick features. Default is None.
	haralick_options : dict or None, optional
		The options for Haralick feature extraction. If None, no Haralick features are extracted. Default is None.
	mask_timepoints : list of None, optionak
		Frames to hide during tracking. 
	Returns
	-------
	DataFrame
		The DataFrame containing the extracted object features.

	Notes
	-----
	This function extracts objects and features from the segmented labels and image stack. It computes the specified features for each
	labeled object and returns a DataFrame containing the object features. Additional features such as centroid coordinates can also
	be extracted. If Haralick features are enabled, they are computed based on the image stack using the specified options.

	Examples
	--------
	>>> labels = np.array([[1, 1, 2, 2, 0, 0],
						   [1, 1, 1, 2, 2, 0],
						   [0, 0, 1, 2, 0, 0]])
	>>> stack = np.random.rand(3, 6, 3)
	>>> features = ['area', 'mean_intensity']
	>>> df = extract_objects_and_features(labels, stack, features)
	
	"""

	if features is None:
		features = []

	if stack is None:
		haralick_options = None

	if mask_timepoints is not None:
		for f in mask_timepoints:
			labels[f] = 0.

	nbr_frames = len(labels)
	timestep_dataframes = []

	for t in tqdm(range(nbr_frames),desc='frame'):

		if stack is not None:
			img = stack[t]
		else:
			img = None

		if (haralick_options is not None) and (t==0) and (stack is not None):
			if not 'percentiles' in haralick_options:
				haralick_options.update({'percentiles': (0.01,99.99)})
			if not 'target_channel' in haralick_options:
				haralick_options.update({'target_channel': 0})
			haralick_percentiles = haralick_options['percentiles']
			haralick_channel_index = haralick_options['target_channel']
			min_value = np.nanpercentile(img[:,:,haralick_channel_index].flatten(), haralick_percentiles[0])
			max_value = np.nanpercentile(img[:,:,haralick_channel_index].flatten(), haralick_percentiles[1])			
			haralick_options.update({'clip_values': (min_value, max_value)})

		df_props = measure_features(img, labels[t], features = features+['centroid'], border_dist=None, 
										channels=channel_names, haralick_options=haralick_options, verbose=False)
		df_props.rename(columns={'centroid-1': 'x', 'centroid-0': 'y'},inplace=True)
		df_props['t'] = int(t)
		timestep_dataframes.append(df_props)

	df = pd.concat(timestep_dataframes)	
	df.reset_index(inplace=True, drop=True)

	if mask_channels is not None:
		cols_to_drop = []
		for mc in mask_channels:
			columns = df.columns
			col_contains = [mc in c for c in columns]
			to_remove = np.array(columns)[np.array(col_contains)]
			cols_to_drop.extend(to_remove)
		if len(cols_to_drop)>0:
			df = df.drop(cols_to_drop, axis=1)

	return df


def clean_trajectories(trajectories,remove_not_in_first=False,remove_not_in_last=False,
					   minimum_tracklength=0, interpolate_position_gaps=False,
					   extrapolate_tracks_post=False,
					   extrapolate_tracks_pre=False,
					   interpolate_na=False,
					   column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):

	"""
	Clean trajectories by applying various cleaning operations.

	Parameters
	----------
	trajectories : pandas.DataFrame
		The input DataFrame containing trajectory data.
	remove_not_in_first : bool, optional
		Flag indicating whether to remove tracks not present in the first frame.
		Defaults to True.
	remove_not_in_last : bool, optional
		Flag indicating whether to remove tracks not present in the last frame.
		Defaults to True.
	minimum_tracklength : int, optional
		The minimum length of a track to be retained.
		Defaults to 0.
	interpolate_position_gaps : bool, optional
		Flag indicating whether to interpolate position gaps in tracks.
		Defaults to True.
	extrapolate_tracks_post : bool, optional
		Flag indicating whether to extrapolate tracks after the last known position.
		Defaults to True.
	extrapolate_tracks_pre : bool, optional
		Flag indicating whether to extrapolate tracks before the first known position.
		Defaults to False.
	interpolate_na : bool, optional
		Flag indicating whether to interpolate missing values in tracks.
		Defaults to False.
	column_labels : dict, optional
		Dictionary specifying the column labels used in the input DataFrame.
		The keys represent the following column labels:
		- 'track': The column label for the track ID.
		- 'time': The column label for the timestamp.
		- 'x': The column label for the x-coordinate.
		- 'y': The column label for the y-coordinate.
		Defaults to {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	pandas.DataFrame
		The cleaned DataFrame with trajectories.

	Notes
	-----
	This function applies various cleaning operations to the input DataFrame containing trajectory data.
	The cleaning operations include:
	- Filtering tracks based on their endpoints.
	- Filtering tracks based on their length.
	- Interpolating position gaps in tracks.
	- Extrapolating tracks after the last known position.
	- Extrapolating tracks before the first known position.
	- Interpolating missing values in tracks.

	The input DataFrame is expected to have the following columns:
	- track: The unique ID of each track.
	- time: The timestamp of each data point.
	- x: The x-coordinate of each data point.
	- y: The y-coordinate of each data point.

	Examples
	--------
	>>> cleaned_data = clean_trajectories(trajectories, remove_not_in_first=True, remove_not_in_last=True,
	...                                   minimum_tracklength=10, interpolate_position_gaps=True,
	...                                   extrapolate_tracks_post=True, extrapolate_tracks_pre=False,
	...                                   interpolate_na=True, column_labels={'track': "ID", 'time': 'TIME', 'x': 'X', 'y': 'Y'})
	>>> print(cleaned_data.head())

	"""

	trajectories.reset_index
	trajectories.sort_values(by=[column_labels['track'],column_labels['time']],inplace=True)
	
	if minimum_tracklength>0:
		trajectories = filter_by_tracklength(trajectories.copy(), minimum_tracklength, track_label=column_labels['track'])
	
	if np.any([remove_not_in_first, remove_not_in_last]):
		trajectories = filter_by_endpoints(trajectories.copy(), remove_not_in_first=remove_not_in_first,
		remove_not_in_last=remove_not_in_last, column_labels=column_labels)

	if np.any([extrapolate_tracks_post, extrapolate_tracks_pre]):
		trajectories = extrapolate_tracks(trajectories.copy(), post=extrapolate_tracks_post,
		pre=extrapolate_tracks_pre, column_labels=column_labels)

	if interpolate_position_gaps:
		trajectories = interpolate_time_gaps(trajectories.copy(), column_labels=column_labels)    

	if interpolate_na:
		trajectories = interpolate_nan_properties(trajectories.copy(), track_label=column_labels['track'])
	
	trajectories = trajectories.sort_values(by=[column_labels['track'],column_labels['time']])
	trajectories.reset_index(inplace=True, drop=True)

	if 'class_firstdetection' in list(trajectories.columns):
		for tid, track_group in trajectories.groupby(column_labels['track']):
			indices = track_group.index

			class_values = np.array(track_group['class_firstdetection'].unique())
			class_values = class_values[class_values==class_values]
			t_values = np.array(track_group['t_firstdetection'].unique())
			t_values = t_values[t_values==t_values]
			if len(class_values)==0:
				class_values = 2
				t_values = -1
			else:
				class_values = class_values[0]
				t_values = t_values[0]

			trajectories.loc[indices, 'class_firstdetection'] = class_values
			trajectories.loc[indices, 't_firstdetection'] = t_values

	return trajectories

def interpolate_per_track(group_df):

	"""
	Interpolate missing values within a track.

	Parameters
	----------
	group_df : pandas.DataFrame
		The input DataFrame containing data for a single track.

	Returns
	-------
	pandas.DataFrame
		The interpolated DataFrame with missing values filled.

	Notes
	-----
	This function performs linear interpolation to fill missing values within a track.
	Missing values are interpolated based on the neighboring data points in the track.

	"""

	for c in list(group_df.columns):
		group_df_new_dtype = group_df[c].infer_objects(copy=False)
		if group_df_new_dtype.dtype!='O':
			group_df[c] = group_df_new_dtype.interpolate(method='linear',limit_direction="both")
	
	#interpolated_group = group_df.interpolate(method='linear',limit_direction="both")

	return group_df

def interpolate_nan_properties(trajectories, track_label="TRACK_ID"):

	"""
	Interpolate missing values within tracks in the input DataFrame.

	Parameters
	----------
	trajectories : pandas.DataFrame
		The input DataFrame containing trajectory data.
	track_label : str, optional
		The column label for the track ID.
		Defaults to "TRACK_ID".

	Returns
	-------
	pandas.DataFrame
		The DataFrame with missing values interpolated within tracks.

	Notes
	-----
	This function groups the input DataFrame by track ID and applies `interpolate_per_track` function
	to interpolate missing values within each track.
	Missing values are interpolated based on the neighboring data points in each track.

	The input DataFrame is expected to have a column with the specified `track_label` containing the track IDs.

	Examples
	--------
	>>> interpolated_data = interpolate_nan_properties(trajectories, track_label="ID")
	>>> print(interpolated_data.head())

	"""

	trajectories = trajectories.groupby(track_label, group_keys=False).apply(interpolate_per_track)

	return trajectories


def filter_by_endpoints(trajectories, remove_not_in_first=True, remove_not_in_last=False,
						column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):

	"""
	Filter trajectories based on their endpoints.

	Parameters
	----------
	trajectories : pandas.DataFrame
		The input DataFrame containing trajectory data.
	remove_not_in_first : bool, optional
		Flag indicating whether to remove tracks not present in the first frame.
		Defaults to True.
	remove_not_in_last : bool, optional
		Flag indicating whether to remove tracks not present in the last frame.
		Defaults to False.
	column_labels : dict, optional
		Dictionary specifying the column labels used in the input DataFrame.
		The keys represent the following column labels:
		- 'track': The column label for the track ID.
		- 'time': The column label for the timestamp.
		- 'x': The column label for the x-coordinate.
		- 'y': The column label for the y-coordinate.
		Defaults to {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	pandas.DataFrame
		The filtered DataFrame with trajectories based on their endpoints.

	Notes
	-----
	This function filters the input DataFrame based on the endpoints of the trajectories.
	The filtering can be performed in three modes:
	- remove_not_in_first=True and remove_not_in_last=False: Remove tracks that are not present in the first frame.
	- remove_not_in_first=False and remove_not_in_last=True: Remove tracks that are not present in the last frame.
	- remove_not_in_first=True and remove_not_in_last=True: Remove tracks that are not present in both the first and last frames.

	The input DataFrame is expected to have the following columns:
	- track: The unique ID of each track.
	- time: The timestamp of each data point.
	- x: The x-coordinate of each data point.
	- y: The y-coordinate of each data point.

	Examples
	--------
	>>> filtered_data = filter_by_endpoints(trajectories, remove_not_in_first=True, remove_not_in_last=False, column_labels={'track': "ID", 'time': 'TIME', 'x': 'X', 'y': 'Y'})
	>>> print(filtered_data.head())

	"""

	if (remove_not_in_first)*(not remove_not_in_last):
		# filter tracks not in first frame
		leftover_tracks = trajectories.groupby(column_labels['track']).min().index[trajectories.groupby(column_labels['track']).min()[column_labels['time']]==np.amin(trajectories[column_labels['time']])]
		trajectories = trajectories.loc[trajectories[column_labels['track']].isin(leftover_tracks)]
		
	elif (remove_not_in_last)*(not remove_not_in_first):
		# filter tracks not in last frame
		leftover_tracks = trajectories.groupby(column_labels['track']).max().index[trajectories.groupby(column_labels['track']).max()[column_labels['time']]==np.amax(trajectories[column_labels['time']])]
		trajectories = trajectories.loc[trajectories[column_labels['track']].isin(leftover_tracks)]
	  
	elif remove_not_in_first*remove_not_in_last:
		# filter tracks both not in first and last frame
		leftover_tracks = trajectories.groupby(column_labels['track']).max().index[(trajectories.groupby(column_labels['track']).max()[column_labels['time']]==np.amax(trajectories[column_labels['time']]))*(trajectories.groupby(column_labels['track']).min()[column_labels['time']]==np.amin(trajectories[column_labels['time']]))]
		trajectories = trajectories.loc[trajectories[column_labels['track']].isin(leftover_tracks)]

	trajectories = trajectories.sort_values(by=[column_labels['track'],column_labels['time']])

	return trajectories

def filter_by_tracklength(trajectories, minimum_tracklength, track_label="TRACK_ID"):
	
	"""
	Filter trajectories based on the minimum track length.

	Parameters
	----------
	trajectories : pandas.DataFrame
		The input DataFrame containing trajectory data.
	minimum_tracklength : int
		The minimum length required for a track to be included.
	track_label : str, optional
		The column name in the DataFrame that represents the track ID.
		Defaults to "TRACK_ID".

	Returns
	-------
	pandas.DataFrame
		The filtered DataFrame with trajectories that meet the minimum track length.

	Notes
	-----
	This function removes any tracks from the input DataFrame that have a length
	(number of data points) less than the specified minimum track length.

	Examples
	--------
	>>> filtered_data = filter_by_tracklength(trajectories, 10, track_label="TrackID")
	>>> print(filtered_data.head())

	"""
	
	if minimum_tracklength>0:
		
		leftover_tracks = trajectories.groupby(track_label, group_keys=False).size().index[trajectories.groupby(track_label, group_keys=False).size() > minimum_tracklength]
		trajectories = trajectories.loc[trajectories[track_label].isin(leftover_tracks)]
	
	trajectories = trajectories.reset_index(drop=True)
	
	return trajectories
	

def interpolate_time_gaps(trajectories, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):

	"""
	Interpolate time gaps in trajectories.

	Parameters
	----------
	trajectories : pandas.DataFrame
		The input DataFrame containing trajectory data.
	column_labels : dict, optional
		Dictionary specifying the column labels used in the input DataFrame.
		The keys represent the following column labels:
		- 'track': The column label for the track ID.
		- 'time': The column label for the timestamp.
		- 'x': The column label for the x-coordinate.
		- 'y': The column label for the y-coordinate.
		Defaults to {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	pandas.DataFrame
		The interpolated DataFrame with reduced time gaps in trajectories.

	Notes
	-----
	This function performs interpolation on the input trajectories to reduce time gaps between data points.
	It uses linear interpolation to fill missing values for the specified x and y coordinate attributes.

	The input DataFrame is expected to have the following columns:
	- track: The unique ID of each track.
	- time: The timestamp of each data point (in seconds).
	- x: The x-coordinate of each data point.
	- y: The y-coordinate of each data point.

	Examples
	--------
	>>> interpolated_data = interpolate_time_gaps(trajectories, column_labels={'track': "ID", 'time': 'TIME', 'x': 'X', 'y': 'Y'})
	>>> print(interpolated_data.head())

	"""

	trajectories[column_labels['time']] = pd.to_datetime(trajectories[column_labels['time']], unit='s')
	trajectories.set_index(column_labels['track'], inplace=True)
	trajectories = trajectories.groupby(column_labels['track'], group_keys=True).apply(lambda x: x.set_index(column_labels['time']).resample('1s').asfreq()).reset_index()
	trajectories[[column_labels['x'], column_labels['y']]] = trajectories.groupby(column_labels['track'], group_keys=False)[[column_labels['x'], column_labels['y']]].apply(lambda x: x.interpolate(method='linear'))
	trajectories.reset_index(drop=True, inplace=True)
	trajectories[column_labels['time']] = trajectories[column_labels['time']].astype('int64').astype(float) / 10**9
	#trajectories[column_labels['time']] = trajectories[column_labels['time']].astype('int64')
	trajectories.sort_values(by=[column_labels['track'],column_labels['time']],inplace=True)
	
	return trajectories
		

def extrapolate_tracks(trajectories, post=False, pre=False, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):
	
	"""
	Extrapolate tracks in trajectories.

	Parameters
	----------
	trajectories : pandas.DataFrame
		The input DataFrame containing trajectory data.
	post : bool, optional
		Flag indicating whether to perform post-extrapolation.
		Defaults to True.
	pre : bool, optional
		Flag indicating whether to perform pre-extrapolation.
		Defaults to False.
	column_labels : dict, optional
		Dictionary specifying the column labels used in the input DataFrame.
		The keys represent the following column labels:
		- 'track': The column label for the track ID.
		- 'time': The column label for the timestamp.
		- 'x': The column label for the x-coordinate.
		- 'y': The column label for the y-coordinate.
		Defaults to {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	pandas.DataFrame
		The extrapolated DataFrame with extended tracks.

	Notes
	-----
	This function extrapolates tracks in the input DataFrame by repeating the last known position
	either after (post-extrapolation) or before (pre-extrapolation) the available data.

	The input DataFrame is expected to have the following columns:
	- track: The unique ID of each track.
	- time: The timestamp of each data point.
	- x: The x-coordinate of each data point.
	- y: The y-coordinate of each data point.

	Examples
	--------
	>>> extrapolated_data = extrapolate_tracks(trajectories, post=True, pre=False, column_labels={'track': "ID", 'time': 'TIME', 'x': 'X', 'y': 'Y'})
	>>> print(extrapolated_data.head())

	"""
	
	if post:
		
		# get the maximum time T in the dataframe
		max_time = trajectories[column_labels['time']].max()

		# extrapolate the position until time T by repeating the last known position
		df_extrapolated = pd.DataFrame()
		for track_id, group in trajectories.groupby(column_labels['track']):
			last_known_position = group.loc[group[column_labels['time']] <= max_time].tail(1)[[column_labels['time'],column_labels['x'], column_labels['y']]].values
			extrapolated_frames = pd.DataFrame({column_labels['time']: np.arange(last_known_position[0][0] + 1, max_time + 1)})
			extrapolated_positions = pd.DataFrame({column_labels['x']: last_known_position[0][1], column_labels['y']: last_known_position[0][2]}, index=np.arange(last_known_position[0][0] + 1, max_time + 1))
			track_data = extrapolated_frames.join(extrapolated_positions, how="inner", on=column_labels['time'])
			track_data[column_labels['track']] = track_id

			if len(df_extrapolated)==0:
				df_extrapolated = track_data
			elif len(track_data)!=0:
				df_extrapolated = pd.concat([df_extrapolated, track_data])


		# concatenate the original dataframe and the extrapolated dataframe
		trajectories = pd.concat([trajectories, df_extrapolated], axis=0)
		# sort the dataframe by TRACK_ID and FRAME
		trajectories.sort_values([column_labels['track'], column_labels['time']], inplace=True)

	if pre:
		
		# get the maximum time T in the dataframe
		min_time = 0 #trajectories[column_labels['time']].min()

		# extrapolate the position until time T by repeating the last known position
		df_extrapolated = pd.DataFrame()
		for track_id, group in trajectories.groupby(column_labels['track']):
			last_known_position = group.loc[group[column_labels['time']] >= min_time].head(1)[[column_labels['time'],column_labels['x'], column_labels['y']]].values
			extrapolated_frames = pd.DataFrame({column_labels['time']: np.arange(min_time, last_known_position[0][0] + 1)})
			extrapolated_positions = pd.DataFrame({column_labels['x']: last_known_position[0][1], column_labels['y']: last_known_position[0][2]}, index=np.arange(min_time, last_known_position[0][0]))
			track_data = extrapolated_frames.join(extrapolated_positions, how="inner", on=column_labels['time'])
			track_data[column_labels['track']] = track_id
			df_extrapolated = pd.concat([df_extrapolated, track_data])

		# concatenate the original dataframe and the extrapolated dataframe
		trajectories = pd.concat([trajectories, df_extrapolated], axis=0)

		# sort the dataframe by TRACK_ID and FRAME
		trajectories.sort_values([column_labels['track'], column_labels['time']], inplace=True)
	
	return trajectories

def compute_instantaneous_velocity(trajectories, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):

	"""
	
	Compute the instantaneous velocity for each point in the trajectories.

	Parameters
	----------
	trajectories : pandas.DataFrame
		The input DataFrame containing trajectory data.
	column_labels : dict, optional
		A dictionary specifying the column labels for track ID, time, position X, and position Y.
		Defaults to {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	pandas.DataFrame
		The DataFrame with added 'velocity' column representing the instantaneous velocity for each point.

	Notes
	-----
	This function calculates the instantaneous velocity for each point in the trajectories.
	The velocity is computed as the Euclidean distance traveled divided by the time difference between consecutive points.

	The input DataFrame is expected to have columns with the specified column labels for track ID, time, position X, and position Y.

	Examples
	--------
	>>> velocity_data = compute_instantaneous_velocity(trajectories)
	>>> print(velocity_data.head())

	"""

	# Calculate the time differences and position differences
	trajectories['dt'] = trajectories.groupby(column_labels['track'])[column_labels['time']].diff()
	trajectories['dx'] = trajectories.groupby(column_labels['track'])[column_labels['x']].diff()
	trajectories['dy'] = trajectories.groupby(column_labels['track'])[column_labels['y']].diff()

	# Calculate the instantaneous velocity
	trajectories['velocity'] = np.sqrt(trajectories['dx']**2 +trajectories['dy']**2) / trajectories['dt']
	trajectories = trajectories.drop(['dx', 'dy', 'dt'], axis=1)
	trajectories = trajectories.sort_values(by=[column_labels['track'],column_labels['time']])    

	return trajectories

def instantaneous_diffusion(positions_x, positions_y, timeline):
	
	"""
	Compute the instantaneous diffusion coefficients for each position coordinate.

	Parameters
	----------
	positions_x : numpy.ndarray
		Array of x-coordinates of positions.
	positions_y : numpy.ndarray
		Array of y-coordinates of positions.
	timeline : numpy.ndarray
		Array of corresponding time points.

	Returns
	-------
	numpy.ndarray
		Array of instantaneous diffusion coefficients for each position coordinate.

	Notes
	-----
	The function calculates the instantaneous diffusion coefficients for each position coordinate (x, y) based on the provided positions and timeline.
	The diffusion coefficient at each time point is computed using the formula:
	D = ((x[t+1] - x[t-1])^2 / (2 * (t[t+1] - t[t-1]))) + (1 / (t[t+1] - t[t-1])) * ((x[t+1] - x[t]) * (x[t] - x[t-1]))
	where x represents the position coordinate (x or y) and t represents the corresponding time point.

	Examples
	--------
	>>> x = np.array([0, 1, 2, 3, 4, 5])
	>>> y = np.array([0, 1, 4, 9, 16, 25])
	>>> t = np.array([0, 1, 2, 3, 4, 5])
	>>> diff = instantaneous_diffusion(x, y, t)
	>>> print(diff)

	"""

	diff = np.zeros((len(positions_x),2))
	diff[:,:] = np.nan
	
	for t in range(1,len(positions_x)-1):
		diff[t,0] = (positions_x[t+1] - positions_x[t-1])**2/(2*(timeline[t+1] - timeline[t-1])) + 1/(timeline[t+1] - timeline[t-1])*((positions_x[t+1] - positions_x[t])*(positions_x[t] - positions_x[t-1]))

	for t in range(1,len(positions_y)-1):
		diff[t,1] = (positions_y[t+1] - positions_y[t-1])**2/(2*(timeline[t+1] - timeline[t-1])) + 1/(timeline[t+1] - timeline[t-1])*((positions_y[t+1] - positions_y[t])*(positions_y[t] - positions_y[t-1]))

	return diff

def magnitude_diffusion(diffusion_vector):

	"""
	Compute the magnitude of diffusion for each diffusion vector.

	Parameters
	----------
	diffusion_vector : numpy.ndarray
		Array of diffusion vectors.

	Returns
	-------
	numpy.ndarray
		Array of magnitudes of diffusion.

	Notes
	-----
	The function calculates the magnitude of diffusion for each diffusion vector (x, y) based on the provided diffusion vectors.
	The magnitude of diffusion is computed as the Euclidean norm of the diffusion vector.

	Examples
	--------
	>>> diffusion = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 0.5]])
	>>> magnitudes = magnitude_diffusion(diffusion)
	>>> print(magnitudes)

	"""

	return np.sqrt(diffusion_vector[:,0]**2+diffusion_vector[:,1]**2)


def compute_instantaneous_diffusion(trajectories, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):

	"""
	
	Compute the instantaneous diffusion for each track in the provided trajectories DataFrame.

	Parameters
	----------
	trajectories : DataFrame
		The input DataFrame containing trajectories with position and time information.
	column_labels : dict, optional
		A dictionary specifying the column labels for track ID, time, x-coordinate, and y-coordinate. 
		The default is {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}.

	Returns
	-------
	DataFrame
		The modified DataFrame with an additional column "diffusion" containing the computed diffusion values.

	Notes
	-----

	The instantaneous diffusion is calculated using the positions and times of each track. The diffusion values
	are computed for each track individually and added as a new column "diffusion" in the output DataFrame.

	Examples
	--------
	>>> trajectories = pd.DataFrame({'TRACK_ID': [1, 1, 1, 2, 2, 2],
	...                              'FRAME': [0, 1, 2, 0, 1, 2],
	...                              'POSITION_X': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
	...                              'POSITION_Y': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})
	>>> compute_instantaneous_diffusion(trajectories)
	# Output DataFrame with added "diffusion" column

	"""

	trajectories = trajectories.sort_values(by=[column_labels['track'],column_labels['time']])
	trajectories['diffusion'] = np.nan  
	
	for tid,group in trajectories.groupby(column_labels['track']):

		indices = group.index
		x = group[column_labels['x']].to_numpy()
		y = group[column_labels['y']].to_numpy()
		t = group[column_labels['time']].to_numpy()
		
		if len(x)>3: #to have t-1,t,t+1
			diff = instantaneous_diffusion(x,y,t)
			d = magnitude_diffusion(diff)
			trajectories.loc[indices, "diffusion"] = d

	return trajectories

def track_at_position(pos, mode, return_tracks=False, view_on_napari=False, threads=1):
	
	pos = pos.replace('\\','/')
	pos = rf"{pos}"
	assert os.path.exists(pos),f'Position {pos} is not a valid path.'
	if not pos.endswith('/'):
		pos += '/'

	script_path = os.sep.join([abs_path, 'scripts', 'track_cells.py'])	
	cmd = f'python "{script_path}" --pos "{pos}" --mode "{mode}" --threads "{threads}"'
	subprocess.call(cmd, shell=True)

	track_table = pos + os.sep.join(["output","tables",f"trajectories_{mode}.csv"])
	if return_tracks:
		df = pd.read_csv(track_table)
		return df
	else:
		return None	

def write_first_detection_class(df, img_shape=None, edge_threshold=20, column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}):
	
	"""
	Assigns a classification and first detection time to tracks in the given DataFrame. This function must be called
	before any track post-processing.

	This function computes the first detection time and a detection class (`class_firstdetection`) for each track in the data. 
	Tracks that start on or near the image edge, or those detected at the initial frame, are marked with special classes.

	Parameters
	----------
	df : pandas.DataFrame
		A DataFrame containing track data. Expected to have at least the columns specified in `column_labels` and `class_id` (mask value).

	img_shape : tuple of int, optional
		The shape of the image as `(height, width)`. Used to determine whether the first detection occurs near the image edge.

	edge_threshold : int, optional, default=20
		The distance in pixels from the image edge to consider a detection as near the edge.

	column_labels : dict, optional
		A dictionary mapping logical column names to actual column names in `tab`. Keys include:
		
			- `'track'`: The column indicating the track ID (default: `"TRACK_ID"`).
			- `'time'`: The column indicating the frame/time (default: `"FRAME"`).
			- `'x'`: The column indicating the X-coordinate (default: `"POSITION_X"`).
			- `'y'`: The column indicating the Y-coordinate (default: `"POSITION_Y"`).

	Returns
	-------
	pandas.DataFrame
		The input DataFrame `df` with two additional columns:
		
			- `'class_firstdetection'`: A class assigned based on detection status:
			
				- `0`: Valid detection not near the edge and not at the initial frame.
				- `2`: Detection near the edge, at the initial frame, or no detection available.
		
			- `'t_firstdetection'`: The adjusted first detection time (in frame units):
			
				- `-1`: Indicates no valid detection or detection near the edge.
				- A float value representing the adjusted first detection time otherwise.

	Notes
	-----
	- The function assumes that tracks are grouped and sorted by track ID and frame.
	- Detections near the edge or at the initial frame (frame 0) are considered invalid and assigned special values.
	- If `img_shape` is not provided, edge checks are skipped.

	"""

	df = df.sort_values(by=[column_labels['track'],column_labels['time']])
	for tid,track_group in df.groupby(column_labels['track']):
		indices = track_group.index
		detection = track_group['class_id'].values
		timeline = track_group[column_labels['time']].values
		positions_x = track_group[column_labels['x']].values
		positions_y = track_group[column_labels['y']].values
		dt = 1
		
		# Initialize
		cclass = 2; t_first = np.nan;

		if np.any(detection==detection):
			t_first = timeline[detection==detection][0]
			x_first = positions_x[detection==detection][0]; y_first = positions_y[detection==detection][0];

			edge_test = False
			if img_shape is not None:
				edge_test = (x_first < edge_threshold) or (y_first < edge_threshold) or (y_first > (img_shape[0] - edge_threshold)) or (x_first > (img_shape[1] - edge_threshold))

			cclass = 0
			if t_first<=0 or edge_test:
				t_first = -1
				cclass = 2
			else:
				t_first =  float(t_first) - float(dt)
				if t_first==0:
					t_first += 0.01
		else:
			t_first = -1
			cclass = 2

		df.loc[indices, 'class_firstdetection'] = cclass
		df.loc[indices, 't_firstdetection'] = t_first

	return df



if __name__ == "__main__":
	track_at_position("/home/limozin/Documents/Experiments/MinimumJan/W4/401",
					  "targets",
					  )