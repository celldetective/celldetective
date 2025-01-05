from natsort import natsorted
from PyQt5.QtWidgets import QMessageBox
from glob import glob
from tifffile import imread, TiffFile
import numpy as np
import os
import pandas as pd
import napari
import json

import gc
from tqdm import tqdm
import threading
import concurrent.futures

from csbdeep.utils import normalize_mi_ma
from csbdeep.io import save_tiff_imagej_compatible

import skimage.io as skio
from skimage.measure import regionprops_table, label

from btrack.datasets import cell_config
from magicgui import magicgui
from pathlib import Path, PurePath
from shutil import copyfile, rmtree

from celldetective.utils import _rearrange_multichannel_frame, _fix_no_contrast, zoom_multiframes,ConfigSectionMap, extract_experiment_channels, _extract_labels_from_config, get_zenodo_files, download_zenodo_file
from celldetective.utils import interpolate_nan_multichannel, _estimate_scale_factor, _extract_channel_indices_from_config, _extract_channel_indices, _extract_nbr_channels_from_config, _get_img_num_per_channel, normalize_per_channel

from stardist import fill_label_holes
from skimage.transform import resize


def extract_experiment_from_well(well_path):

	"""
	Extracts the experiment directory path from a given well directory path.

	Parameters
	----------
	well_path : str
		The file system path to a well directory. The path should end with the well folder, 
		but it does not need to include a trailing separator.

	Returns
	-------
	str
		The path to the experiment directory, which is assumed to be two levels above the well directory.

	Notes
	-----
	- This function expects the well directory to be organized such that the experiment directory is 
	  two levels above it in the file system hierarchy.
	- If the input path does not end with a file separator (`os.sep`), one is appended before processing.

	Example
	-------
	>>> well_path = "/path/to/experiment/plate/well"
	>>> extract_experiment_from_well(well_path)
	'/path/to/experiment'
	"""

	if not well_path.endswith(os.sep):
		well_path += os.sep
	exp_path_blocks = well_path.split(os.sep)[:-2]
	experiment = os.sep.join(exp_path_blocks)
	return experiment

def extract_well_from_position(pos_path):

	"""
	Extracts the well directory path from a given position directory path.

	Parameters
	----------
	pos_path : str
		The file system path to a position directory. The path should end with the position folder,
		but it does not need to include a trailing separator.

	Returns
	-------
	str
		The path to the well directory, which is assumed to be two levels above the position directory,
		with a trailing separator appended.

	Notes
	-----
	- This function expects the position directory to be organized such that the well directory is
	  two levels above it in the file system hierarchy.
	- If the input path does not end with a file separator (`os.sep`), one is appended before processing.

	Example
	-------
	>>> pos_path = "/path/to/experiment/plate/well/position"
	>>> extract_well_from_position(pos_path)
	'/path/to/experiment/plate/well/'
	"""

	if not pos_path.endswith(os.sep):
		pos_path += os.sep
	well_path_blocks = pos_path.split(os.sep)[:-2]
	well_path = os.sep.join(well_path_blocks)+os.sep
	return well_path

def extract_experiment_from_position(pos_path):

	"""
	Extracts the experiment directory path from a given position directory path.

	Parameters
	----------
	pos_path : str
		The file system path to a position directory. The path should end with the position folder,
		but it does not need to include a trailing separator.

	Returns
	-------
	str
		The path to the experiment directory, which is assumed to be three levels above the position directory.

	Notes
	-----
	- This function expects the position directory to be organized hierarchically such that the experiment directory
	  is three levels above it in the file system hierarchy.
	- If the input path does not end with a file separator (`os.sep`), one is appended before processing.

	Example
	-------
	>>> pos_path = "/path/to/experiment/plate/well/position"
	>>> extract_experiment_from_position(pos_path)
	'/path/to/experiment'
	"""

	pos_path = pos_path.replace(os.sep, '/')
	if not pos_path.endswith('/'):
		pos_path += '/'
	exp_path_blocks = pos_path.split('/')[:-3]
	experiment = os.sep.join(exp_path_blocks)

	return experiment

def collect_experiment_metadata(pos_path=None, well_path=None):
	
	"""
	Collects and organizes metadata for an experiment based on a given position or well directory path.

	Parameters
	----------
	pos_path : str, optional
		The file system path to a position directory. If provided, it will be used to extract metadata.
		This parameter takes precedence over `well_path`.
	well_path : str, optional
		The file system path to a well directory. If `pos_path` is not provided, this path will be used to extract metadata.

	Returns
	-------
	dict
		A dictionary containing the following metadata:
		- `"pos_path"`: The path to the position directory (or `None` if not provided).
		- `"position"`: The same as `pos_path`.
		- `"pos_name"`: The name of the position (or `0` if `pos_path` is not provided).
		- `"well_path"`: The path to the well directory.
		- `"well_name"`: The name of the well.
		- `"well_nbr"`: The numerical identifier of the well.
		- `"experiment"`: The path to the experiment directory.
		- `"antibody"`: The antibody associated with the well.
		- `"concentration"`: The concentration associated with the well.
		- `"cell_type"`: The cell type associated with the well.
		- `"pharmaceutical_agent"`: The pharmaceutical agent associated with the well.

	Notes
	-----
	- At least one of `pos_path` or `well_path` must be provided.
	- The function determines the experiment path by navigating the directory structure and extracts metadata for the 
	  corresponding well and position.
	- The metadata is derived using helper functions like `extract_experiment_from_position`, `extract_well_from_position`,
	  and `get_experiment_*` family of functions.

	Example
	-------
	>>> pos_path = "/path/to/experiment/plate/well/position"
	>>> metadata = collect_experiment_metadata(pos_path=pos_path)
	>>> metadata["well_name"]
	'W1'

	>>> well_path = "/path/to/experiment/plate/well"
	>>> metadata = collect_experiment_metadata(well_path=well_path)
	>>> metadata["concentration"]
	10.0
	"""

	if pos_path is not None:
		if not pos_path.endswith(os.sep):
			pos_path += os.sep
		experiment = extract_experiment_from_position(pos_path)
		well_path = extract_well_from_position(pos_path)
	elif well_path is not None:
		if not well_path.endswith(os.sep):
			well_path += os.sep
		experiment = extract_experiment_from_well(well_path)
		
	wells = list(get_experiment_wells(experiment))
	idx = wells.index(well_path)
	well_name, well_nbr = extract_well_name_and_number(well_path)
	if pos_path is not None:
		pos_name = extract_position_name(pos_path)
	else:
		pos_name = 0
	concentrations = get_experiment_concentrations(experiment, dtype=float)
	cell_types = get_experiment_cell_types(experiment)
	antibodies = get_experiment_antibodies(experiment)
	pharmaceutical_agents = get_experiment_pharmaceutical_agents(experiment)
	
	dico = {"pos_path": pos_path, "position": pos_path, "pos_name": pos_name, "well_path": well_path, "well_name": well_name, "well_nbr": well_nbr, "experiment": experiment, "antibody": antibodies[idx], "concentration": concentrations[idx], "cell_type": cell_types[idx], "pharmaceutical_agent": pharmaceutical_agents[idx]}

	meta = get_experiment_metadata(experiment) # None or dict of metadata
	if meta is not None:
		keys = list(meta.keys())
		for k in keys:
			dico.update({k: meta[k]})
	
	return dico


def get_experiment_wells(experiment):
	
	"""
	Retrieves the list of well directories from a given experiment directory, sorted
	naturally and returned as a NumPy array of strings.

	Parameters
	----------
	experiment : str
		The path to the experiment directory from which to retrieve well directories.

	Returns
	-------
	np.ndarray
		An array of strings, each representing the full path to a well directory within the specified
		experiment. The array is empty if no well directories are found.

	Notes
	-----
	- The function assumes well directories are prefixed with 'W' and uses this to filter directories
	  within the experiment folder.

	- Natural sorting is applied to the list of wells to ensure that the order is intuitive (e.g., 'W2'
	  comes before 'W10'). This sorting method is especially useful when dealing with numerical sequences
	  that are part of the directory names.

	"""

	if not experiment.endswith(os.sep):
		experiment += os.sep

	wells = natsorted(glob(experiment + "W*" + os.sep))
	return np.array(wells, dtype=str)


def get_config(experiment):

	"""
	Retrieves the path to the configuration file for a given experiment.

	Parameters
	----------
	experiment : str
		The file system path to the experiment directory.

	Returns
	-------
	str
		The full path to the configuration file (`config.ini`) within the experiment directory.

	Raises
	------
	AssertionError
		If the `config.ini` file does not exist in the specified experiment directory.

	Notes
	-----
	- The function ensures that the provided experiment path ends with the appropriate file separator (`os.sep`) 
	  before appending `config.ini` to locate the configuration file.
	- The configuration file is expected to be named `config.ini` and located at the root of the experiment directory.

	Example
	-------
	>>> experiment = "/path/to/experiment"
	>>> config_path = get_config(experiment)
	>>> print(config_path)
	'/path/to/experiment/config.ini'
	"""

	if not experiment.endswith(os.sep):
		experiment += os.sep

	config = experiment + 'config.ini'
	config = rf"{config}"
	assert os.path.exists(config), 'The experiment configuration could not be located...'
	return config


def get_spatial_calibration(experiment):
	
	"""
	Retrieves the spatial calibration factor for an experiment.

	Parameters
	----------
	experiment : str
		The file system path to the experiment directory.

	Returns
	-------
	float
		The spatial calibration factor (pixels to micrometers conversion), extracted from the experiment's configuration file.

	Raises
	------
	AssertionError
		If the configuration file (`config.ini`) does not exist in the specified experiment directory.
	KeyError
		If the "pxtoum" key is not found under the "MovieSettings" section in the configuration file.
	ValueError
		If the retrieved "pxtoum" value cannot be converted to a float.

	Notes
	-----
	- The function retrieves the calibration factor by first locating the configuration file for the experiment using `get_config()`.
	- It expects the configuration file to have a section named `MovieSettings` containing the key `pxtoum`.
	- This factor defines the conversion from pixels to micrometers for spatial measurements.

	Example
	-------
	>>> experiment = "/path/to/experiment"
	>>> calibration = get_spatial_calibration(experiment)
	>>> print(calibration)
	0.325  # pixels-to-micrometers conversion factor
	"""

	config = get_config(experiment)
	PxToUm = float(ConfigSectionMap(config, "MovieSettings")["pxtoum"])

	return PxToUm


def get_temporal_calibration(experiment):
	
	"""
	Retrieves the temporal calibration factor for an experiment.

	Parameters
	----------
	experiment : str
		The file system path to the experiment directory.

	Returns
	-------
	float
		The temporal calibration factor (frames to minutes conversion), extracted from the experiment's configuration file.

	Raises
	------
	AssertionError
		If the configuration file (`config.ini`) does not exist in the specified experiment directory.
	KeyError
		If the "frametomin" key is not found under the "MovieSettings" section in the configuration file.
	ValueError
		If the retrieved "frametomin" value cannot be converted to a float.

	Notes
	-----
	- The function retrieves the calibration factor by locating the configuration file for the experiment using `get_config()`.
	- It expects the configuration file to have a section named `MovieSettings` containing the key `frametomin`.
	- This factor defines the conversion from frames to minutes for temporal measurements.

	Example
	-------
	>>> experiment = "/path/to/experiment"
	>>> calibration = get_temporal_calibration(experiment)
	>>> print(calibration)
	0.5  # frames-to-minutes conversion factor
	"""

	config = get_config(experiment)
	FrameToMin = float(ConfigSectionMap(config, "MovieSettings")["frametomin"])

	return FrameToMin

def get_experiment_metadata(experiment):

	config = get_config(experiment)
	metadata = ConfigSectionMap(config, "Metadata")
	return metadata


def get_experiment_concentrations(experiment, dtype=str):
	
	"""
	Retrieves the concentrations associated with each well in an experiment.

	Parameters
	----------
	experiment : str
		The file system path to the experiment directory.
	dtype : type, optional
		The data type to which the concentrations should be converted (default is `str`).

	Returns
	-------
	numpy.ndarray
		An array of concentrations for each well, converted to the specified data type.

	Raises
	------
	AssertionError
		If the configuration file (`config.ini`) does not exist in the specified experiment directory.
	KeyError
		If the "concentrations" key is not found under the "Labels" section in the configuration file.
	ValueError
		If the retrieved concentrations cannot be converted to the specified data type.

	Notes
	-----
	- The function retrieves the configuration file using `get_config()` and expects a section `Labels` containing 
	  a key `concentrations`.
	- The concentrations are assumed to be comma-separated values.
	- If the number of wells does not match the number of concentrations, the function generates a default set 
	  of values ranging from 0 to the number of wells minus 1.
	- The resulting concentrations are converted to the specified `dtype` before being returned.

	Example
	-------
	>>> experiment = "/path/to/experiment"
	>>> concentrations = get_experiment_concentrations(experiment, dtype=float)
	>>> print(concentrations)
	[0.1, 0.2, 0.5, 1.0]
	"""
	
	config = get_config(experiment)
	wells = get_experiment_wells(experiment)
	nbr_of_wells = len(wells)

	concentrations = ConfigSectionMap(config, "Labels")["concentrations"].split(",")
	if nbr_of_wells != len(concentrations):
		concentrations = [str(s) for s in np.linspace(0, nbr_of_wells - 1, nbr_of_wells)]

	return np.array([dtype(c) for c in concentrations])


def get_experiment_cell_types(experiment, dtype=str):

	"""
	Retrieves the cell types associated with each well in an experiment.

	Parameters
	----------
	experiment : str
		The file system path to the experiment directory.
	dtype : type, optional
		The data type to which the cell types should be converted (default is `str`).

	Returns
	-------
	numpy.ndarray
		An array of cell types for each well, converted to the specified data type.

	Raises
	------
	AssertionError
		If the configuration file (`config.ini`) does not exist in the specified experiment directory.
	KeyError
		If the "cell_types" key is not found under the "Labels" section in the configuration file.
	ValueError
		If the retrieved cell types cannot be converted to the specified data type.

	Notes
	-----
	- The function retrieves the configuration file using `get_config()` and expects a section `Labels` containing 
	  a key `cell_types`.
	- The cell types are assumed to be comma-separated values.
	- If the number of wells does not match the number of cell types, the function generates a default set 
	  of values ranging from 0 to the number of wells minus 1.
	- The resulting cell types are converted to the specified `dtype` before being returned.

	Example
	-------
	>>> experiment = "/path/to/experiment"
	>>> cell_types = get_experiment_cell_types(experiment, dtype=str)
	>>> print(cell_types)
	['TypeA', 'TypeB', 'TypeC', 'TypeD']
	"""

	config = get_config(experiment)
	wells = get_experiment_wells(experiment)
	nbr_of_wells = len(wells)

	cell_types = ConfigSectionMap(config, "Labels")["cell_types"].split(",")
	if nbr_of_wells != len(cell_types):
		cell_types = [str(s) for s in np.linspace(0, nbr_of_wells - 1, nbr_of_wells)]

	return np.array([dtype(c) for c in cell_types])


def get_experiment_antibodies(experiment, dtype=str):
	
	"""
	Retrieve the list of antibodies used in an experiment.

	This function extracts antibody labels for the wells in the given experiment
	based on the configuration file. If the number of wells does not match the
	number of antibody labels provided in the configuration, it generates a
	sequence of default numeric labels.

	Parameters
	----------
	experiment : str
		The identifier or name of the experiment to retrieve antibodies for.
	dtype : type, optional
		The data type to which the antibody labels should be cast. Default is `str`.

	Returns
	-------
	numpy.ndarray
		An array of antibody labels with the specified data type. If no antibodies
		are specified or there is a mismatch, numeric labels are generated instead.

	Notes
	-----
	- The function assumes the experiment's configuration can be loaded using
	  `get_config` and that the antibodies are listed under the "Labels" section
	  with the key `"antibodies"`.
	- A mismatch between the number of wells and antibody labels will result in
	  numeric labels generated using `numpy.linspace`.

	Examples
	--------
	>>> get_experiment_antibodies("path/to/experiment1")
	array(['A1', 'A2', 'A3'], dtype='<U2')

	>>> get_experiment_antibodies("path/to/experiment2", dtype=int)
	array([0, 1, 2])
	"""

	config = get_config(experiment)
	wells = get_experiment_wells(experiment)
	nbr_of_wells = len(wells)

	antibodies = ConfigSectionMap(config, "Labels")["antibodies"].split(",")
	if nbr_of_wells != len(antibodies):
		antibodies = [str(s) for s in np.linspace(0, nbr_of_wells - 1, nbr_of_wells)]

	return np.array([dtype(c) for c in antibodies])


def get_experiment_pharmaceutical_agents(experiment, dtype=str):
	
	"""
	Retrieves the antibodies associated with each well in an experiment.

	Parameters
	----------
	experiment : str
		The file system path to the experiment directory.
	dtype : type, optional
		The data type to which the antibodies should be converted (default is `str`).

	Returns
	-------
	numpy.ndarray
		An array of antibodies for each well, converted to the specified data type.

	Raises
	------
	AssertionError
		If the configuration file (`config.ini`) does not exist in the specified experiment directory.
	KeyError
		If the "antibodies" key is not found under the "Labels" section in the configuration file.
	ValueError
		If the retrieved antibody values cannot be converted to the specified data type.

	Notes
	-----
	- The function retrieves the configuration file using `get_config()` and expects a section `Labels` containing 
	  a key `antibodies`.
	- The antibody names are assumed to be comma-separated values.
	- If the number of wells does not match the number of antibodies, the function generates a default set 
	  of values ranging from 0 to the number of wells minus 1.
	- The resulting antibody names are converted to the specified `dtype` before being returned.

	Example
	-------
	>>> experiment = "/path/to/experiment"
	>>> antibodies = get_experiment_antibodies(experiment, dtype=str)
	>>> print(antibodies)
	['AntibodyA', 'AntibodyB', 'AntibodyC', 'AntibodyD']
	"""

	config = get_config(experiment)
	wells = get_experiment_wells(experiment)
	nbr_of_wells = len(wells)

	pharmaceutical_agents = ConfigSectionMap(config, "Labels")["pharmaceutical_agents"].split(",")
	if nbr_of_wells != len(pharmaceutical_agents):
		pharmaceutical_agents = [str(s) for s in np.linspace(0, nbr_of_wells - 1, nbr_of_wells)]

	return np.array([dtype(c) for c in pharmaceutical_agents])


def interpret_wells_and_positions(experiment, well_option, position_option):
	"""
	Interpret well and position options for a given experiment.

	This function takes an experiment and well/position options to return the selected
	wells and positions. It supports selection of all wells or specific wells/positions
	as specified. The well numbering starts from 0 (i.e., Well 0 is W1 and so on).

	Parameters
	----------
	experiment : object
		The experiment object containing well information.
	well_option : str, int, or list of int
		The well selection option:
		- '*' : Select all wells.
		- int : Select a specific well by its index.
		- list of int : Select multiple wells by their indices.
	position_option : str, int, or list of int
		The position selection option:
		- '*' : Select all positions (returns None).
		- int : Select a specific position by its index.
		- list of int : Select multiple positions by their indices.

	Returns
	-------
	well_indices : numpy.ndarray or list of int
		The indices of the selected wells.
	position_indices : numpy.ndarray or list of int or None
		The indices of the selected positions. Returns None if all positions are selected.

	Examples
	--------
	>>> experiment = ...  # Some experiment object
	>>> interpret_wells_and_positions(experiment, '*', '*')
	(array([0, 1, 2, ..., n-1]), None)

	>>> interpret_wells_and_positions(experiment, 2, '*')
	([2], None)

	>>> interpret_wells_and_positions(experiment, [1, 3, 5], 2)
	([1, 3, 5], array([2]))

	"""

	wells = get_experiment_wells(experiment)
	nbr_of_wells = len(wells)

	if well_option == '*':
		well_indices = np.arange(nbr_of_wells)
	elif isinstance(well_option, int) or isinstance(well_option, np.int_):
		well_indices = [int(well_option)]
	elif isinstance(well_option, list):
		well_indices = well_option

	if position_option == '*':
		position_indices = None
	elif isinstance(position_option, int):
		position_indices = np.array([position_option], dtype=int)
	elif isinstance(position_option, list):
		position_indices = position_option

	return well_indices, position_indices


def extract_well_name_and_number(well):
	"""
	Extract the well name and number from a given well path.

	This function takes a well path string, splits it by the OS-specific path separator,
	and extracts the well name and number. The well name is the last component of the path,
	and the well number is derived by removing the 'W' prefix and converting the remaining
	part to an integer.

	Parameters
	----------
	well : str
		The well path string, where the well name is the last component.

	Returns
	-------
	well_name : str
		The name of the well, extracted from the last component of the path.
	well_number : int
		The well number, obtained by stripping the 'W' prefix from the well name
		and converting the remainder to an integer.

	Examples
	--------
	>>> well_path = "path/to/W23"
	>>> extract_well_name_and_number(well_path)
	('W23', 23)

	>>> well_path = "another/path/W1"
	>>> extract_well_name_and_number(well_path)
	('W1', 1)
	"""

	split_well_path = well.split(os.sep)
	split_well_path = list(filter(None, split_well_path))
	well_name = split_well_path[-1]
	well_number = int(split_well_path[-1].replace('W', ''))

	return well_name, well_number


def extract_position_name(pos):
	
	"""
	Extract the position name from a given position path.

	This function takes a position path string, splits it by the OS-specific path separator,
	filters out any empty components, and extracts the position name, which is the last
	component of the path.

	Parameters
	----------
	pos : str
		The position path string, where the position name is the last component.

	Returns
	-------
	pos_name : str
		The name of the position, extracted from the last component of the path.

	Examples
	--------
	>>> pos_path = "path/to/position1"
	>>> extract_position_name(pos_path)
	'position1'

	>>> pos_path = "another/path/positionA"
	>>> extract_position_name(pos_path)
	'positionA'
	"""

	split_pos_path = pos.split(os.sep)
	split_pos_path = list(filter(None, split_pos_path))
	pos_name = split_pos_path[-1]

	return pos_name


def get_position_table(pos, population, return_path=False):
	
	"""
	Retrieves the data table for a specified population at a given position, optionally returning the table's file path.

	This function locates and loads a CSV data table associated with a specific population (e.g., 'targets', 'cells')
	from a specified position directory. The position directory should contain an 'output/tables' subdirectory where
	the CSV file named 'trajectories_{population}.csv' is expected to be found. If the file exists, it is loaded into
	a pandas DataFrame; otherwise, None is returned.

	Parameters
	----------
	pos : str
		The path to the position directory from which to load the data table.
	population : str
		The name of the population for which the data table is to be retrieved. This name is used to construct the
		file name of the CSV file to be loaded.
	return_path : bool, optional
		If True, returns a tuple containing the loaded data table (or None) and the path to the CSV file. If False,
		only the loaded data table (or None) is returned (default is False).

	Returns
	-------
	pandas.DataFrame or None, or (pandas.DataFrame or None, str)
		If return_path is False, returns the loaded data table as a pandas DataFrame, or None if the table file does
		not exist. If return_path is True, returns a tuple where the first element is the data table (or None) and the
		second element is the path to the CSV file.

	Examples
	--------
	>>> df_pos = get_position_table('/path/to/position', 'targets')
	# This will load the 'trajectories_targets.csv' table from the specified position directory into a pandas DataFrame.

	>>> df_pos, table_path = get_position_table('/path/to/position', 'targets', return_path=True)
	# This will load the 'trajectories_targets.csv' table and also return the path to the CSV file.

	"""

	if not pos.endswith(os.sep):
		table = os.sep.join([pos, 'output', 'tables', f'trajectories_{population}.csv'])
	else:
		table = pos + os.sep.join(['output', 'tables', f'trajectories_{population}.csv'])

	if os.path.exists(table):
		df_pos = pd.read_csv(table, low_memory=False)
	else:
		df_pos = None
	
	if return_path:
		return df_pos, table
	else:
		return df_pos

def get_position_pickle(pos, population, return_path=False):

	"""
	Retrieves the data table for a specified population at a given position, optionally returning the table's file path.

	This function locates and loads a CSV data table associated with a specific population (e.g., 'targets', 'cells')
	from a specified position directory. The position directory should contain an 'output/tables' subdirectory where
	the CSV file named 'trajectories_{population}.csv' is expected to be found. If the file exists, it is loaded into
	a pandas DataFrame; otherwise, None is returned.

	Parameters
	----------
	pos : str
		The path to the position directory from which to load the data table.
	population : str
		The name of the population for which the data table is to be retrieved. This name is used to construct the
		file name of the CSV file to be loaded.
	return_path : bool, optional
		If True, returns a tuple containing the loaded data table (or None) and the path to the CSV file. If False,
		only the loaded data table (or None) is returned (default is False).

	Returns
	-------
	pandas.DataFrame or None, or (pandas.DataFrame or None, str)
		If return_path is False, returns the loaded data table as a pandas DataFrame, or None if the table file does
		not exist. If return_path is True, returns a tuple where the first element is the data table (or None) and the
		second element is the path to the CSV file.

	Examples
	--------
	>>> df_pos = get_position_table('/path/to/position', 'targets')
	# This will load the 'trajectories_targets.csv' table from the specified position directory into a pandas DataFrame.

	>>> df_pos, table_path = get_position_table('/path/to/position', 'targets', return_path=True)
	# This will load the 'trajectories_targets.csv' table and also return the path to the CSV file.

	"""

	if not pos.endswith(os.sep):
		table = os.sep.join([pos,'output','tables',f'trajectories_{population}.pkl'])
	else:
		table = pos + os.sep.join(['output','tables',f'trajectories_{population}.pkl'])

	if os.path.exists(table):
		df_pos = np.load(table, allow_pickle=True)
	else:
		df_pos = None

	if return_path:
		return df_pos, table
	else:
		return df_pos


def get_position_movie_path(pos, prefix=''):

	"""
	Get the path of the movie file for a given position.

	This function constructs the path to a movie file within a given position directory.
	It searches for TIFF files that match the specified prefix. If multiple matching files
	are found, the first one is returned.

	Parameters
	----------
	pos : str
		The directory path for the position.
	prefix : str, optional
		The prefix to filter movie files. Defaults to an empty string.

	Returns
	-------
	stack_path : str or None
		The path to the first matching movie file, or None if no matching file is found.

	Examples
	--------
	>>> pos_path = "path/to/position1"
	>>> get_position_movie_path(pos_path, prefix='experiment_')
	'path/to/position1/movie/experiment_001.tif'

	>>> pos_path = "another/path/positionA"
	>>> get_position_movie_path(pos_path)
	'another/path/positionA/movie/001.tif'

	>>> pos_path = "nonexistent/path"
	>>> get_position_movie_path(pos_path)
	None
	"""
	

	if not pos.endswith(os.sep):
		pos += os.sep
	movies = glob(pos + os.sep.join(['movie', prefix + '*.tif']))
	if len(movies) > 0:
		stack_path = movies[0]
	else:
		stack_path = None

	return stack_path


def load_experiment_tables(experiment, population='targets', well_option='*', position_option='*',
						   return_pos_info=False, load_pickle=False):

	"""
	Load tabular data for an experiment, optionally including position-level information.

	This function retrieves and processes tables associated with positions in an experiment.
	It supports filtering by wells and positions, and can load either CSV data or pickle files.

	Parameters
	----------
	experiment : str
		Path to the experiment folder to load data for.
	population : str, optional
		The population to extract from the position tables (`'targets'` or `'effectors'`). Default is `'targets'`.
	well_option : str or list, optional
		Specifies which wells to include. Default is `'*'`, meaning all wells.
	position_option : str or list, optional
		Specifies which positions to include within selected wells. Default is `'*'`, meaning all positions.
	return_pos_info : bool, optional
		If `True`, also returns a DataFrame containing position-level metadata. Default is `False`.
	load_pickle : bool, optional
		If `True`, loads pre-processed pickle files for the positions instead of raw data. Default is `False`.

	Returns
	-------
	df : pandas.DataFrame or None
		A DataFrame containing aggregated data for the specified wells and positions, or `None` if no data is found.
		The DataFrame includes metadata such as well and position identifiers, concentrations, antibodies, and other
		experimental parameters.
	df_pos_info : pandas.DataFrame, optional
		A DataFrame with metadata for each position, including file paths and experimental details. Returned only
		if `return_pos_info=True`.

	Notes
	-----
	- The function assumes the experiment's configuration includes details about movie prefixes, concentrations,
	  cell types, antibodies, and pharmaceutical agents.
	- Wells and positions can be filtered using `well_option` and `position_option`, respectively. If filtering
	  fails or is invalid, those specific wells/positions are skipped.
	- Position-level metadata is assembled into `df_pos_info` and includes paths to data and movies.

	Examples
	--------
	Load all data for an experiment:

	>>> df = load_experiment_tables("path/to/experiment1")

	Load data for specific wells and positions, including position metadata:

	>>> df, df_pos_info = load_experiment_tables(
	...     "experiment_01", well_option=["A1", "B1"], position_option=[0, 1], return_pos_info=True
	... )

	Use pickle files for faster loading:

	>>> df = load_experiment_tables("experiment_01", load_pickle=True)
	"""

	config = get_config(experiment)
	wells = get_experiment_wells(experiment)

	movie_prefix = ConfigSectionMap(config, "MovieSettings")["movie_prefix"]
	concentrations = get_experiment_concentrations(experiment, dtype=float)
	cell_types = get_experiment_cell_types(experiment)
	antibodies = get_experiment_antibodies(experiment)
	pharmaceutical_agents = get_experiment_pharmaceutical_agents(experiment)
	metadata = get_experiment_metadata(experiment) # None or dict of metadata
	well_labels = _extract_labels_from_config(config, len(wells))

	well_indices, position_indices = interpret_wells_and_positions(experiment, well_option, position_option)

	df = []
	df_pos_info = []
	real_well_index = 0

	for k, well_path in enumerate(tqdm(wells[well_indices])):

		any_table = False  # assume no table

		well_name, well_number = extract_well_name_and_number(well_path)
		widx = well_indices[k]

		well_alias = well_labels[widx]

		well_concentration = concentrations[widx]
		well_antibody = antibodies[widx]
		well_cell_type = cell_types[widx]
		well_pharmaceutical_agent = pharmaceutical_agents[widx]

		positions = get_positions_in_well(well_path)
		if position_indices is not None:
			try:
				positions = positions[position_indices]
			except Exception as e:
				print(e)
				continue

		real_pos_index = 0
		for pidx, pos_path in enumerate(positions):

			pos_name = extract_position_name(pos_path)

			stack_path = get_position_movie_path(pos_path, prefix=movie_prefix)

			if not load_pickle:
				df_pos, table = get_position_table(pos_path, population=population, return_path=True)
			else:
				df_pos, table = get_position_pickle(pos_path, population=population, return_path=True)

			if df_pos is not None:
			
				df_pos['position'] = pos_path
				df_pos['well'] = well_path
				df_pos['well_index'] = well_number
				df_pos['well_name'] = well_name
				df_pos['pos_name'] = pos_name

				df_pos['concentration'] = well_concentration
				df_pos['antibody'] = well_antibody
				df_pos['cell_type'] = well_cell_type
				df_pos['pharmaceutical_agent'] = well_pharmaceutical_agent
				if metadata is not None:
					keys = list(metadata.keys())
					for k in keys:
						df_pos[k] = metadata[k]

				df.append(df_pos)
				any_table = True

				pos_dict = {'pos_path': pos_path, 'pos_index': real_pos_index, 'pos_name': pos_name, 'table_path': table,
					 'stack_path': stack_path,'well_path': well_path, 'well_index': real_well_index, 'well_name': well_name,
					 'well_number': well_number, 'well_alias': well_alias}
				# if metadata is not None:
				# 	keys = list(metadata.keys())
				# 	for k in keys:
				# 		pos_dict.update({k: metadata[k]})

				df_pos_info.append(pos_dict)

				real_pos_index += 1

		if any_table:
			real_well_index += 1

	df_pos_info = pd.DataFrame(df_pos_info)
	if len(df) > 0:
		df = pd.concat(df)
		df = df.reset_index(drop=True)
	else:
		df = None

	if return_pos_info:
		return df, df_pos_info
	else:
		return df


def locate_stack(position, prefix='Aligned'):

	"""

	Locate and load a stack of images.

	Parameters
	----------
	position : str
		The position folder within the well where the stack is located.
	prefix : str, optional
		The prefix used to identify the stack. The default is 'Aligned'.

	Returns
	-------
	stack : ndarray
		The loaded stack as a NumPy array.

	Raises
	------
	AssertionError
		If no stack with the specified prefix is found.

	Notes
	-----
	This function locates and loads a stack of images based on the specified position and prefix.
	It assumes that the stack is stored in a directory named 'movie' within the specified position.
	The function loads the stack as a NumPy array and performs shape manipulation to have the channels
	at the end.

	Examples
	--------
	>>> stack = locate_stack(position, prefix='Aligned')
	# Locate and load a stack of images for further processing.

	"""

	if not position.endswith(os.sep):
		position += os.sep

	stack_path = glob(position + os.sep.join(['movie', f'{prefix}*.tif']))
	assert len(stack_path) > 0, f"No movie with prefix {prefix} found..."
	stack = imread(stack_path[0].replace('\\', '/'))
	stack_length = auto_load_number_of_frames(stack_path[0])

	if stack.ndim == 4:
		stack = np.moveaxis(stack, 1, -1)
	elif stack.ndim == 3:
		if min(stack.shape)!=stack_length:
			channel_axis = np.argmin(stack.shape)
			if channel_axis!=(stack.ndim-1):
				stack = np.moveaxis(stack, channel_axis, -1)
			stack = stack[np.newaxis, :, :, :]
		else:
			stack = stack[:, :, :, np.newaxis]
	elif stack.ndim==2:
		stack = stack[np.newaxis, :, :, np.newaxis]

	return stack

def locate_labels(position, population='target', frames=None):

	"""
	Locate and load label images for a given position and population in an experiment.

	This function retrieves and optionally loads labeled images (e.g., targets or effectors) 
	for a specified position in an experiment. It supports loading all frames, a specific 
	frame, or a list of frames.

	Parameters
	----------
	position : str
		Path to the position directory containing label images.
	population : str, optional
		The population to load labels for. Options are `'target'` (or `'targets'`) and 
		`'effector'` (or `'effectors'`). Default is `'target'`.
	frames : int, list of int, numpy.ndarray, or None, optional
		Specifies which frames to load:
		- `None`: Load all frames (default).
		- `int`: Load a single frame, identified by its index.
		- `list` or `numpy.ndarray`: Load multiple specific frames.

	Returns
	-------
	numpy.ndarray or list of numpy.ndarray
		If `frames` is `None` or a single integer, returns a NumPy array of the corresponding 
		labels. If `frames` is a list or array, returns a list of NumPy arrays for each frame. 
		If a frame is not found, `None` is returned for that frame.

	Notes
	-----
	- The function assumes label images are stored in subdirectories named `"labels_targets"` 
	  or `"labels_effectors"`, with filenames formatted as `####.tif` (e.g., `0001.tif`).
	- Frame indices are zero-padded to four digits for matching.
	- If `frames` is invalid or a frame is not found, `None` is returned for that frame.

	Examples
	--------
	Load all label images for a position:

	>>> labels = locate_labels("/path/to/position", population="target")

	Load a single frame (frame index 3):

	>>> label = locate_labels("/path/to/position", population="effector", frames=3)

	Load multiple specific frames:

	>>> labels = locate_labels("/path/to/position", population="target", frames=[0, 1, 2])
	"""

	if not position.endswith(os.sep):
		position += os.sep

	if population.lower() == "target" or population.lower() == "targets":
		label_path = natsorted(glob(position + os.sep.join(["labels_targets", "*.tif"])))
	elif population.lower() == "effector" or population.lower() == "effectors":
		label_path = natsorted(glob(position + os.sep.join(["labels_effectors", "*.tif"])))

	label_names = [os.path.split(lbl)[-1] for lbl in label_path]

	if frames is None:

		labels = np.array([imread(i.replace('\\', '/')) for i in label_path])
	
	elif isinstance(frames, (int,float, np.int_)):

		tzfill = str(int(frames)).zfill(4)
		try:
			idx = label_names.index(f"{tzfill}.tif")
		except:
			idx = -1

		if idx==-1:
			labels = None
		else:
			labels = np.array(imread(label_path[idx].replace('\\', '/')))

	elif isinstance(frames, (list,np.ndarray)):
		labels = []
		for f in frames:
			tzfill = str(int(f)).zfill(4)
			try:
				idx = label_names.index(f"{tzfill}.tif")
			except:
				idx = -1
			
			if idx==-1:
				labels.append(None)
			else:
				labels.append(np.array(imread(label_path[idx].replace('\\', '/'))))
	else:
		print('Frames argument must be None, int or list...')

	return labels

def fix_missing_labels(position, population='target', prefix='Aligned'):
	
	"""
	Fix missing label files by creating empty label images for frames that do not have corresponding label files.

	This function locates missing label files in a sequence of frames and creates empty labels (filled with zeros) 
	for the frames that are missing. The function works for two types of populations: 'target' or 'effector'.

	Parameters
	----------
	position : str
		The file path to the folder containing the images/label files. This is the root directory where 
		the label files are expected to be found.
	population : str, optional
		Specifies whether to look for 'target' or 'effector' labels. Accepts 'target' or 'effector' 
		as valid values. Default is 'target'.
	prefix : str, optional
		The prefix used to locate the image stack (default is 'Aligned').

	Returns
	-------
	None
		The function creates new label files in the corresponding folder for any frames missing label files.
	"""

	if not position.endswith(os.sep):
		position += os.sep

	stack = locate_stack(position, prefix=prefix)
	template = np.zeros((stack[0].shape[0], stack[0].shape[1]),dtype=int)
	all_frames = np.arange(len(stack))

	if population.lower() == "target" or population.lower() == "targets":
		label_path = natsorted(glob(position + os.sep.join(["labels_targets", "*.tif"])))
		path = position + os.sep + "labels_targets"
	elif population.lower() == "effector" or population.lower() == "effectors":
		label_path = natsorted(glob(position + os.sep.join(["labels_effectors", "*.tif"])))
		path = position + os.sep + "labels_effectors"

	if label_path!=[]:
		#path = os.path.split(label_path[0])[0]
		int_valid = [int(lbl.split(os.sep)[-1].split('.')[0]) for lbl in label_path]
		to_create = [x for x in all_frames if x not in int_valid]
	else:
		to_create = all_frames
	to_create = [str(x).zfill(4)+'.tif' for x in to_create]
	for file in to_create:
		save_tiff_imagej_compatible(os.sep.join([path, file]), template.astype(np.int16), axes='YX')
		#imwrite(os.sep.join([path, file]), template.astype(int))


def locate_stack_and_labels(position, prefix='Aligned', population="target"):

	"""

	Locate and load the stack and corresponding segmentation labels.

	Parameters
	----------
	position : str
		The position or directory path where the stack and labels are located.
	prefix : str, optional
		The prefix used to identify the stack. The default is 'Aligned'.
	population : str, optional
		The population for which the segmentation must be located. The default is 'target'.

	Returns
	-------
	stack : ndarray
		The loaded stack as a NumPy array.
	labels : ndarray
		The loaded segmentation labels as a NumPy array.

	Raises
	------
	AssertionError
		If no stack with the specified prefix is found or if the shape of the stack and labels do not match.

	Notes
	-----
	This function locates the stack and corresponding segmentation labels based on the specified position and population.
	It assumes that the stack and labels are stored in separate directories: 'movie' for the stack and 'labels' or 'labels_effectors' for the labels.
	The function loads the stack and labels as NumPy arrays and performs shape validation.

	Examples
	--------
	>>> stack, labels = locate_stack_and_labels(position, prefix='Aligned', population="target")
	# Locate and load the stack and segmentation labels for further processing.

	"""

	position = position.replace('\\', '/')
	labels = locate_labels(position, population=population)
	stack = locate_stack(position, prefix=prefix)
	if len(labels) < len(stack):
		fix_missing_labels(position, population=population, prefix=prefix)
		labels = locate_labels(position, population=population)
	assert len(stack) == len(
		labels), f"The shape of the stack {stack.shape} does not match with the shape of the labels {labels.shape}"

	return stack, labels

def load_tracking_data(position, prefix="Aligned", population="target"):
	"""

	Load the tracking data, labels, and stack for a given position and population.

	Parameters
	----------
	position : str
		The position or directory where the data is located.
	prefix : str, optional
		The prefix used in the filenames of the stack images (default is "Aligned").
	population : str, optional
		The population to load the data for. Options are "target" or "effector" (default is "target").

	Returns
	-------
	trajectories : DataFrame
		The tracking data loaded as a pandas DataFrame.
	labels : ndarray
		The segmentation labels loaded as a numpy ndarray.
	stack : ndarray
		The image stack loaded as a numpy ndarray.

	Notes
	-----
	This function loads the tracking data, labels, and stack for a given position and population.
	It reads the trajectories from the appropriate CSV file based on the specified population.
	The stack and labels are located using the `locate_stack_and_labels` function.
	The resulting tracking data is returned as a pandas DataFrame, and the labels and stack are returned as numpy ndarrays.

	Examples
	--------
	>>> trajectories, labels, stack = load_tracking_data(position, population="target")
	# Load the tracking data, labels, and stack for the specified position and target population.

	"""

	position = position.replace('\\', '/')
	if population.lower() == "target" or population.lower() == "targets":
		trajectories = pd.read_csv(position + os.sep.join(['output', 'tables', 'trajectories_targets.csv']))
	elif population.lower() == "effector" or population.lower() == "effectors":
		trajectories = pd.read_csv(position + os.sep.join(['output', 'tables', 'trajectories_effectors.csv']))

	stack, labels = locate_stack_and_labels(position, prefix=prefix, population=population)

	return trajectories, labels, stack


def auto_load_number_of_frames(stack_path):
	
	"""
	Automatically determine the number of frames in a TIFF image stack.

	This function extracts the number of frames (time slices) from the metadata of a TIFF file 
	or infers it from the stack dimensions when metadata is unavailable. It is robust to 
	variations in metadata structure and handles multi-channel images.

	Parameters
	----------
	stack_path : str
		Path to the TIFF image stack file.

	Returns
	-------
	int or None
		The number of frames in the image stack. Returns `None` if the path is `None` 
		or the frame count cannot be determined.

	Notes
	-----
	- The function attempts to extract the `frames` or `slices` attributes from the 
	  TIFF metadata, specifically the `ImageDescription` tag.
	- If metadata extraction fails, the function reads the image stack and infers 
	  the number of frames based on the stack dimensions.
	- Multi-channel stacks are handled by assuming the number of channels is specified 
	  in the metadata under the `channels` attribute.

	Examples
	--------
	Automatically detect the number of frames in a TIFF stack:

	>>> frames = auto_load_number_of_frames("experiment_stack.tif")
	Automatically detected stack length: 120...

	Handle a single-frame TIFF:

	>>> frames = auto_load_number_of_frames("single_frame_stack.tif")
	Automatically detected stack length: 1...

	Handle invalid or missing paths gracefully:

	>>> frames = auto_load_number_of_frames(None)
	>>> print(frames)
	None
	"""

	if stack_path is None:
		return None

	stack_path = stack_path.replace('\\','/')
	n_channels=1

	with TiffFile(stack_path) as tif:
		try:
			tif_tags = {}
			for tag in tif.pages[0].tags.values():
				name, value = tag.name, tag.value
				tif_tags[name] = value
			img_desc = tif_tags["ImageDescription"]
			attr = img_desc.split("\n")
			n_channels = int(attr[np.argmax([s.startswith("channels") for s in attr])].split("=")[-1])
		except Exception as e:
			pass
		try:
			# Try nframes
			nslices = int(attr[np.argmax([s.startswith("frames") for s in attr])].split("=")[-1])
			if nslices > 1:
				len_movie = nslices
			else:
				break_the_code()
		except:
			try:
				# try nslices
				frames = int(attr[np.argmax([s.startswith("slices") for s in attr])].split("=")[-1])
				len_movie = frames
			except:
				pass

	try:
		del tif;
		del tif_tags;
		del img_desc;
	except:
		pass

	if 'len_movie' not in locals():
		stack = imread(stack_path)
		len_movie = len(stack)
		if len_movie==n_channels and stack.ndim==3:
			len_movie = 1
		if stack.ndim==2:
			len_movie = 1
		del stack
	gc.collect()

	print(f'Automatically detected stack length: {len_movie}...')

	return len_movie if 'len_movie' in locals() else None


def parse_isotropic_radii(string):

	"""
	Parse a string representing isotropic radii into a structured list.

	This function extracts integer values and ranges (denoted by square brackets) 
	from a string input and returns them as a list. Single values are stored as integers, 
	while ranges are represented as lists of two integers.

	Parameters
	----------
	string : str
		The input string containing radii and ranges, separated by commas or spaces. 
		Ranges should be enclosed in square brackets, e.g., `[1 2]`.

	Returns
	-------
	list
		A list of parsed radii where:
		- Single integers are included as `int`.
		- Ranges are included as two-element lists `[start, end]`.

	Examples
	--------
	Parse a string with single radii and ranges:

	>>> parse_isotropic_radii("1, [2 3], 4")
	[1, [2, 3], 4]

	Handle inputs with mixed delimiters:

	>>> parse_isotropic_radii("5 [6 7], 8")
	[5, [6, 7], 8]

	Notes
	-----
	- The function splits the input string by commas or spaces.
	- It identifies ranges using square brackets and assumes that ranges are always 
	  two consecutive values.
	- Non-integer sections of the string are ignored.
	"""

	sections = re.split(',| ', string)
	radii = []
	for k, s in enumerate(sections):
		if s.isdigit():
			radii.append(int(s))
		if '[' in s:
			ring = [int(s.replace('[', '')), int(sections[k + 1].replace(']', ''))]
			radii.append(ring)
		else:
			pass
	return radii


def get_tracking_configs_list(return_path=False):
	"""

	Retrieve a list of available tracking configurations.

	Parameters
	----------
	return_path : bool, optional
		If True, also returns the path to the models. Default is False.

	Returns
	-------
	list or tuple
		If return_path is False, returns a list of available tracking configurations.
		If return_path is True, returns a tuple containing the list of models and the path to the models.

	Notes
	-----
	This function retrieves the list of available tracking configurations by searching for model directories
	in the predefined model path. The model path is derived from the parent directory of the current script
	location and the path to the model directory. By default, it returns only the names of the models.
	If return_path is set to True, it also returns the path to the models.

	Examples
	--------
	>>> models = get_tracking_configs_list()
	# Retrieve a list of available tracking configurations.

	>>> models, path = get_tracking_configs_list(return_path=True)
	# Retrieve a list of available tracking configurations.

	"""

	modelpath = os.sep.join(
		[os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective", "models", "tracking_configs",
		 os.sep])
	available_models = glob(modelpath + '*.json')
	available_models = [m.replace('\\', '/').split('/')[-1] for m in available_models]
	available_models = [m.replace('\\', '/').split('.')[0] for m in available_models]

	if not return_path:
		return available_models
	else:
		return available_models, modelpath


def interpret_tracking_configuration(config):
	
	"""
	Interpret and resolve the path for a tracking configuration file.

	This function determines the appropriate configuration file path based on the input.
	If the input is a string representing an existing path or a known configuration name, 
	it resolves to the correct file path. If the input is invalid or `None`, a default 
	configuration is returned.

	Parameters
	----------
	config : str or None
		The input configuration, which can be:
		- A string representing the full path to a configuration file.
		- A short name of a configuration file without the `.json` extension.
		- `None` to use a default configuration.

	Returns
	-------
	str
		The resolved path to the configuration file.

	Notes
	-----
	- If `config` is a string and the specified path exists, it is returned as-is.
	- If `config` is a name, the function searches in the `tracking_configs` directory 
	  within the `celldetective` models folder.
	- If the file or name is not found, or if `config` is `None`, the function falls 
	  back to a default configuration using `cell_config()`.

	Examples
	--------
	Resolve a full path:

	>>> interpret_tracking_configuration("/path/to/config.json")
	'/path/to/config.json'

	Resolve a named configuration:

	>>> interpret_tracking_configuration("default_tracking")
	'/path/to/celldetective/models/tracking_configs/default_tracking.json'

	Handle `None` to return the default configuration:

	>>> interpret_tracking_configuration(None)
	'/path/to/default/config.json'
	"""

	if isinstance(config, str):
		if os.path.exists(config):
			return config
		else:
			modelpath = os.sep.join(
				[os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective", "models",
				 "tracking_configs", os.sep])
			if os.path.exists(modelpath + config + '.json'):
				return modelpath + config + '.json'
			else:
				config = cell_config()
	elif config is None:
		config = cell_config()

	return config


def get_signal_models_list(return_path=False):

	"""

	Retrieve a list of available signal detection models.

	Parameters
	----------
	return_path : bool, optional
		If True, also returns the path to the models. Default is False.

	Returns
	-------
	list or tuple
		If return_path is False, returns a list of available signal detection models.
		If return_path is True, returns a tuple containing the list of models and the path to the models.

	Notes
	-----
	This function retrieves the list of available signal detection models by searching for model directories
	in the predefined model path. The model path is derived from the parent directory of the current script
	location and the path to the model directory. By default, it returns only the names of the models.
	If return_path is set to True, it also returns the path to the models.

	Examples
	--------
	>>> models = get_signal_models_list()
	# Retrieve a list of available signal detection models.

	>>> models, path = get_signal_models_list(return_path=True)
	# Retrieve a list of available signal detection models and the path to the models.

	"""

	modelpath = os.sep.join(
		[os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective", "models", "signal_detection",
		 os.sep])
	repository_models = get_zenodo_files(cat=os.sep.join(["models", "signal_detection"]))

	available_models = glob(modelpath + f'*{os.sep}')
	available_models = [m.replace('\\', '/').split('/')[-2] for m in available_models]
	for rm in repository_models:
		if rm not in available_models:
			available_models.append(rm)

	if not return_path:
		return available_models
	else:
		return available_models, modelpath

def get_pair_signal_models_list(return_path=False):
	"""

	Retrieve a list of available signal detection models.

	Parameters
	----------
	return_path : bool, optional
		If True, also returns the path to the models. Default is False.

	Returns
	-------
	list or tuple
		If return_path is False, returns a list of available signal detection models.
		If return_path is True, returns a tuple containing the list of models and the path to the models.

	Notes
	-----
	This function retrieves the list of available signal detection models by searching for model directories
	in the predefined model path. The model path is derived from the parent directory of the current script
	location and the path to the model directory. By default, it returns only the names of the models.
	If return_path is set to True, it also returns the path to the models.

	Examples
	--------
	>>> models = get_signal_models_list()
	# Retrieve a list of available signal detection models.

	>>> models, path = get_signal_models_list(return_path=True)
	# Retrieve a list of available signal detection models and the path to the models.

	"""

	modelpath = os.sep.join(
		[os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective", "models", "pair_signal_detection",
		 os.sep])
	#repository_models = get_zenodo_files(cat=os.sep.join(["models", "pair_signal_detection"]))

	available_models = glob(modelpath + f'*{os.sep}')
	available_models = [m.replace('\\', '/').split('/')[-2] for m in available_models]
	#for rm in repository_models:
	#   if rm not in available_models:
	#       available_models.append(rm)

	if not return_path:
		return available_models
	else:
		return available_models, modelpath


def locate_signal_model(name, path=None, pairs=False):

	"""
	Locate a signal detection model by name, either locally or from Zenodo.

	This function searches for a signal detection model with the specified name in the local
	`celldetective` directory. If the model is not found locally, it attempts to download
	the model from Zenodo.

	Parameters
	----------
	name : str
		The name of the signal detection model to locate.
	path : str, optional
		An additional directory path to search for the model. If provided, this directory
		is also scanned for matching models. Default is `None`.
	pairs : bool, optional
		If `True`, searches for paired signal detection models in the `pair_signal_detection`
		subdirectory. If `False`, searches in the `signal_detection` subdirectory. Default is `False`.

	Returns
	-------
	str or None
		The full path to the located model directory if found, or `None` if the model is not available
		locally or on Zenodo.

	Notes
	-----
	- The function first searches in the `celldetective/models/signal_detection` or 
	  `celldetective/models/pair_signal_detection` directory based on the `pairs` argument.
	- If a `path` is specified, it is searched in addition to the default directories.
	- If the model is not found locally, the function queries Zenodo for the model. If available,
	  the model is downloaded to the appropriate `celldetective` subdirectory.

	Examples
	--------
	Search for a signal detection model locally:

	>>> locate_signal_model("example_model")
	'path/to/celldetective/models/signal_detection/example_model/'

	Search for a paired signal detection model:

	>>> locate_signal_model("paired_model", pairs=True)
	'path/to/celldetective/models/pair_signal_detection/paired_model/'

	Include an additional search path:

	>>> locate_signal_model("custom_model", path="/additional/models/")
	'/additional/models/custom_model/'

	Handle a model available only on Zenodo:

	>>> locate_signal_model("remote_model")
	'path/to/celldetective/models/signal_detection/remote_model/'
	"""

	main_dir = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"])
	modelpath = os.sep.join([main_dir, "models", "signal_detection", os.sep])
	if pairs:
		modelpath = os.sep.join([main_dir, "models", "pair_signal_detection", os.sep])
	print(f'Looking for {name} in {modelpath}')
	models = glob(modelpath + f'*{os.sep}')
	if path is not None:
		if not path.endswith(os.sep):
			path += os.sep
		models += glob(path + f'*{os.sep}')

	match = None
	for m in models:
		if name == m.replace('\\', os.sep).split(os.sep)[-2]:
			match = m
			return match
	# else no match, try zenodo
	files, categories = get_zenodo_files()
	if name in files:
		index = files.index(name)
		cat = categories[index]
		download_zenodo_file(name, os.sep.join([main_dir, cat]))
		match = os.sep.join([main_dir, cat, name]) + os.sep
	return match

def locate_pair_signal_model(name, path=None):

	"""
	Locate a pair signal detection model by name.

	This function searches for a pair signal detection model in the default 
	`celldetective` directory and optionally in an additional user-specified path.

	Parameters
	----------
	name : str
		The name of the pair signal detection model to locate.
	path : str, optional
		An additional directory path to search for the model. If provided, this directory 
		is also scanned for matching models. Default is `None`.

	Returns
	-------
	str or None
		The full path to the located model directory if found, or `None` if no matching 
		model is located.

	Notes
	-----
	- The function first searches in the default `celldetective/models/pair_signal_detection` 
	  directory.
	- If a `path` is specified, it is searched in addition to the default directory.
	- The function prints the search path and model name during execution.

	Examples
	--------
	Locate a model in the default directory:

	>>> locate_pair_signal_model("example_model")
	'path/to/celldetective/models/pair_signal_detection/example_model/'

	Include an additional search directory:

	>>> locate_pair_signal_model("custom_model", path="/additional/models/")
	'/additional/models/custom_model/'
	"""


	main_dir = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"])
	modelpath = os.sep.join([main_dir, "models", "pair_signal_detection", os.sep])
	print(f'Looking for {name} in {modelpath}')
	models = glob(modelpath + f'*{os.sep}')
	if path is not None:
		if not path.endswith(os.sep):
			path += os.sep
		models += glob(path + f'*{os.sep}')

def relabel_segmentation(labels, df, exclude_nans=True, column_labels={'track': "TRACK_ID", 'frame': 'FRAME', 'y': 'POSITION_Y', 'x': 'POSITION_X', 'label': 'class_id'}, threads=1):

	"""
	Relabel the segmentation labels with the tracking IDs from the tracks.

	The function reassigns the mask value for each cell with the associated `TRACK_ID`, if it exists
	in the trajectory table (`df`). If no track uses the cell mask, a new track with a single point
	is generated on the fly (max of `TRACK_ID` values + i, for i=0 to N such cells). It supports 
	multithreaded processing for faster execution on large datasets.

	Parameters
	----------
	labels : np.ndarray
		A (TYX) array where each frame contains a 2D segmentation mask. Each unique 
		non-zero integer represents a labeled object.
	df : pandas.DataFrame
		A DataFrame containing tracking information with columns 
		specified in `column_labels`. Must include at least frame, track ID, and object ID.
	exclude_nans : bool, optional
		Whether to exclude rows in `df` with NaN values in the column specified by 
		`column_labels['label']`. Default is `True`.
	column_labels : dict, optional
		A dictionary specifying the column names in `df`. Default is:
		- `'track'`: Track ID column name (`"TRACK_ID"`)
		- `'frame'`: Frame column name (`"FRAME"`)
		- `'y'`: Y-coordinate column name (`"POSITION_Y"`)
		- `'x'`: X-coordinate column name (`"POSITION_X"`)
		- `'label'`: Object ID column name (`"class_id"`)
	threads : int, optional
		Number of threads to use for multithreaded processing. Default is `1`.

	Returns
	-------
	np.ndarray
		A new (TYX) array with the same shape as `labels`, where objects are relabeled 
		according to their tracking identity in `df`.

	Notes
	-----
	- For frames where labeled objects in `labels` do not match any entries in the `df`,
	  new track IDs are generated for the unmatched labels.
	- The relabeling process maintains synchronization across threads using a shared
	  counter for generating unique track IDs.

	Examples
	--------
	Relabel segmentation using tracking data:

	>>> labels = np.random.randint(0, 5, (10, 100, 100))
	>>> df = pd.DataFrame({
	...     "TRACK_ID": [1, 2, 1, 2],
	...     "FRAME": [0, 0, 1, 1],
	...     "class_id": [1, 2, 1, 2],
	... })
	>>> new_labels = relabel_segmentation(labels, df, threads=2)
	Done.

	Use custom column labels and exclude rows with NaNs:

	>>> column_labels = {
	...     'track': "track_id",
	...     'frame': "time",
	...     'label': "object_id"
	... }
	>>> new_labels = relabel_segmentation(labels, df, column_labels=column_labels, exclude_nans=True)
	Done.
	"""

	n_threads = threads
	df = df.sort_values(by=[column_labels['track'],column_labels['frame']])
	if exclude_nans:
		df = df.dropna(subset=column_labels['label'])

	new_labels = np.zeros_like(labels)
	shared_data = {"s": 0}

	def rewrite_labels(indices):

		all_track_ids = df[column_labels['track']].unique()
		
		for t in tqdm(indices):

			f = int(t)
			cells = df.loc[df[column_labels['frame']] == f, [column_labels['track'], column_labels['label']]].to_numpy()
			tracks_at_t = list(cells[:,0])
			identities = list(cells[:,1])

			labels_at_t = list(np.unique(labels[f]))
			if 0 in labels_at_t:
				labels_at_t.remove(0)
			labels_not_in_df = [lbl for lbl in labels_at_t if lbl not in identities]
			for lbl in labels_not_in_df:
				with threading.Lock():  # Synchronize access to `shared_data["s"]`
					track_id = max(all_track_ids) + shared_data["s"]
					shared_data["s"] += 1
				tracks_at_t.append(track_id)
				identities.append(lbl)

			# exclude NaN
			tracks_at_t = np.array(tracks_at_t)
			identities = np.array(identities)

			tracks_at_t = tracks_at_t[identities == identities]
			identities = identities[identities == identities]

			for k in range(len(identities)):

				# need routine to check values from labels not in class_id of this frame and add new track id
				
				loc_i, loc_j = np.where(labels[f] == identities[k])
				track_id = tracks_at_t[k]
				new_labels[f, loc_i, loc_j] = round(track_id)

	# Multithreading
	indices = list(df[column_labels['frame']].unique())
	chunks = np.array_split(indices, n_threads)

	with concurrent.futures.ThreadPoolExecutor() as executor:
		executor.map(rewrite_labels, chunks)
	
	print("\nDone.")

	return new_labels


def control_tracks(position, prefix="Aligned", population="target", relabel=True, flush_memory=True, threads=1):

	"""
	Controls the tracking of cells or objects within a given position by locating the relevant image stack and label data,
	and then visualizing and managing the tracks in the Napari viewer.

	Parameters
	----------
	position : str
		The path to the directory containing the position's data. The function will ensure the path uses forward slashes.
	
	prefix : str, optional, default="Aligned"
		The prefix of the file names for the image stack and labels. This parameter helps locate the relevant data files.
	
	population : str, optional, default="target"
		The population to be tracked, typically either "target" or "effectors". This is used to identify the group of interest for tracking.
	
	relabel : bool, optional, default=True
		If True, will relabel the tracks, potentially assigning new track IDs to the detected objects.
	
	flush_memory : bool, optional, default=True
		If True, will flush memory after processing to free up resources.
	
	threads : int, optional, default=1
		The number of threads to use for processing. This can speed up the task in multi-threaded environments.

	Returns
	-------
	None
		The function performs visualization and management of tracks in the Napari viewer. It does not return any value.

	Notes
	-----
	- This function assumes that the necessary data for tracking (stack and labels) are located in the specified position directory.
	- The `locate_stack_and_labels` function is used to retrieve the image stack and labels from the specified directory.
	- The tracks are visualized using the `view_tracks_in_napari` function, which handles the display in the Napari viewer.
	- The function can be used for tracking biological entities (e.g., cells) and their movement across time frames in an image stack.
	
	Example
	-------
	>>> control_tracks("/path/to/data/position_1", prefix="Aligned", population="target", relabel=True, flush_memory=True, threads=4)
	"""
	
	if not position.endswith(os.sep):
		position += os.sep

	position = position.replace('\\','/')
	stack, labels = locate_stack_and_labels(position, prefix=prefix, population=population)

	view_tracks_in_napari(position, population, labels=labels, stack=stack, relabel=relabel,
						  flush_memory=flush_memory, threads=threads)


def tracks_to_btrack(df, exclude_nans=False):
	
	"""
	Converts a dataframe of tracked objects into the bTrack output format.
	The function prepares tracking data, properties, and an empty graph structure for further processing.

	Parameters
	----------
	df : pandas.DataFrame
		A dataframe containing tracking information. The dataframe must have columns for `TRACK_ID`, 
		`FRAME`, `POSITION_Y`, `POSITION_X`, and `class_id` (among others).

	exclude_nans : bool, optional, default=False
		If True, rows with NaN values in the `class_id` column will be excluded from the dataset.
		If False, the dataframe will retain all rows, including those with NaN in `class_id`.

	Returns
	-------
	data : numpy.ndarray
		A 2D numpy array containing the tracking data with columns `[TRACK_ID, FRAME, z, POSITION_Y, POSITION_X]`.
		The `z` column is set to zero for all rows.

	properties : dict
		A dictionary where keys are property names (e.g., 'FRAME', 'state', 'generation', etc.) and values are numpy arrays
		containing the corresponding values from the dataframe.

	graph : dict
		An empty dictionary intended to store graph-related information for the tracking data. It can be extended
		later to represent relationships between different tracking objects.
	
	Notes
	-----
	- The function assumes that the dataframe contains specific columns: `TRACK_ID`, `FRAME`, `POSITION_Y`, `POSITION_X`, 
	  and `class_id`. These columns are used to construct the tracking data and properties.
	- The `z` coordinate is set to 0 for all tracks since the function does not process 3D data.
	- This function is useful for transforming tracking data into a format that can be used by tracking graph algorithms.

	Example
	-------
	>>> data, properties, graph = tracks_to_btrack(df, exclude_nans=True)
	"""

	graph = {}
	if exclude_nans:
		df.dropna(subset='class_id',inplace=True)

	df["z"] = 0.
	data = df[["TRACK_ID","FRAME","z","POSITION_Y","POSITION_X"]].to_numpy()
	
	df['dummy'] = False
	prop_cols = ['FRAME','state','generation','root','parent','dummy','class_id']
	properties = {}
	for col in prop_cols:
		properties.update({col: df[col].to_numpy()})
	
	return data, properties, graph

def tracks_to_napari(df, exclude_nans=False):
	
	data, properties, graph = tracks_to_btrack(df, exclude_nans=exclude_nans)
	vertices = data[:, [1,-2,-1]]
	if data.shape[1]==4:
		tracks = data
	else:
		tracks = data[:,[0,1,3,4]]
	return vertices, tracks, properties, graph


def view_tracks_in_napari(position, population, stack=None, labels=None, relabel=True, flush_memory=True, threads=1):
	
	"""
	Updated
	"""

	df, df_path = get_position_table(position, population=population, return_path=True)
	if df is None:
		print('Please compute trajectories first... Abort...')
		return None
	shared_data = {"df": df, "path": df_path, "position": position, "population": population, 'selected_frame': None}

	if (labels is not None) * relabel:
		print('Replacing the cell mask labels with the track ID...')
		labels = relabel_segmentation(labels, df, exclude_nans=True, threads=threads)

	vertices, tracks, properties, graph = tracks_to_napari(df, exclude_nans=True)

	viewer = napari.Viewer()
	if stack is not None:
		viewer.add_image(stack, channel_axis=-1, colormap=["gray"] * stack.shape[-1])
	if labels is not None:
		labels_layer = viewer.add_labels(labels.astype(int), name='segmentation', opacity=0.4)
	viewer.add_points(vertices, size=4, name='points', opacity=0.3)
	viewer.add_tracks(tracks, properties=properties, graph=graph, name='tracks')     

	def lock_controls(layer, widgets=(), locked=True):
		qctrl = viewer.window.qt_viewer.controls.widgets[layer]
		for wdg in widgets:
			try:
				getattr(qctrl, wdg).setEnabled(not locked)
			except:
				pass

	label_widget_list = ['paint_button', 'erase_button', 'fill_button', 'polygon_button', 'transform_button']
	lock_controls(viewer.layers['segmentation'], label_widget_list)

	point_widget_list = ['addition_button', 'delete_button', 'select_button', 'transform_button']
	lock_controls(viewer.layers['points'], point_widget_list)

	track_widget_list = ['transform_button']
	lock_controls(viewer.layers['tracks'], track_widget_list)

	# Initialize selected frame
	selected_frame = viewer.dims.current_step[0]
	shared_data['selected_frame'] = selected_frame

	def export_modifications():

		from celldetective.tracking import write_first_detection_class, clean_trajectories
		from celldetective.utils import velocity_per_track

		df = shared_data['df']
		position = shared_data['position']
		population = shared_data['population']
		df = velocity_per_track(df, window_size=3, mode='bi')
		df = write_first_detection_class(df, img_shape=labels[0].shape)

		experiment = extract_experiment_from_position(position)
		instruction_file = "/".join([experiment,"configs", f"tracking_instructions_{population}.json"])
		print(f"{instruction_file=}")
		if os.path.exists(instruction_file):
			print('Tracking configuration file found...')
			with open(instruction_file, 'r') as f:
				instructions = json.load(f)			
				if 'post_processing_options' in instructions:
					post_processing_options = instructions['post_processing_options']
					print(f'Applying the following track postprocessing: {post_processing_options}...')
					df = clean_trajectories(df.copy(),**post_processing_options)
		unnamed_cols = [c for c in list(df.columns) if c.startswith('Unnamed')]
		df = df.drop(unnamed_cols, axis=1)
		print(f"{list(df.columns)=}")
		df.to_csv(shared_data['path'], index=False)
		print('Done...')

	@magicgui(call_button='Export the modified\ntracks...')
	def export_table_widget():
		return export_modifications()

	def label_changed(event):

		value = viewer.layers['segmentation'].selected_label
		if value != 0:
			selected_frame = viewer.dims.current_step[0]
			shared_data['selected_frame'] = selected_frame


	viewer.layers['segmentation'].events.selected_label.connect(label_changed)

	viewer.window.add_dock_widget(export_table_widget, area='right')

	@labels_layer.mouse_double_click_callbacks.append
	def on_second_click_of_double_click(layer, event):

		df = shared_data['df']
		position = shared_data['position']
		population = shared_data['population']

		frame, x, y = event.position
		try:
			value_under = viewer.layers['segmentation'].data[int(frame), int(x), int(y)] #labels[0,int(y),int(x)]
			if value_under==0:
				return None
		except:
			print('Invalid mask value...')
			return None

		target_track_id = viewer.layers['segmentation'].selected_label

		msgBox = QMessageBox()
		msgBox.setIcon(QMessageBox.Question)
		msgBox.setText(f"Do you want to propagate track {target_track_id} to the cell under the mouse, track {value_under}?")
		msgBox.setWindowTitle("Info")
		msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
		returnValue = msgBox.exec()
		if returnValue == QMessageBox.No:
			return None
		else:

			if target_track_id not in df['TRACK_ID'].unique() and target_track_id in np.unique(viewer.layers['segmentation'].data[shared_data['selected_frame']]):
				# the selected cell in frame -1 is not in the table... we can add it to DataFrame
				current_labelm1 = viewer.layers['segmentation'].data[shared_data['selected_frame']]
				original_labelm1 = locate_labels(position, population=population, frames=shared_data['selected_frame'])				
				original_labelm1[current_labelm1!=target_track_id] = 0
				props = regionprops_table(original_labelm1, intensity_image=None, properties=['centroid', 'label'])
				props = pd.DataFrame(props)
				new_cell = props[['centroid-1', 'centroid-0','label']].copy()
				new_cell.rename(columns={'centroid-1': 'POSITION_X', 'centroid-0': 'POSITION_Y', 'label': 'class_id'},inplace=True)
				new_cell['FRAME'] = shared_data['selected_frame']
				new_cell['TRACK_ID'] = target_track_id
				df = pd.concat([df, new_cell], ignore_index=True)				

			if value_under not in df['TRACK_ID'].unique():
				# the cell to add is not currently part of DataFrame, need to add measurement

				current_label = viewer.layers['segmentation'].data[int(frame)]
				original_label = locate_labels(position, population=population, frames=int(frame))

				new_datapoint = {'TRACK_ID': value_under, 'FRAME': frame, 'POSITION_X': np.nan, 'POSITION_Y': np.nan, 'class_id': np.nan}
				
				original_label[current_label!=value_under] = 0

				props = regionprops_table(original_label, intensity_image=None, properties=['centroid', 'label'])
				props = pd.DataFrame(props)

				new_cell = props[['centroid-1', 'centroid-0','label']].copy()
				new_cell.rename(columns={'centroid-1': 'POSITION_X', 'centroid-0': 'POSITION_Y', 'label': 'class_id'},inplace=True)
				new_cell['FRAME'] = int(frame)
				new_cell['TRACK_ID'] = value_under
				df = pd.concat([df, new_cell], ignore_index=True)

			relabel = np.amax(viewer.layers['segmentation'].data) + 1
			for f in viewer.layers['segmentation'].data[int(frame):]:
				if target_track_id!=0:
					f[np.where(f==target_track_id)] = relabel
				f[np.where(f==value_under)] = target_track_id

			if target_track_id!=0:
				df.loc[(df['FRAME']>=frame)&(df['TRACK_ID']==target_track_id),'TRACK_ID'] = relabel
			df.loc[(df['FRAME']>=frame)&(df['TRACK_ID']==value_under),'TRACK_ID'] = target_track_id
			df = df.loc[~(df['TRACK_ID']==0),:]
			df = df.sort_values(by=['TRACK_ID','FRAME'])

			vertices, tracks, properties, graph = tracks_to_napari(df, exclude_nans=True)
			
			viewer.layers['tracks'].data = tracks
			viewer.layers['tracks'].properties = properties
			viewer.layers['tracks'].graph = graph

			viewer.layers['points'].data = vertices

			viewer.layers['segmentation'].refresh()
			viewer.layers['tracks'].refresh()
			viewer.layers['points'].refresh()

		shared_data['df'] = df

	viewer.show(block=True)

	if flush_memory:

		# temporary fix for slight napari memory leak
		for i in range(10000):
			try:
				viewer.layers.pop()
			except:
				pass

		del viewer
		del stack
		del labels
		gc.collect()


def load_napari_data(position, prefix="Aligned", population="target", return_stack=True):
	
	"""
	Load the necessary data for visualization in napari.

	Parameters
	----------
	position : str
		The path to the position or experiment directory.
	prefix : str, optional
		The prefix used to identify the the movie file. The default is "Aligned".
	population : str, optional
		The population type to load, either "target" or "effector". The default is "target".

	Returns
	-------
	tuple
		A tuple containing the loaded data, properties, graph, labels, and stack.

	Examples
	--------
	>>> data, properties, graph, labels, stack = load_napari_data("path/to/position")
	# Load the necessary data for visualization of target trajectories.

	"""

	if not position.endswith(os.sep):
		position += os.sep

	position = position.replace('\\','/')
	if population.lower()=="target" or population.lower()=="targets":
		if os.path.exists(position+os.sep.join(['output','tables','napari_target_trajectories.npy'])):
			napari_data = np.load(position+os.sep.join(['output','tables','napari_target_trajectories.npy']), allow_pickle=True)
		else:
			napari_data = None
	elif population.lower()=="effector" or population.lower()=="effectors":
		if os.path.exists(position+os.sep.join(['output', 'tables', 'napari_effector_trajectories.npy'])):
			napari_data = np.load(position+os.sep.join(['output', 'tables', 'napari_effector_trajectories.npy']), allow_pickle=True)
		else:
			napari_data = None

	if napari_data is not None:
		data = napari_data.item()['data']
		properties = napari_data.item()['properties']
		graph = napari_data.item()['graph']
	else:
		data = None
		properties = None
		graph = None
	if return_stack:
		stack, labels = locate_stack_and_labels(position, prefix=prefix, population=population)
	else:
		labels = locate_labels(position, population=population)
		stack = None
	return data, properties, graph, labels, stack


def auto_correct_masks(masks, bbox_factor = 1.75, min_area=9, fill_labels=False):

	"""
	Correct segmentation masks to ensure consistency and remove anomalies.

	This function processes a labeled mask image to correct anomalies and reassign labels. 
	It performs the following operations:
	
	1. Corrects negative mask values by taking their absolute values.
	2. Identifies and corrects segmented objects with a bounding box area that is disproportionately 
	   larger than the actual object area. This indicates potential segmentation errors where separate objects 
	   share the same label.
	3. Removes small objects that are considered noise (default threshold is an area of less than 9 pixels).
	4. Reorders the labels so they are consecutive from 1 up to the number of remaining objects (to avoid encoding errors).

	Parameters
	----------
	masks : np.ndarray
		A 2D array representing the segmented mask image with labeled regions. Each unique value 
		in the array represents a different object or cell.

	Returns
	-------
	clean_labels : np.ndarray
		A corrected version of the input mask, with anomalies corrected, small objects removed, 
		and labels reordered to be consecutive integers.

	Notes
	-----
	- This function is useful for post-processing segmentation outputs to ensure high-quality 
	  object detection, particularly in applications such as cell segmentation in microscopy images.
	- The function assumes that the input masks contain integer labels and that the background 
	  is represented by 0.

	Examples
	--------
	>>> masks = np.array([[0, 0, 1, 1], [0, 2, 2, 1], [0, 2, 0, 0]])
	>>> corrected_masks = auto_correct_masks(masks)
	>>> corrected_masks
	array([[0, 0, 1, 1],
		   [0, 2, 2, 1],
		   [0, 2, 0, 0]])
	"""

	# Avoid negative mask values
	masks[masks<0] = np.abs(masks[masks<0])
	
	props = pd.DataFrame(regionprops_table(masks, properties=('label', 'area', 'area_bbox')))
	max_lbl = props['label'].max()
	corrected_lbl = masks.copy() #.astype(int)

	for cell in props['label'].unique():

		bbox_area = props.loc[props['label'] == cell, 'area_bbox'].values
		area = props.loc[props['label'] == cell, 'area'].values

		if bbox_area > bbox_factor * area:  # condition for anomaly

			lbl = masks == cell
			lbl = lbl.astype(int)

			relabelled = label(lbl, connectivity=2)
			relabelled += max_lbl
			relabelled[np.where(lbl == 0)] = 0

			corrected_lbl[np.where(relabelled != 0)] = relabelled[np.where(relabelled != 0)]

		max_lbl = np.amax(corrected_lbl)

	# Second routine to eliminate objects too small
	props2 = pd.DataFrame(regionprops_table(corrected_lbl, properties=('label', 'area', 'area_bbox')))
	for cell in props2['label'].unique():
		area = props2.loc[props2['label'] == cell, 'area'].values
		lbl = corrected_lbl == cell
		if area < min_area:
			corrected_lbl[lbl] = 0

	# Additionnal routine to reorder labels from 1 to number of cells
	label_ids = np.unique(corrected_lbl)[1:]
	clean_labels = corrected_lbl.copy()

	for k,lbl in enumerate(label_ids):
		clean_labels[corrected_lbl==lbl] = k+1

	clean_labels = clean_labels.astype(int)

	if fill_labels:
		clean_labels = fill_label_holes(clean_labels)

	return clean_labels



def control_segmentation_napari(position, prefix='Aligned', population="target", flush_memory=False):

	"""

	Control the visualization of segmentation labels using the napari viewer.

	Parameters
	----------
	position : str
		The position or directory path where the segmentation labels and stack are located.
	prefix : str, optional
		The prefix used to identify the stack. The default is 'Aligned'.
	population : str, optional
		The population type for which the segmentation is performed. The default is 'target'.

	Notes
	-----
	This function loads the segmentation labels and stack corresponding to the specified position and population.
	It then creates a napari viewer and adds the stack and labels as layers for visualization.

	Examples
	--------
	>>> control_segmentation_napari(position, prefix='Aligned', population="target")
	# Control the visualization of segmentation labels using the napari viewer.

	"""

	def export_labels():
		labels_layer = viewer.layers['segmentation'].data
		for t, im in enumerate(tqdm(labels_layer)):

			try:
				im = auto_correct_masks(im)
			except Exception as e:
				print(e)

			save_tiff_imagej_compatible(output_folder + f"{str(t).zfill(4)}.tif", im.astype(np.int16), axes='YX')
		print("The labels have been successfully rewritten.")

	def export_annotation():

		# Locate experiment config
		parent1 = Path(position).parent
		expfolder = parent1.parent
		config = PurePath(expfolder, Path("config.ini"))
		expfolder = str(expfolder)
		exp_name = os.path.split(expfolder)[-1]
		print(exp_name)

		wells = get_experiment_wells(expfolder)
		well_idx = list(wells).index(str(parent1)+os.sep)
		ab = get_experiment_antibodies(expfolder)[well_idx]
		conc = get_experiment_concentrations(expfolder)[well_idx]
		ct = get_experiment_cell_types(expfolder)[well_idx]
		pa = get_experiment_pharmaceutical_agents(expfolder)[well_idx]


		spatial_calibration = float(ConfigSectionMap(config,"MovieSettings")["pxtoum"])
		channel_names, channel_indices = extract_experiment_channels(config)

		annotation_folder = expfolder + os.sep + f'annotations_{population}' + os.sep
		if not os.path.exists(annotation_folder):
			os.mkdir(annotation_folder)

		print('exporting!')
		t = viewer.dims.current_step[0]
		labels_layer = viewer.layers['segmentation'].data[t]  # at current time

		try:
			labels_layer = auto_correct_masks(labels_layer)
		except Exception as e:
			print(e)

		fov_export = True

		if "Shapes" in viewer.layers:
			squares = viewer.layers['Shapes'].data
			test_in_frame = np.array([squares[i][0, 0] == t and len(squares[i]) == 4 for i in range(len(squares))])
			squares = np.array(squares)
			squares = squares[test_in_frame]
			nbr_squares = len(squares)
			print(f"Found {nbr_squares} ROIs...")
			if nbr_squares > 0:
				# deactivate field of view mode
				fov_export = False

			for k, sq in enumerate(squares):
				print(f"ROI: {sq}")
				pad_to_256=False

				xmin = int(sq[0, 1])
				xmax = int(sq[2, 1])
				if xmax < xmin:
					xmax, xmin = xmin, xmax
				ymin = int(sq[0, 2])
				ymax = int(sq[1, 2])
				if ymax < ymin:
					ymax, ymin = ymin, ymax
				print(f"{xmin=};{xmax=};{ymin=};{ymax=}")
				frame = viewer.layers['Image'].data[t][xmin:xmax, ymin:ymax]
				if frame.shape[1] < 256 or frame.shape[0] < 256:
					pad_to_256 = True
					print("Crop too small! Padding with zeros to reach 256*256 pixels...")
					#continue
				multichannel = [frame]
				for i in range(len(channel_indices) - 1):
					try:
						frame = viewer.layers[f'Image [{i + 1}]'].data[t][xmin:xmax, ymin:ymax]
						multichannel.append(frame)
					except:
						pass
				multichannel = np.array(multichannel)
				lab = labels_layer[xmin:xmax,ymin:ymax].astype(np.int16)
				if pad_to_256:
					shape = multichannel.shape
					pad_length_x = max([0,256 - multichannel.shape[1]])
					if pad_length_x>0 and pad_length_x%2==1:
						pad_length_x += 1
					pad_length_y = max([0,256 - multichannel.shape[2]])
					if pad_length_y>0 and pad_length_y%2==1:
						pad_length_y += 1
					padded_image = np.array([np.pad(im, ((pad_length_x//2,pad_length_x//2), (pad_length_y//2,pad_length_y//2)), mode='constant') for im in multichannel])
					padded_label = np.pad(lab,((pad_length_x//2,pad_length_x//2), (pad_length_y//2,pad_length_y//2)), mode='constant')
					lab = padded_label; multichannel = padded_image;

				save_tiff_imagej_compatible(annotation_folder + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_roi_{xmin}_{xmax}_{ymin}_{ymax}_labelled.tif", lab, axes='YX')
				save_tiff_imagej_compatible(annotation_folder + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_roi_{xmin}_{xmax}_{ymin}_{ymax}.tif", multichannel, axes='CYX')
				info = {"spatial_calibration": spatial_calibration, "channels": list(channel_names), 'cell_type': ct, 'antibody': ab, 'concentration': conc, 'pharmaceutical_agent': pa}
				info_name = annotation_folder + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_roi_{xmin}_{xmax}_{ymin}_{ymax}.json"
				with open(info_name, 'w') as f:
					json.dump(info, f, indent=4)

		if fov_export:
			frame = viewer.layers['Image'].data[t]
			multichannel = [frame]
			for i in range(len(channel_indices) - 1):
				try:
					frame = viewer.layers[f'Image [{i + 1}]'].data[t]
					multichannel.append(frame)
				except:
					pass
			multichannel = np.array(multichannel)       
			save_tiff_imagej_compatible(annotation_folder + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}_labelled.tif", labels_layer, axes='YX')
			save_tiff_imagej_compatible(annotation_folder + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}.tif", multichannel, axes='CYX')
			info = {"spatial_calibration": spatial_calibration, "channels": list(channel_names), 'cell_type': ct, 'antibody': ab, 'concentration': conc, 'pharmaceutical_agent': pa}
			info_name = annotation_folder + f"{exp_name}_{position.split(os.sep)[-2]}_{str(t).zfill(4)}.json"
			with open(info_name, 'w') as f:
				json.dump(info, f, indent=4)
			print('Done.')

	@magicgui(call_button='Save the modified labels')
	def save_widget():
		return export_labels()

	@magicgui(call_button='Export the annotation\nof the current frame')
	def export_widget():
		return export_annotation()

	stack, labels = locate_stack_and_labels(position, prefix=prefix, population=population)

	if not population.endswith('s'):
		population += 's'
	output_folder = position + f'labels_{population}{os.sep}'

	print(f"{stack.shape}")
	viewer = napari.Viewer()
	viewer.add_image(stack, channel_axis=-1, colormap=["gray"] * stack.shape[-1])
	viewer.add_labels(labels.astype(int), name='segmentation', opacity=0.4)
	viewer.window.add_dock_widget(save_widget, area='right')
	viewer.window.add_dock_widget(export_widget, area='right')

	def lock_controls(layer, widgets=(), locked=True):
		qctrl = viewer.window.qt_viewer.controls.widgets[layer]
		for wdg in widgets:
			try:
				getattr(qctrl, wdg).setEnabled(not locked)
			except:
				pass

	label_widget_list = ['polygon_button', 'transform_button']
	lock_controls(viewer.layers['segmentation'], label_widget_list)


	viewer.show(block=True)

	if flush_memory:
		# temporary fix for slight napari memory leak
		for i in range(10000):
			try:
				viewer.layers.pop()
			except:
				pass

		del viewer
		del stack
		del labels
		gc.collect()

def correct_annotation(filename):

	"""
	New function to reannotate an annotation image in post, using napari and save update inplace.
	"""

	def export_labels():
		labels_layer = viewer.layers['segmentation'].data
		for t,im in enumerate(tqdm(labels_layer)):

			try:
				im = auto_correct_masks(im)
			except Exception as e:
				print(e)

			save_tiff_imagej_compatible(existing_lbl, im.astype(np.int16), axes='YX')
		print("The labels have been successfully rewritten.")

	@magicgui(call_button='Save the modified labels')
	def save_widget():
		return export_labels()

	img = imread(filename.replace('\\','/'))
	if img.ndim==3:
		img = np.moveaxis(img, 0, -1)
	elif img.ndim==2:
		img = img[:,:,np.newaxis]

	existing_lbl = filename.replace('.tif','_labelled.tif')
	if os.path.exists(existing_lbl):
		labels = imread(existing_lbl)[np.newaxis,:,:].astype(int)
	else:
		labels = np.zeros_like(img[:,:,0]).astype(int)[np.newaxis,:,:]

	stack = img[np.newaxis,:,:,:]

	viewer = napari.Viewer()
	viewer.add_image(stack,channel_axis=-1,colormap=["gray"]*stack.shape[-1])
	viewer.add_labels(labels, name='segmentation',opacity=0.4)
	viewer.window.add_dock_widget(save_widget, area='right')
	viewer.show(block=True)

	# temporary fix for slight napari memory leak
	for i in range(100):
		try:
			viewer.layers.pop()
		except:
			pass
	del viewer
	del stack
	del labels
	gc.collect()


def _view_on_napari(tracks=None, stack=None, labels=None):
	
	"""

	Visualize tracks, stack, and labels using Napari.

	Parameters
	----------
	tracks : pandas DataFrame
		DataFrame containing track information.
	stack : numpy array, optional
		Stack of images with shape (T, Y, X, C), where T is the number of frames, Y and X are the spatial dimensions,
		and C is the number of channels. Default is None.
	labels : numpy array, optional
		Label stack with shape (T, Y, X) representing cell segmentations. Default is None.

	Returns
	-------
	None

	Notes
	-----
	This function visualizes tracks, stack, and labels using Napari, an interactive multi-dimensional image viewer.
	The tracks are represented as line segments on the viewer. If a stack is provided, it is displayed as an image.
	If labels are provided, they are displayed as a segmentation overlay on the stack.

	Examples
	--------
	>>> tracks = pd.DataFrame({'track': [1, 2, 3], 'time': [1, 1, 1],
	...                        'x': [10, 20, 30], 'y': [15, 25, 35]})
	>>> stack = np.random.rand(100, 100, 3)
	>>> labels = np.random.randint(0, 2, (100, 100))
	>>> view_on_napari(tracks, stack=stack, labels=labels)
	# Visualize tracks, stack, and labels using Napari.

	"""

	viewer = napari.Viewer()
	if stack is not None:
		viewer.add_image(stack, channel_axis=-1, colormap=["gray"] * stack.shape[-1])
	if labels is not None:
		viewer.add_labels(labels, name='segmentation', opacity=0.4)
	if tracks is not None:
		viewer.add_tracks(tracks, name='tracks')
	viewer.show(block=True)


def control_tracking_table(position, calibration=1, prefix="Aligned", population="target",
						   column_labels={'track': "TRACK_ID", 'frame': 'FRAME', 'y': 'POSITION_Y', 'x': 'POSITION_X',
										  'label': 'class_id'}):
	"""

	Control the tracking table and visualize tracks using Napari.

	Parameters
	----------
	position : str
		The position or directory of the tracking data.
	calibration : float, optional
		Calibration factor for converting pixel coordinates to physical units. Default is 1.
	prefix : str, optional
		Prefix used for the tracking data file. Default is "Aligned".
	population : str, optional
		Population type, either "target" or "effector". Default is "target".
	column_labels : dict, optional
		Dictionary containing the column labels for the tracking table. Default is
		{'track': "TRACK_ID", 'frame': 'FRAME', 'y': 'POSITION_Y', 'x': 'POSITION_X', 'label': 'class_id'}.

	Returns
	-------
	None

	Notes
	-----
	This function loads the tracking data, applies calibration to the spatial coordinates, and visualizes the tracks
	using Napari. The tracking data is loaded from the specified `position` directory with the given `prefix` and
	`population`. The spatial coordinates (x, y) in the tracking table are divided by the `calibration` factor to
	convert them from pixel units to the specified physical units. The tracks are then visualized using Napari.

	Examples
	--------
	>>> control_tracking_table('path/to/tracking_data', calibration=0.1, prefix='Aligned', population='target')
	# Control the tracking table and visualize tracks using Napari.

	"""

	position = position.replace('\\', '/')
	tracks, labels, stack = load_tracking_data(position, prefix=prefix, population=population)
	tracks = tracks.loc[:,
			 [column_labels['track'], column_labels['frame'], column_labels['y'], column_labels['x']]].to_numpy()
	tracks[:, -2:] /= calibration
	_view_on_napari(tracks, labels=labels, stack=stack)


def get_segmentation_models_list(mode='targets', return_path=False):
	if mode == 'targets':
		modelpath = os.sep.join(
			[os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective", "models",
			 "segmentation_targets", os.sep])
		repository_models = get_zenodo_files(cat=os.sep.join(["models", "segmentation_targets"]))
	elif mode == 'effectors':
		modelpath = os.sep.join(
			[os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective", "models",
			 "segmentation_effectors", os.sep])
		repository_models = get_zenodo_files(cat=os.sep.join(["models", "segmentation_effectors"]))
	elif mode == 'generic':
		modelpath = os.sep.join(
			[os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective", "models",
			 "segmentation_generic", os.sep])
		repository_models = get_zenodo_files(cat=os.sep.join(["models", "segmentation_generic"]))

	available_models = natsorted(glob(modelpath + '*/'))
	available_models = [m.replace('\\', '/').split('/')[-2] for m in available_models]

	# Auto model cleanup
	to_remove = []
	for model in available_models:
		path = modelpath + model
		files = glob(path+os.sep+"*")
		if path+os.sep+"config_input.json" not in files:
			rmtree(path)
			to_remove.append(model)
	for m in to_remove:
		available_models.remove(m)


	for rm in repository_models:
		if rm not in available_models:
			available_models.append(rm)


	if not return_path:
		return available_models
	else:
		return available_models, modelpath


def locate_segmentation_model(name, download=True):
	
	"""
	Locates a specified segmentation model within the local 'celldetective' directory or
	downloads it from Zenodo if not found locally.

	This function attempts to find a segmentation model by name within a predefined directory
	structure starting from the 'celldetective/models/segmentation*' path. If the model is not
	found locally, it then tries to locate and download the model from Zenodo, placing it into
	the appropriate category directory within 'celldetective'. The function prints the search
	directory path and returns the path to the found or downloaded model.

	Parameters
	----------
	name : str
		The name of the segmentation model to locate.

	Returns
	-------
	str or None
		The full path to the located or downloaded segmentation model directory, or None if the
		model could not be found or downloaded.

	Raises
	------
	FileNotFoundError
		If the model cannot be found locally and also cannot be found or downloaded from Zenodo.

	"""

	main_dir = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0],"celldetective"])
	modelpath = os.sep.join([main_dir, "models", "segmentation*"]) + os.sep
	#print(f'Looking for {name} in {modelpath}')
	models = glob(modelpath + f'*{os.sep}')

	match = None
	for m in models:
		if name == m.replace('\\', os.sep).split(os.sep)[-2]:
			match = m
			return match
	if download:
		# else no match, try zenodo
		files, categories = get_zenodo_files()
		if name in files:
			index = files.index(name)
			cat = categories[index]
			download_zenodo_file(name, os.sep.join([main_dir, cat]))
			match = os.sep.join([main_dir, cat, name]) + os.sep

	return match


def get_segmentation_datasets_list(return_path=False):
	"""
	Retrieves a list of available segmentation datasets from both the local 'celldetective/datasets/segmentation_annotations'
	directory and a Zenodo repository, optionally returning the path to the local datasets directory.

	This function compiles a list of available segmentation datasets by first identifying datasets stored locally
	within a specified path related to the script's directory. It then extends this list with datasets available
	in a Zenodo repository, ensuring no duplicates are added. The function can return just the list of dataset
	names or, if specified, also return the path to the local datasets directory.

	Parameters
	----------
	return_path : bool, optional
		If True, the function returns a tuple containing the list of available dataset names and the path to the
		local datasets directory. If False, only the list of dataset names is returned (default is False).

	Returns
	-------
	list or (list, str)
		If return_path is False, returns a list of strings, each string being the name of an available dataset.
		If return_path is True, returns a tuple where the first element is this list and the second element is a
		string representing the path to the local datasets directory.

	"""

	datasets_path = os.sep.join(
		[os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective", "datasets",
		 "segmentation_annotations", os.sep])
	repository_datasets = get_zenodo_files(cat=os.sep.join(["datasets", "segmentation_annotations"]))

	available_datasets = natsorted(glob(datasets_path + '*/'))
	available_datasets = [m.replace('\\', '/').split('/')[-2] for m in available_datasets]
	for rm in repository_datasets:
		if rm not in available_datasets:
			available_datasets.append(rm)

	if not return_path:
		return available_datasets
	else:
		return available_datasets, datasets_path



def locate_segmentation_dataset(name):
	
	"""
	Locates a specified segmentation dataset within the local 'celldetective/datasets/segmentation_annotations' directory
	or downloads it from Zenodo if not found locally.

	This function attempts to find a segmentation dataset by name within a predefined directory structure. If the dataset
	is not found locally, it then tries to locate and download the dataset from Zenodo, placing it into the appropriate
	category directory within 'celldetective'. The function prints the search directory path and returns the path to the
	found or downloaded dataset.

	Parameters
	----------
	name : str
		The name of the segmentation dataset to locate.

	Returns
	-------
	str or None
		The full path to the located or downloaded segmentation dataset directory, or None if the dataset could not be
		found or downloaded.

	Raises
	------
	FileNotFoundError
		If the dataset cannot be found locally and also cannot be found or downloaded from Zenodo.

	"""

	main_dir = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"])
	modelpath = os.sep.join([main_dir, "datasets", "segmentation_annotations", os.sep])
	print(f'Looking for {name} in {modelpath}')
	models = glob(modelpath + f'*{os.sep}')

	match = None
	for m in models:
		if name == m.replace('\\', os.sep).split(os.sep)[-2]:
			match = m
			return match
	# else no match, try zenodo
	files, categories = get_zenodo_files()
	if name in files:
		index = files.index(name)
		cat = categories[index]
		download_zenodo_file(name, os.sep.join([main_dir, cat]))
		match = os.sep.join([main_dir, cat, name]) + os.sep
	return match


def get_signal_datasets_list(return_path=False):

	"""
	Retrieves a list of available signal datasets from both the local 'celldetective/datasets/signal_annotations' directory
	and a Zenodo repository, optionally returning the path to the local datasets directory.

	This function compiles a list of available signal datasets by first identifying datasets stored locally within a specified
	path related to the script's directory. It then extends this list with datasets available in a Zenodo repository, ensuring
	no duplicates are added. The function can return just the list of dataset names or, if specified, also return the path to
	the local datasets directory.

	Parameters
	----------
	return_path : bool, optional
		If True, the function returns a tuple containing the list of available dataset names and the path to the local datasets
		directory. If False, only the list of dataset names is returned (default is False).

	Returns
	-------
	list or (list, str)
		If return_path is False, returns a list of strings, each string being the name of an available dataset. If return_path
		is True, returns a tuple where the first element is this list and the second element is a string representing the path
		to the local datasets directory.

	"""

	datasets_path = os.sep.join(
		[os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective", "datasets",
		 "signal_annotations", os.sep])
	repository_datasets = get_zenodo_files(cat=os.sep.join(["datasets", "signal_annotations"]))

	available_datasets = natsorted(glob(datasets_path + '*/'))
	available_datasets = [m.replace('\\', '/').split('/')[-2] for m in available_datasets]
	for rm in repository_datasets:
		if rm not in available_datasets:
			available_datasets.append(rm)

	if not return_path:
		return available_datasets
	else:
		return available_datasets, datasets_path


def locate_signal_dataset(name):
	
	"""
	Locates a specified signal dataset within the local 'celldetective/datasets/signal_annotations' directory or downloads
	it from Zenodo if not found locally.

	This function attempts to find a signal dataset by name within a predefined directory structure. If the dataset is not
	found locally, it then tries to locate and download the dataset from Zenodo, placing it into the appropriate category
	directory within 'celldetective'. The function prints the search directory path and returns the path to the found or
	downloaded dataset.

	Parameters
	----------
	name : str
		The name of the signal dataset to locate.

	Returns
	-------
	str or None
		The full path to the located or downloaded signal dataset directory, or None if the dataset could not be found or
		downloaded.

	Raises
	------
	FileNotFoundError
		If the dataset cannot be found locally and also cannot be found or downloaded from Zenodo.

	"""

	main_dir = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], "celldetective"])
	modelpath = os.sep.join([main_dir, "datasets", "signal_annotations", os.sep])
	print(f'Looking for {name} in {modelpath}')
	models = glob(modelpath + f'*{os.sep}')

	match = None
	for m in models:
		if name == m.replace('\\', os.sep).split(os.sep)[-2]:
			match = m
			return match
	# else no match, try zenodo
	files, categories = get_zenodo_files()
	if name in files:
		index = files.index(name)
		cat = categories[index]
		download_zenodo_file(name, os.sep.join([main_dir, cat]))
		match = os.sep.join([main_dir, cat, name]) + os.sep
	return match

def normalize(frame, percentiles=(0.0,99.99), values=None, ignore_gray_value=0., clip=False, amplification=None, dtype=float):

	"""

	Normalize the intensity values of a frame.

	Parameters
	----------
	frame : ndarray
		The input frame to be normalized.
	percentiles : tuple, optional
		The percentiles used to determine the minimum and maximum values for normalization. Default is (0.0, 99.99).
	values : tuple or None, optional
		The specific minimum and maximum values to use for normalization. If None, percentiles are used. Default is None.
	ignore_gray_value : float or None, optional
		The gray value to ignore during normalization. If specified, the pixels with this value will not be normalized. Default is 0.0.

	Returns
	-------
	ndarray
		The normalized frame.

	Notes
	-----
	This function performs intensity normalization on a frame. It computes the minimum and maximum values for normalization either
	using the specified values or by calculating percentiles from the frame. The frame is then normalized between the minimum and
	maximum values using the `normalize_mi_ma` function. If `ignore_gray_value` is specified, the pixels with this value will be
	left unmodified during normalization.

	Examples
	--------
	>>> frame = np.array([[10, 20, 30],
						  [40, 50, 60],
						  [70, 80, 90]])
	>>> normalized = normalize(frame)
	>>> normalized

	array([[0. , 0.2, 0.4],
		   [0.6, 0.8, 1. ],
		   [1.2, 1.4, 1.6]], dtype=float32)

	>>> normalized = normalize(frame, percentiles=(10.0, 90.0))
	>>> normalized

	array([[0.33333334, 0.44444445, 0.5555556 ],
		   [0.6666667 , 0.7777778 , 0.8888889 ],
		   [1.        , 1.1111112 , 1.2222222 ]], dtype=float32)

	"""

	frame = frame.astype(float)

	if ignore_gray_value is not None:
		subframe = frame[frame != ignore_gray_value]
	else:
		subframe = frame.copy()

	if values is not None:
		mi = values[0];
		ma = values[1]
	else:
		mi = np.nanpercentile(subframe.flatten(), percentiles[0], keepdims=True)
		ma = np.nanpercentile(subframe.flatten(), percentiles[1], keepdims=True)

	frame0 = frame.copy()
	frame = normalize_mi_ma(frame0, mi, ma, clip=False, eps=1e-20, dtype=np.float32)
	if amplification is not None:
		frame *= amplification
	if clip:
		if amplification is None:
			amplification = 1.
		frame[frame >= amplification] = amplification
		frame[frame <= 0.] = 0.
	if ignore_gray_value is not None:
		frame[np.where(frame0) == ignore_gray_value] = ignore_gray_value

	return frame.copy().astype(dtype)


def normalize_multichannel(multichannel_frame, percentiles=None,
						   values=None, ignore_gray_value=0., clip=False,
						   amplification=None, dtype=float):
	
	"""
	Normalizes a multichannel frame by adjusting the intensity values of each channel based on specified percentiles,
	direct value ranges, or amplification factors, with options to ignore a specific gray value and to clip the output.

	Parameters
	----------
	multichannel_frame : ndarray
		The input multichannel image frame to be normalized, expected to be a 3-dimensional array where the last dimension
		represents the channels.
	percentiles : list of tuples or tuple, optional
		Percentile ranges (low, high) for each channel used to scale the intensity values. If a single tuple is provided,
		it is applied to all channels. If None, the default percentile range of (0., 99.99) is used for each channel.
	values : list of tuples or tuple, optional
		Direct value ranges (min, max) for each channel to scale the intensity values. If a single tuple is provided, it
		is applied to all channels. This parameter overrides `percentiles` if provided.
	ignore_gray_value : float, optional
		A specific gray value to ignore during normalization (default is 0.).
	clip : bool, optional
		If True, clips the output values to the range [0, 1] or the specified `dtype` range if `dtype` is not float
		(default is False).
	amplification : float, optional
		A factor by which to amplify the intensity values after normalization. If None, no amplification is applied.
	dtype : data-type, optional
		The desired data-type for the output normalized frame. The default is float, but other types can be specified
		to change the range of the output values.

	Returns
	-------
	ndarray
		The normalized multichannel frame as a 3-dimensional array of the same shape as `multichannel_frame`.

	Raises
	------
	AssertionError
		If the input `multichannel_frame` does not have 3 dimensions, or if the length of `values` does not match the
		number of channels in `multichannel_frame`.

	Notes
	-----
	- This function provides flexibility in normalization by allowing the use of percentile ranges, direct value ranges,
	  or amplification factors.
	- The function makes a copy of the input frame to avoid altering the original data.
	- When both `percentiles` and `values` are provided, `values` takes precedence for normalization.

	Examples
	--------
	>>> multichannel_frame = np.random.rand(100, 100, 3)  # Example multichannel frame
	>>> normalized_frame = normalize_multichannel(multichannel_frame, percentiles=((1, 99), (2, 98), (0, 100)))
	# Normalizes each channel of the frame using specified percentile ranges.

	"""



	mf = multichannel_frame.copy().astype(float)
	assert mf.ndim == 3, f'Wrong shape for the multichannel frame: {mf.shape}.'
	if percentiles is None:
		percentiles = [(0., 99.99)] * mf.shape[-1]
	elif isinstance(percentiles, tuple):
		percentiles = [percentiles] * mf.shape[-1]
	if values is not None:
		if isinstance(values, tuple):
			values = [values] * mf.shape[-1]
		assert len(values) == mf.shape[
			-1], 'Mismatch between the normalization values provided and the number of channels.'

	mf_new = []
	for c in range(mf.shape[-1]):
		if values is not None:
			v = values[c]
		else:
			v = None

		if np.all(mf[:,:,c]==0.):
			mf_new.append(mf[:,:,c].copy())
		else:
			norm = normalize(mf[:,:,c].copy(),
								  percentiles=percentiles[c],
								  values=v,
								  ignore_gray_value=ignore_gray_value,
								  clip=clip,
								  amplification=amplification,
								  dtype=dtype,
								  )
			mf_new.append(norm)

	return np.moveaxis(mf_new,0,-1)

def load_frames(img_nums, stack_path, scale=None, normalize_input=True, dtype=float, normalize_kwargs={"percentiles": (0.,99.99)}):

	"""
	Loads and optionally normalizes and rescales specified frames from a stack located at a given path.

	This function reads specified frames from a stack file, applying systematic adjustments to ensure
	the channel axis is last. It supports optional normalization of the input frames and rescaling. An
	artificial pixel modification is applied to frames with uniform values to prevent errors during
	normalization.

	Parameters
	----------
	img_nums : int or list of int
		The index (or indices) of the image frame(s) to load from the stack.
	stack_path : str
		The file path to the stack from which frames are to be loaded.
	scale : float, optional
		The scaling factor to apply to the frames. If None, no scaling is applied (default is None).
	normalize_input : bool, optional
		Whether to normalize the loaded frames. If True, normalization is applied according to
		`normalize_kwargs` (default is True).
	dtype : data-type, optional
		The desired data-type for the output frames (default is float).
	normalize_kwargs : dict, optional
		Keyword arguments to pass to the normalization function (default is {"percentiles": (0., 99.99)}).

	Returns
	-------
	ndarray or None
		The loaded, and possibly normalized and rescaled, frames as a NumPy array. Returns None if there 
		is an error in loading the frames.

	Raises
	------
	Exception
		Prints an error message if the specified frames cannot be loaded or if there is a mismatch between
		the provided experiment channel information and the stack format.

	Notes
	-----
	- The function uses scikit-image for reading frames and supports multi-frame TIFF stacks.
	- Normalization and scaling are optional and can be customized through function parameters.
	- A workaround is implemented for frames with uniform pixel values to prevent normalization errors by
	  adding a 'fake' pixel.

	Examples
	--------
	>>> frames = load_frames([0, 1, 2], '/path/to/stack.tif', scale=0.5, normalize_input=True, dtype=np.uint8)
	# Loads the first three frames from '/path/to/stack.tif', normalizes them, rescales by a factor of 0.5,
	# and converts them to uint8 data type.

	"""

	try:
		frames = skio.imread(stack_path, key=img_nums, plugin="tifffile")
	except Exception as e:
		print(
			f'Error in loading the frame {img_nums} {e}. Please check that the experiment channel information is consistent with the movie being read.')
		return None

	frames = _rearrange_multichannel_frame(frames)

	if normalize_input:
		frames = normalize_multichannel(frames, **normalize_kwargs)

	if scale is not None:
		frames = zoom_multiframes(frames, scale)

	# add a fake pixel to prevent auto normalization errors on images that are uniform
	frames = _fix_no_contrast(frames)

	return frames.astype(dtype)


def get_stack_normalization_values(stack, percentiles=None, ignore_gray_value=0.):

	"""
	Computes the normalization value ranges (minimum and maximum) for each channel in a 4D stack based on specified percentiles.

	This function calculates the value ranges for normalizing each channel within a 4-dimensional stack, with dimensions
	expected to be in the order of Time (T), Y (height), X (width), and Channels (C). The normalization values are determined
	by the specified percentiles for each channel. An option to ignore a specific gray value during computation is provided,
	though its effect is not implemented in this snippet.

	Parameters
	----------
	stack : ndarray
		The input 4D stack with dimensions TYXC from which to calculate normalization values.
	percentiles : tuple, list of tuples, optional
		The percentile values (low, high) used to calculate the normalization ranges for each channel. If a single tuple
		is provided, it is applied to all channels. If a list of tuples is provided, each tuple is applied to the
		corresponding channel. If None, defaults to (0., 99.99) for each channel.
	ignore_gray_value : float, optional
		A gray value to potentially ignore during the calculation. This parameter is provided for interface consistency
		but is not utilized in the current implementation (default is 0.).

	Returns
	-------
	list of tuples
		A list where each tuple contains the (minimum, maximum) values for normalizing each channel based on the specified
		percentiles.

	Raises
	------
	AssertionError
		If the input stack does not have 4 dimensions, or if the length of the `percentiles` list does not match the number
		of channels in the stack.

	Notes
	-----
	- The function assumes the input stack is in TYXC format, where T is the time dimension, Y and X are spatial dimensions,
	  and C is the channel dimension.
	- Memory management via `gc.collect()` is employed after calculating normalization values for each channel to mitigate
	  potential memory issues with large datasets.

	Examples
	--------
	>>> stack = np.random.rand(5, 100, 100, 3)  # Example 4D stack with 3 channels
	>>> normalization_values = get_stack_normalization_values(stack, percentiles=((1, 99), (2, 98), (0, 100)))
	# Calculates normalization ranges for each channel using the specified percentiles.

	"""

	assert stack.ndim == 4, f'Wrong number of dimensions for the stack, expect TYXC (4) got {stack.ndim}.'
	if percentiles is None:
		percentiles = [(0., 99.99)] * stack.shape[-1]
	elif isinstance(percentiles, tuple):
		percentiles = [percentiles] * stack.shape[-1]
	elif isinstance(percentiles, list):
		assert len(percentiles) == stack.shape[
			-1], f'Mismatch between the provided percentiles and the number of channels {stack.shape[-1]}. If you meant to apply the same percentiles to all channels, please provide a single tuple.'

	values = []
	for c in range(stack.shape[-1]):
		perc = percentiles[c]
		mi = np.nanpercentile(stack[:, :, :, c].flatten(), perc[0], keepdims=True)[0]
		ma = np.nanpercentile(stack[:, :, :, c].flatten(), perc[1], keepdims=True)[0]
		values.append(tuple((mi, ma)))
		gc.collect()

	return values


def get_positions_in_well(well):
	
	"""
	Retrieves the list of position directories within a specified well directory,
	formatted as a NumPy array of strings.

	This function identifies position directories based on their naming convention,
	which must include a numeric identifier following the well's name. The well's name
	is expected to start with 'W' (e.g., 'W1'), followed by a numeric identifier. Position
	directories are assumed to be named with this numeric identifier directly after the well
	identifier, without the 'W'. For example, positions within well 'W1' might be named
	'101', '102', etc. This function will glob these directories and return their full
	paths as a NumPy array.

	Parameters
	----------
	well : str
		The path to the well directory from which to retrieve position directories.

	Returns
	-------
	np.ndarray
		An array of strings, each representing the full path to a position directory within
		the specified well. The array is empty if no position directories are found.

	Notes
	-----
	- This function relies on a specific naming convention for wells and positions. It assumes
	  that each well directory is prefixed with 'W' followed by a numeric identifier, and
	  position directories are named starting with this numeric identifier directly.

	Examples
	--------
	>>> get_positions_in_well('/path/to/experiment/W1')
	# This might return an array like array(['/path/to/experiment/W1/101', '/path/to/experiment/W1/102'])
	if position directories '101' and '102' exist within the well 'W1' directory.

	"""

	if well.endswith(os.sep):
		well = well[:-1]

	w_numeric = os.path.split(well)[-1].replace('W', '')
	positions = natsorted(glob(os.sep.join([well, f'{w_numeric}*{os.sep}'])))

	return np.array(positions, dtype=str)


def extract_experiment_folder_output(experiment_folder, destination_folder):
	
	"""
	Copies the output subfolder and associated tables from an experiment folder to a new location,
	making the experiment folder much lighter by only keeping essential data.

	This function takes the path to an experiment folder and a destination folder as input.
	It creates a copy of the experiment folder at the destination, but only includes the output subfolders
	and their associated tables for each well and position within the experiment.
	This operation significantly reduces the size of the experiment data by excluding non-essential files.

	The structure of the copied experiment folder is preserved, including the configuration file,
	well directories, and position directories within each well.
	Only the 'output' subfolder and its 'tables' subdirectory are copied for each position.

	Parameters
	----------
	experiment_folder : str
		The path to the source experiment folder from which to extract data.
	destination_folder : str
		The path to the destination folder where the reduced copy of the experiment
		will be created.

	Notes
	-----
	- This function assumes that the structure of the experiment folder is consistent,
	  with wells organized in subdirectories and each containing a position subdirectory.
	  Each position subdirectory should have an 'output' folder and a 'tables' subfolder within it.

	- The function also assumes the existence of a configuration file in the root of the
	  experiment folder, which is copied to the root of the destination experiment folder.

	Examples
	--------
	>>> extract_experiment_folder_output('/path/to/experiment_folder', '/path/to/destination_folder')
	# This will copy the 'experiment_folder' to 'destination_folder', including only
	# the output subfolders and their tables for each well and position.
	
	"""
	

	if experiment_folder.endswith(os.sep):
		experiment_folder = experiment_folder[:-1]
	if destination_folder.endswith(os.sep):
		destination_folder = destination_folder[:-1]

	exp_name = experiment_folder.split(os.sep)[-1]
	output_path = os.sep.join([destination_folder, exp_name])
	if not os.path.exists(output_path):
		os.mkdir(output_path)

	config = get_config(experiment_folder)
	copyfile(config, os.sep.join([output_path, os.path.split(config)[-1]]))

	wells_src = get_experiment_wells(experiment_folder)
	wells = [w.split(os.sep)[-2] for w in wells_src]

	for k, w in enumerate(wells):

		well_output_path = os.sep.join([output_path, w])
		if not os.path.exists(well_output_path):
			os.mkdir(well_output_path)

		positions = get_positions_in_well(wells_src[k])

		for pos in positions:
			pos_name = extract_position_name(pos)
			output_pos = os.sep.join([well_output_path, pos_name])
			if not os.path.exists(output_pos):
				os.mkdir(output_pos)
			output_folder = os.sep.join([output_pos, 'output'])
			output_tables_folder = os.sep.join([output_folder, 'tables'])

			if not os.path.exists(output_folder):
				os.mkdir(output_folder)

			if not os.path.exists(output_tables_folder):
				os.mkdir(output_tables_folder)

			tab_path = glob(pos + os.sep.join(['output', 'tables', f'*']))

			for t in tab_path:
				copyfile(t, os.sep.join([output_tables_folder, os.path.split(t)[-1]]))

def _load_frames_to_segment(file, indices, scale_model=None, normalize_kwargs=None):
	
	frames = load_frames(indices, file, scale=scale_model, normalize_input=True, normalize_kwargs=normalize_kwargs)
	frames = interpolate_nan_multichannel(frames)

	if np.any(indices==-1):
		frames[:,:,np.where(indices==-1)[0]] = 0.

	return frames

def _load_frames_to_measure(file, indices):
	return load_frames(indices, file, scale=None, normalize_input=False)


def _check_label_dims(lbl, file=None, template=None):
	
	if file is not None:
		template = load_frames(0,file,scale=1,normalize_input=False)
	elif template is not None:
		template = template
	else:
		return lbl

	if lbl.shape != template.shape[:2]:
		lbl = resize(lbl, template.shape[:2], order=0)
	return lbl


if __name__ == '__main__':
	control_segmentation_napari("/home/limozin/Documents/Experiments/MinimumJan/W4/401/", prefix='Aligned',
								population="target", flush_memory=False)
