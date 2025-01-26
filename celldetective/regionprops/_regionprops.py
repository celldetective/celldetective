from skimage.measure._regionprops import RegionProperties, regionprops, _cached, _props_to_dict, _infer_number_of_required_args
import numpy as np
import inspect
import json
import os
from scipy.ndimage import find_objects

abs_path = os.sep.join([os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]])

with open(os.sep.join([abs_path, 'regionprops', 'props.json'])) as f:
	PROPS = json.load(f)

COL_DTYPES = {
	'area': float,
	'area_bbox': float,
	'area_convex': float,
	'area_filled': float,
	'axis_major_length': float,
	'axis_minor_length': float,
	'bbox': int,
	'centroid': float,
	'centroid_local': float,
	'centroid_weighted': float,
	'centroid_weighted_local': float,
	'coords': object,
	'coords_scaled': object,
	'eccentricity': float,
	'equivalent_diameter_area': float,
	'euler_number': int,
	'extent': float,
	'feret_diameter_max': float,
	'image': object,
	'image_convex': object,
	'image_filled': object,
	'image_intensity': object,
	'inertia_tensor': float,
	'inertia_tensor_eigvals': float,
	'intensity_max': float,
	'intensity_mean': float,
	'intensity_min': float,
	'intensity_std': float,
	'label': int,
	'moments': float,
	'moments_central': float,
	'moments_hu': float,
	'moments_normalized': float,
	'moments_weighted': float,
	'moments_weighted_central': float,
	'moments_weighted_hu': float,
	'moments_weighted_normalized': float,
	'num_pixels': int,
	'orientation': float,
	'perimeter': float,
	'perimeter_crofton': float,
	'slice': object,
	'solidity': float,
}

OBJECT_COLUMNS = [col for col, dtype in COL_DTYPES.items() if dtype == object]
PROP_VALS = set(PROPS.values())

class CustomRegionProps(RegionProperties):

	"""
	From https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_regionprops.py with a modification to not mask the intensity image itself before measurements
	"""

	def __init__(self, channel_names, *args, **kwargs):

		self.channel_names = channel_names
		if isinstance(self.channel_names, np.ndarray):
			self.channel_names = list(self.channel_names)
		super().__init__(*args, **kwargs)


	def __getattr__(self, attr):

		if self.channel_names is not None and self._multichannel:
			assert len(self.channel_names)==self._intensity_image.shape[-1],'Mismatch between provided channel names and the number of channels in the image...'

		if attr == "__setstate__":
			# When deserializing this object with pickle, `__setstate__`
			# is accessed before any other attributes like `self._intensity_image`
			# are available which leads to a RecursionError when trying to
			# access them later on in this function. So guard against this by
			# provoking the default AttributeError (gh-6465).
			return self.__getattribute__(attr)

		if self._intensity_image is None and attr in _require_intensity_image:
			raise AttributeError(
				f"Attribute '{attr}' unavailable when `intensity_image` "
				f"has not been specified."
			)
		if attr in self._extra_properties:
			func = self._extra_properties[attr]
			n_args = _infer_number_of_required_args(func)
			# determine whether func requires intensity image
			if n_args == 2:
				if self._intensity_image is not None:
					if self._multichannel:
						arg_dict = dict(inspect.signature(func).parameters)
						if self.channel_names is not None and 'target_channel' in arg_dict:
							multichannel_list = [np.nan for i in range(self.image_intensity.shape[-1])]
							default_channel = arg_dict['target_channel']._default
							if default_channel in self.channel_names:
								idx = self.channel_names.index(default_channel)
								multichannel_list[idx] = func(self.image, self.image_intensity[..., idx])
							else:
								print(f'Warning... Channel required by custom measurement ({default_channel}) could not be found in your data...')
							return np.stack(multichannel_list, axis=-1)
						else:
							multichannel_list = [
								func(self.image, self.image_intensity[..., i])
								for i in range(self.image_intensity.shape[-1])
							]
							return np.stack(multichannel_list, axis=-1)
					else:
						return func(self.image, self.image_intensity)
				else:
					raise AttributeError(
						f'intensity image required to calculate {attr}'
					)
			elif n_args == 1:
				return func(self.image)
			else:
				raise AttributeError(
					f'Custom regionprop function\'s number of arguments must '
					f'be 1 or 2, but {attr} takes {n_args} arguments.'
				)
		elif attr in PROPS and attr.lower() == attr:
			if (
				self._intensity_image is None
				and PROPS[attr] in _require_intensity_image
			):
				raise AttributeError(
					f"Attribute '{attr}' unavailable when `intensity_image` "
					f"has not been specified."
				)
			# retrieve deprecated property (excluding old CamelCase ones)
			return getattr(self, PROPS[attr])
		else:
			raise AttributeError(f"'{type(self)}' object has no attribute '{attr}'")

	@property
	@_cached
	def image_intensity(self):
		if self._intensity_image is None:
			raise AttributeError('No intensity image specified.')
		image = (
			self.image
			if not self._multichannel
			else np.expand_dims(self.image, self._ndim)
		)
		return self._intensity_image[self.slice]

def regionprops(label_image, intensity_image=None,cache=True,channel_names=None,*,extra_properties=None,spacing=None,offset=None):

	"""
	From https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_regionprops.py with a modification to use CustomRegionProps
	"""
	
	if label_image.ndim not in (2, 3):
		raise TypeError('Only 2-D and 3-D images supported.')
	
	if not np.issubdtype(label_image.dtype, np.integer):
		if np.issubdtype(label_image.dtype, bool):
			raise TypeError(
				'Non-integer image types are ambiguous: '
				'use skimage.measure.label to label the connected '
				'components of label_image, '
				'or label_image.astype(np.uint8) to interpret '
				'the True values as a single label.'
			)
		else:
			raise TypeError('Non-integer label_image types are ambiguous')
	
	if offset is None:
		offset_arr = np.zeros((label_image.ndim,), dtype=int)
	else:
		offset_arr = np.asarray(offset)
		if offset_arr.ndim != 1 or offset_arr.size != label_image.ndim:
			raise ValueError(
				'Offset should be an array-like of integers '
				'of shape (label_image.ndim,); '
				f'{offset} was provided.'
			)
	
	regions = []
	
	objects = find_objects(label_image)
	for i, sl in enumerate(objects):
		if sl is None:
			continue
	
		label = i + 1

		props = CustomRegionProps(
			channel_names,
			sl,
			label,
			label_image,
			intensity_image,
			cache,
			spacing=spacing,
			extra_properties=extra_properties,
			offset=offset_arr,
		)
		regions.append(props)
	
	return regions


def _props_to_dict(regions, properties=('label', 'bbox'), separator='-'):


	out = {}
	n = len(regions)
	for prop in properties:
		r = regions[0]
		# Copy the original property name so the output will have the
		# user-provided property name in the case of deprecated names.
		orig_prop = prop
		# determine the current property name for any deprecated property.
		prop = PROPS.get(prop, prop)
		rp = getattr(r, prop)
		if prop in COL_DTYPES:
			dtype = COL_DTYPES[prop]
		else:
			func = r._extra_properties[prop]
			# dtype = _infer_regionprop_dtype(
			# 	func,
			# 	intensity=r._intensity_image is not None,
			# 	ndim=r.image.ndim,
			# )
			dtype = np.float64

		# scalars and objects are dedicated one column per prop
		# array properties are raveled into multiple columns
		# for more info, refer to notes 1
		if np.isscalar(rp) or prop in OBJECT_COLUMNS or dtype is np.object_:
			column_buffer = np.empty(n, dtype=dtype)
			for i in range(n):
				column_buffer[i] = regions[i][prop]
			out[orig_prop] = np.copy(column_buffer)
		else:
			# precompute property column names and locations
			modified_props = []
			locs = []
			for ind in np.ndindex(np.shape(rp)):
				modified_props.append(separator.join(map(str, (orig_prop,) + ind)))
				locs.append(ind if len(ind) > 1 else ind[0])

			# fill temporary column data_array
			n_columns = len(locs)
			column_data = np.empty((n, n_columns), dtype=dtype)
			for k in range(n):
				# we coerce to a numpy array to ensure structures like
				# tuple-of-arrays expand correctly into columns
				rp = np.asarray(regions[k][prop])
				for i, loc in enumerate(locs):
					column_data[k, i] = rp[loc]

			# add the columns to the output dictionary
			for i, modified_prop in enumerate(modified_props):
				out[modified_prop] = column_data[:, i]
	return out


def regionprops_table(label_image,intensity_image=None,properties=('label', 'bbox'),*,cache=True,separator='-',extra_properties=None,spacing=None,channel_names=None):
	
	"""
	From https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_regionprops.py
	"""
	regions = regionprops(
		label_image,
		intensity_image=intensity_image,
		cache=cache,
		extra_properties=extra_properties,
		spacing=spacing,
		channel_names=channel_names,
	)
	if extra_properties is not None:
		properties = list(properties) + [prop.__name__ for prop in extra_properties]
	if len(regions) == 0:
		ndim = label_image.ndim
		label_image = np.zeros((3,) * ndim, dtype=int)
		label_image[(1,) * ndim] = 1
		if intensity_image is not None:
			intensity_image = np.zeros(
				label_image.shape + intensity_image.shape[ndim:],
				dtype=intensity_image.dtype,
			)
		regions = regionprops(
			label_image,
			intensity_image=intensity_image,
			cache=cache,
			extra_properties=extra_properties,
			spacing=spacing,
			channel_names=channel_names,
		)
		out_d = _props_to_dict(regions, properties=properties, separator=separator)
		return {k: v[:0] for k, v in out_d.items()}

	good_props = []
	for prop in properties:
		nan_test = [np.isnan(getattr(r,prop)) for r in regions]
		if not np.all(nan_test):
			good_props.append(prop)

	return _props_to_dict(regions, properties=good_props, separator=separator)