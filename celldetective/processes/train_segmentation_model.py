from multiprocessing import Process
import time
import os
import shutil
from glob import glob
import json
from tqdm import tqdm
import numpy as np
import random

from celldetective.utils import load_image_dataset, augmenter, interpolate_nan
from celldetective.io import normalize_multichannel
from stardist import fill_label_holes
from art import tprint
from distutils.dir_util import copy_tree
from csbdeep.utils import save_json


class TrainSegModelProcess(Process):

	def __init__(self, queue=None, process_args=None, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		
		self.queue = queue

		if process_args is not None:
			for key, value in process_args.items():
				setattr(self, key, value)

		tprint("Train segmentation")
		self.read_instructions()
		self.extract_training_params()
		self.load_dataset()
		self.split_test_train()

		self.sum_done = 0
		self.t0 = time.time()

	def read_instructions(self):

		if os.path.exists(self.instructions):
			with open(self.instructions, 'r') as f:
				self.training_instructions = json.load(f)
		else:
			print('Training instructions could not be found. Abort.')
			self.abort_process()

	def run(self):

		if self.model_type=="cellpose":
			self.train_cellpose_model()
		elif self.model_type=="stardist":
			self.train_stardist_model()

		self.queue.put("finished")
		self.queue.close()

	def train_stardist_model(self):

		from stardist import calculate_extents, gputools_available
		from stardist.models import Config2D, StarDist2D
		
		n_rays = 32
		print(gputools_available())
		
		n_channel = self.X_trn[0].shape[-1]
		
		# Predict on subsampled grid for increased efficiency and larger field of view
		grid = (2,2)
		conf = Config2D(
			n_rays       = n_rays,
			grid         = grid,
			use_gpu      = self.use_gpu,
			n_channel_in = n_channel,
			train_learning_rate = self.learning_rate,
			train_patch_size = (256,256),
			train_epochs = self.epochs,
			train_reduce_lr = {'factor': 0.1, 'patience': 30, 'min_delta': 0},
			train_batch_size = self.batch_size,
			train_steps_per_epoch = int(self.augmentation_factor*len(self.X_trn)),
		)
		
		if self.use_gpu:
			from csbdeep.utils.tf import limit_gpu_memory
			limit_gpu_memory(None, allow_growth=True)

		if self.pretrained is None:
			model = StarDist2D(conf, name=self.model_name, basedir=self.target_directory)
		else:
			os.rename(self.instructions, os.sep.join([self.target_directory, self.model_name, 'temp.json']))
			copy_tree(self.pretrained, os.sep.join([self.target_directory, self.model_name]))
			
			if os.path.exists(os.sep.join([self.target_directory, self.model_name, 'training_instructions.json'])):
				os.remove(os.sep.join([self.target_directory, self.model_name, 'training_instructions.json']))
			if os.path.exists(os.sep.join([self.target_directory, self.model_name, 'config_input.json'])):
				os.remove(os.sep.join([self.target_directory, self.model_name, 'config_input.json']))
			if os.path.exists(os.sep.join([self.target_directory, self.model_name, 'logs'+os.sep])):
				shutil.rmtree(os.sep.join([self.target_directory, self.model_name, 'logs']))
			os.rename(os.sep.join([self.target_directory, self.model_name, 'temp.json']),os.sep.join([self.target_directory, self.model_name, 'training_instructions.json']))

			#shutil.copytree(pretrained, os.sep.join([target_directory, model_name]))
			model = StarDist2D(None, name=self.model_name, basedir=self.target_directory)
			model.config.train_epochs = self.epochs
			model.config.train_batch_size = min(len(self.X_trn),self.batch_size)
			model.config.train_learning_rate = self.learning_rate # perf seems bad if lr is changed in transfer
			model.config.use_gpu = self.use_gpu
			model.config.train_reduce_lr = {'factor': 0.1, 'patience': 10, 'min_delta': 0}
			print(f'{model.config=}')

			save_json(vars(model.config), os.sep.join([self.target_directory, self.model_name, 'config.json']))

		median_size = calculate_extents(list(self.Y_trn), np.mean)
		fov = np.array(model._axes_tile_overlap('YX'))
		print(f"median object size:      {median_size}")
		print(f"network field of view :  {fov}")
		if any(median_size > fov):
			print("WARNING: median object size larger than field of view of the neural network.")

		if self.augmentation_factor==1.0:
			model.train(self.X_trn, self.Y_trn, validation_data=(self.X_val,self.Y_val))
		else:
			model.train(self.X_trn, self.Y_trn, validation_data=(self.X_val,self.Y_val), augmenter=augmenter)
		model.optimize_thresholds(self.X_val,self.Y_val)

		config_inputs = {"channels": self.target_channels, 'normalization_percentile': self.normalization_percentile,
		'normalization_clip': self.normalization_clip, 'normalization_values': self.normalization_values, 
		'model_type': 'stardist', 'spatial_calibration': self.spatial_calibration, 'dataset': {'train': self.files_train, 'validation': self.files_val}}

		json_input_config = json.dumps(config_inputs, indent=4)
		with open(os.sep.join([self.target_directory, self.model_name, "config_input.json"]), "w") as outfile:
			outfile.write(json_input_config)

	def train_cellpose_model(self):

		# do augmentation in place
		X_aug = []; Y_aug = [];
		n_val = max(1, int(round(self.augmentation_factor * len(self.X_trn))))
		indices = random.choices(list(np.arange(len(self.X_trn))), k=n_val)
		print('Performing image augmentation pre-training...')
		for i in tqdm(indices):
			x_aug,y_aug = augmenter(self.X_trn[i], self.Y_trn[i])
			X_aug.append(x_aug)
			Y_aug.append(y_aug)
		
		# Channel axis in front for cellpose
		X_aug = [np.moveaxis(x,-1,0) for x in X_aug]
		self.X_val = [np.moveaxis(x,-1,0) for x in self.X_val]
		print('number of augmented images: %3d' % len(X_aug))

		from cellpose.models import CellposeModel
		from cellpose.io import logger_setup
		import torch
		
		if not self.use_gpu:
			print('Using CPU for training...')
			device = torch.device("cpu")
		else:
			print('Using GPU for training...')
			
		logger, log_file = logger_setup()
		print(f'Pretrained model: ', self.pretrained)
		if self.pretrained is not None:
			pretrained_path = os.sep.join([self.pretrained,os.path.split(self.pretrained)[-1]])
		else:
			pretrained_path = self.pretrained
		
		model = CellposeModel(gpu=self.use_gpu, model_type=None, pretrained_model=pretrained_path, diam_mean=30.0, nchan=X_aug[0].shape[0],) 
		model.train(train_data=X_aug, train_labels=Y_aug, normalize=False, channels=None, batch_size=self.batch_size,
					min_train_masks=1,save_path=self.target_directory+os.sep+self.model_name,n_epochs=self.epochs, model_name=self.model_name, learning_rate=self.learning_rate, test_data = self.X_val, test_labels=self.Y_val)

		file_to_move = glob(os.sep.join([self.target_directory, self.model_name, 'models','*']))[0]
		shutil.move(file_to_move, os.sep.join([self.target_directory, self.model_name,''])+os.path.split(file_to_move)[-1])
		os.rmdir(os.sep.join([self.target_directory, self.model_name, 'models']))

		diameter = model.diam_labels

		if self.pretrained is not None and os.path.split(self.pretrained)[-1]=='CP_nuclei':
			standard_diameter = 17.0
		else:
			standard_diameter = 30.0

		input_spatial_calibration = self.spatial_calibration #*diameter / standard_diameter

		config_inputs = {"channels": self.target_channels, "diameter": standard_diameter, 'cellprob_threshold': 0., 'flow_threshold': 0.4,
		'normalization_percentile': self.normalization_percentile, 'normalization_clip': self.normalization_clip,
		'normalization_values': self.normalization_values, 'model_type': 'cellpose',
		'spatial_calibration': input_spatial_calibration, 'dataset': {'train': self.files_train, 'validation': self.files_val}}
		json_input_config = json.dumps(config_inputs, indent=4)
		with open(os.sep.join([self.target_directory, self.model_name, "config_input.json"]), "w") as outfile:
			outfile.write(json_input_config)


	def split_test_train(self):

		if not len(self.X) > 1:
			print("Not enough training data")
			self.abort_process()

		rng = np.random.RandomState()
		ind = rng.permutation(len(self.X))
		n_val = max(1, int(round(self.validation_split * len(ind))))
		ind_train, ind_val = ind[:-n_val], ind[-n_val:]
		self.X_val, self.Y_val = [self.X[i] for i in ind_val]  , [self.Y[i] for i in ind_val]
		self.X_trn, self.Y_trn = [self.X[i] for i in ind_train], [self.Y[i] for i in ind_train]

		self.files_train = [self.filenames[i] for i in ind_train]
		self.files_val = [self.filenames[i] for i in ind_val]

		print('number of images: %3d' % len(self.X))
		print('- training:       %3d' % len(self.X_trn))
		print('- validation:     %3d' % len(self.X_val))

	def extract_training_params(self):

		self.model_name = self.training_instructions['model_name']
		self.target_directory = self.training_instructions['target_directory']
		self.model_type = self.training_instructions['model_type']
		self.pretrained = self.training_instructions['pretrained']

		self.datasets = self.training_instructions['ds']

		self.target_channels = self.training_instructions['channel_option']
		self.normalization_percentile = self.training_instructions['normalization_percentile']
		self.normalization_clip = self.training_instructions['normalization_clip']
		self.normalization_values = self.training_instructions['normalization_values']
		self.spatial_calibration = self.training_instructions['spatial_calibration']

		self.validation_split = self.training_instructions['validation_split']
		self.augmentation_factor = self.training_instructions['augmentation_factor']

		self.learning_rate = self.training_instructions['learning_rate']
		self.epochs = self.training_instructions['epochs']
		self.batch_size = self.training_instructions['batch_size']

	def load_dataset(self):

		print(f'Datasets: {self.datasets}')
		self.X,self.Y,self.filenames = load_image_dataset(self.datasets, self.target_channels, train_spatial_calibration=self.spatial_calibration,
								mask_suffix='labelled')
		print('Dataset loaded...')

		self.values = []
		self.percentiles = []
		for k in range(len(self.normalization_percentile)):
			if self.normalization_percentile[k]:
				self.percentiles.append(self.normalization_values[k])
				self.values.append(None)
			else:
				self.percentiles.append(None)
				self.values.append(self.normalization_values[k])

		self.X = [normalize_multichannel(x, **{"percentiles": self.percentiles, 'values': self.values, 'clip': self.normalization_clip}) for x in self.X]

		for k in range(len(self.X)):
			x = self.X[k].copy()
			x_interp = np.moveaxis([interpolate_nan(x[:,:,c].copy()) for c in range(x.shape[-1])],0,-1)
			self.X[k] = x_interp

		self.Y = [fill_label_holes(y) for y in tqdm(self.Y)]

	def end_process(self):

		self.terminate()
		self.queue.put("finished")

	def abort_process(self):
		
		self.terminate()
		self.queue.put("error")