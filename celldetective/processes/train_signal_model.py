from multiprocessing import Process
import time
import os
import json
from glob import glob
import numpy as np
from art import tprint
from celldetective.signals import SignalDetectionModel
from celldetective.io import locate_signal_model


class TrainSignalModelProcess(Process):

	def __init__(self, queue=None, process_args=None, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		
		self.queue = queue

		if process_args is not None:
			for key, value in process_args.items():
				setattr(self, key, value)

		tprint("Train segmentation")
		self.read_instructions()
		self.extract_training_params()


		self.sum_done = 0
		self.t0 = time.time()

	def read_instructions(self):

		if os.path.exists(self.instructions):
			with open(self.instructions, 'r') as f:
				self.training_instructions = json.load(f)
		else:
			print('Training instructions could not be found. Abort.')
			self.abort_process()

		all_classes = []
		for d in self.training_instructions["ds"]:
			datasets = glob(d+os.sep+"*.npy")
			for dd in datasets:
				data = np.load(dd, allow_pickle=True)
				classes = np.unique([ddd["class"] for ddd in data])
				all_classes.extend(classes)
		all_classes = np.unique(all_classes)
		n_classes = len(all_classes)

		self.model_params = {k:self.training_instructions[k] for k in ('pretrained', 'model_signal_length', 'channel_option', 'n_channels', 'label') if k in self.training_instructions}
		self.model_params.update({'n_classes': n_classes})
		self.train_params = {k:self.training_instructions[k] for k in ('model_name', 'target_directory', 'channel_option','recompile_pretrained', 'test_split', 'augment', 'epochs', 'learning_rate', 'batch_size', 'validation_split','normalization_percentile','normalization_values','normalization_clip') if k in self.training_instructions}

	def neighborhood_postprocessing(self):

		# if neighborhood of interest in training instructions, write it in config!
		if 'neighborhood_of_interest' in self.training_instructions:
			if self.training_instructions['neighborhood_of_interest'] is not None:
				
				model_path = locate_signal_model(self.training_instructions['model_name'], path=None, pairs=True)
				complete_path = model_path #+model
				complete_path = rf"{complete_path}"
				model_config_path = os.sep.join([complete_path,'config_input.json'])
				model_config_path = rf"{model_config_path}"

				f = open(model_config_path)
				config = json.load(f)
				config.update({'neighborhood_of_interest': self.training_instructions['neighborhood_of_interest'], 'reference_population': self.training_instructions['reference_population'], 'neighbor_population': self.training_instructions['neighbor_population']})
				json_string = json.dumps(config)
				with open(model_config_path, 'w') as outfile:
					outfile.write(json_string)

	def run(self):

		model = SignalDetectionModel(**self.model_params)
		model.fit_from_directory(self.training_instructions['ds'], **self.train_params)
		self.neighborhood_postprocessing()
		self.queue.put("finished")
		self.queue.close()


	def extract_training_params(self):

		self.training_instructions.update({'n_channels': len(self.training_instructions['channel_option'])})
		if self.training_instructions['augmentation_factor']>1.0:
			self.training_instructions.update({'augment': True})
		else:
			self.training_instructions.update({'augment': False})
		self.training_instructions.update({'test_split': 0.})


	def end_process(self):
		
		# self.terminate()

		# if self.model_type=="stardist":
		# 	from stardist.models import StarDist2D
		# 	self.model = StarDist2D(None, name=self.model_name, basedir=self.target_directory)
		# 	self.model.optimize_thresholds(self.X_val,self.Y_val)

		self.terminate()
		self.queue.put("finished")

	def abort_process(self):
		
		self.terminate()
		self.queue.put("error")