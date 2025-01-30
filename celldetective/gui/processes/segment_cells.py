from multiprocessing import Process
import time
import datetime
import os
import json
import numpy as np
from celldetective.io import extract_position_name, locate_segmentation_model, auto_load_number_of_frames, load_frames, _check_label_dims, _load_frames_to_segment
from celldetective.utils import _rescale_labels, _segment_image_with_stardist_model, _segment_image_with_cellpose_model, _prep_stardist_model, _prep_cellpose_model, _get_normalize_kwargs_from_config, extract_experiment_channels, _estimate_scale_factor, _extract_channel_indices_from_config, ConfigSectionMap, _extract_nbr_channels_from_config, _get_img_num_per_channel

from pathlib import Path, PurePath
from glob import glob
from shutil import rmtree
from tqdm import tqdm
import numpy as np
from csbdeep.io import save_tiff_imagej_compatible
from celldetective.segmentation import segment_frame_from_thresholds, merge_instance_segmentation
import gc
from art import tprint

import concurrent.futures

class BaseSegmentProcess(Process):

	def __init__(self, queue=None, process_args=None, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		
		self.queue = queue

		if process_args is not None:
			for key, value in process_args.items():
				setattr(self, key, value)

		tprint("Segment")

		# Experiment
		self.locate_experiment_config()

		print(f"Position: {extract_position_name(self.pos)}...")
		print("Configuration file: ",self.config)
		print(f"Population: {self.mode}...")

		self.extract_experiment_parameters()
		self.detect_movie_length()
		self.write_folders()

	def write_folders(self):

		self.mode = self.mode.lower()

		if self.mode=="target" or self.mode=="targets":
			self.label_folder = "labels_targets"
		elif self.mode=="effector" or self.mode=="effectors":
			self.label_folder = "labels_effectors"

		if os.path.exists(self.pos+self.label_folder):
			print('Erasing the previous labels folder...')
			rmtree(self.pos+self.label_folder)
		os.mkdir(self.pos+self.label_folder)
		print(f'Labels folder successfully generated...')


	def extract_experiment_parameters(self):

		self.spatial_calibration = float(ConfigSectionMap(self.config,"MovieSettings")["pxtoum"])
		self.len_movie = float(ConfigSectionMap(self.config,"MovieSettings")["len_movie"])
		self.movie_prefix = ConfigSectionMap(self.config,"MovieSettings")["movie_prefix"]
		self.nbr_channels = _extract_nbr_channels_from_config(self.config)
		self.channel_names, self.channel_indices = extract_experiment_channels(self.exp_dir)

	def locate_experiment_config(self):

		parent1 = Path(self.pos).parent
		self.exp_dir = parent1.parent
		self.config = PurePath(self.exp_dir,Path("config.ini"))

		if not os.path.exists(self.config):
			print('The configuration file for the experiment could not be located. Abort.')
			self.abort_process()

	def detect_movie_length(self):

		try:
			self.file = glob(self.pos+f"movie/{self.movie_prefix}*.tif")[0]
		except Exception as e:
			print(f'Error {e}.\nMovie could not be found. Check the prefix.')
			self.abort_process()

		len_movie_auto = auto_load_number_of_frames(self.file)
		if len_movie_auto is not None:
			self.len_movie = len_movie_auto

	def end_process(self):

		self.terminate()
		self.queue.put("finished")

	def abort_process(self):
		
		self.terminate()
		self.queue.put("error")


class SegmentCellDLProcess(BaseSegmentProcess):
	
	def __init__(self, *args, **kwargs):
		
		super().__init__(*args, **kwargs)

		self.check_gpu()

		# Model
		self.locate_model_path()
		self.extract_model_input_parameters()
		self.detect_channels()
		self.detect_rescaling()

		self.write_log()

		self.sum_done = 0
		self.t0 = time.time()

	def extract_model_input_parameters(self):

		self.required_channels = self.input_config["channels"]
		self.normalize_kwargs = _get_normalize_kwargs_from_config(self.input_config)

		self.model_type = self.input_config['model_type']
		self.required_spatial_calibration = self.input_config['spatial_calibration']
		print(f'Spatial calibration expected by the model: {self.required_spatial_calibration}...')
		
		if self.model_type=='cellpose':
			self.diameter = self.input_config['diameter']
			self.cellprob_threshold = self.input_config['cellprob_threshold']
			self.flow_threshold = self.input_config['flow_threshold']

	def write_log(self):

		log=f'segmentation model: {self.model_name}\n'
		with open(self.pos+f'log_{self.mode}.txt', 'a') as f:
			f.write(f'{datetime.datetime.now()} SEGMENT \n')
			f.write(log)

	def detect_channels(self):
		
		self.channel_indices = _extract_channel_indices_from_config(self.config, self.required_channels)
		print(f'Required channels: {self.required_channels} located at channel indices {self.channel_indices}.')
		self.img_num_channels = _get_img_num_per_channel(self.channel_indices, int(self.len_movie), self.nbr_channels)

	def detect_rescaling(self):

		self.scale = _estimate_scale_factor(self.spatial_calibration, self.required_spatial_calibration)
		print(f"Scale: {self.scale}...")

	def locate_model_path(self):

		self.model_complete_path = locate_segmentation_model(self.model_name)
		if self.model_complete_path is None:
			print('Model could not be found. Abort.')
			self.abort_process()
		else:
			print(f'Model path: {self.model_complete_path}...')
		
		if not os.path.exists(self.model_complete_path+"config_input.json"):
			print('The configuration for the inputs to the model could not be located. Abort.')
			self.abort_process()

		with open(self.model_complete_path+"config_input.json") as config_file:
			self.input_config = json.load(config_file)
					
	def check_gpu(self):

		if not self.use_gpu:
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	def run(self):

		try:

			if self.model_type=='stardist':
				model, scale_model = _prep_stardist_model(self.model_name, Path(self.model_complete_path).parent, use_gpu=self.use_gpu, scale=self.scale)

			elif self.model_type=='cellpose':
				model, scale_model = _prep_cellpose_model(self.model_name, self.model_complete_path, use_gpu=self.use_gpu, n_channels=len(self.required_channels), scale=self.scale)

			for t in tqdm(range(self.len_movie),desc="frame"):
				
				f = _load_frames_to_segment(self.file, self.img_num_channels[:,t], scale_model=scale_model, normalize_kwargs=self.normalize_kwargs)

				if self.model_type=="stardist":
					Y_pred = _segment_image_with_stardist_model(f, model=model, return_details=False)

				elif self.model_type=="cellpose":
					Y_pred = _segment_image_with_cellpose_model(f, model=model, diameter=self.diameter, cellprob_threshold=self.cellprob_threshold, flow_threshold=self.flow_threshold)

				if self.scale is not None:
					Y_pred = _rescale_labels(Y_pred, scale_model=scale_model)
				
				Y_pred = _check_label_dims(Y_pred, file=self.file)

				save_tiff_imagej_compatible(self.pos+os.sep.join([self.label_folder,f"{str(t).zfill(4)}.tif"]), Y_pred, axes='YX')
				
				del f;
				del Y_pred;
				gc.collect()
				
				# Send signal for progress bar
				self.sum_done+=1/self.len_movie*100
				mean_exec_per_step = (time.time() - self.t0) / (t+1)
				pred_time = (self.len_movie - (t+1)) * mean_exec_per_step
				self.queue.put([self.sum_done, pred_time])

		except Exception as e:
			print(e)

		try:
			del model
		except:
			pass		
		
		gc.collect()

		# Send end signal
		self.queue.put("finished")
		self.queue.close()


class SegmentCellThresholdProcess(BaseSegmentProcess):
	
	def __init__(self, *args, **kwargs):
		
		super().__init__(*args, **kwargs)

		self.equalize = False

		# Model

		self.load_threshold_config()
		self.extract_threshold_parameters()
		self.detect_channels()
		self.prepare_equalize()

		self.write_log()

		self.sum_done = 0
		self.t0 = time.time()

	def prepare_equalize(self):

		for i in range(len(self.instructions)):

			if self.equalize[i]:
				f_reference = load_frames(self.img_num_channels[:,self.equalize_time[i]], self.file, scale=None, normalize_input=False)
				f_reference = f_reference[:,:,self.instructions[i]['target_channel']]
			else:
				f_reference = None

			self.instructions[i].update({'equalize_reference': f_reference})

	def load_threshold_config(self):

		self.instructions = []
		for inst in self.threshold_instructions:
			if os.path.exists(inst):
				with open(inst, 'r') as f:
					self.instructions.append(json.load(f))
			else:
				print('The configuration path is not valid. Abort.')
				self.abort_process()

	def extract_threshold_parameters(self):
		
		self.required_channels = []
		self.equalize = []
		self.equalize_time = []

		for i in range(len(self.instructions)):
			ch = [self.instructions[i]['target_channel']]
			self.required_channels.append(ch)

			if 'equalize_reference' in self.instructions[i]:
				equalize, equalize_time = self.instructions[i]['equalize_reference']	
				self.equalize.append(equalize)
				self.equalize_time.append(equalize_time)

	def write_log(self):

		log=f'Threshold segmentation: {self.threshold_instructions}\n'
		with open(self.pos+f'log_{self.mode}.txt', 'a') as f:
			f.write(f'{datetime.datetime.now()} SEGMENT \n')
			f.write(log)

	def detect_channels(self):

		for i in range(len(self.instructions)):
			
			self.channel_indices = _extract_channel_indices_from_config(self.config, self.required_channels[i])
			print(f'Required channels: {self.required_channels[i]} located at channel indices {self.channel_indices}.')
			self.instructions[i].update({'target_channel': self.channel_indices[0]})
			self.instructions[i].update({'channel_names': self.channel_names})
		
		self.img_num_channels = _get_img_num_per_channel(np.arange(self.nbr_channels), self.len_movie, self.nbr_channels)

	def parallel_job(self, indices):

		try:

			for t in tqdm(indices,desc="frame"): #for t in tqdm(range(self.len_movie),desc="frame"):
				
				# Load channels at time t
				masks = []
				for i in range(len(self.instructions)):
					f = load_frames(self.img_num_channels[:,t], self.file, scale=None, normalize_input=False)
					mask = segment_frame_from_thresholds(f, **self.instructions[i])
					#print(f'Frame {t}; segment with {self.instructions[i]=}...')
					masks.append(mask)

				if len(self.instructions)>1:
					mask = merge_instance_segmentation(masks, mode='OR')

				save_tiff_imagej_compatible(os.sep.join([self.pos, self.label_folder, f"{str(t).zfill(4)}.tif"]), mask.astype(np.uint16), axes='YX')

				del f;
				del mask;
				gc.collect()

				# Send signal for progress bar
				self.sum_done+=1/self.len_movie*100
				mean_exec_per_step = (time.time() - self.t0) / (self.sum_done*self.len_movie / 100 + 1)
				pred_time = (self.len_movie - (self.sum_done*self.len_movie / 100 + 1)) * mean_exec_per_step
				self.queue.put([self.sum_done, pred_time])

		except Exception as e:
			print(e)

		return
			

	def run(self):

		self.indices = list(range(self.img_num_channels.shape[1]))
		chunks = np.array_split(self.indices, self.n_threads)

		with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
			results = results = executor.map(self.parallel_job, chunks) #list(map(lambda x: executor.submit(self.parallel_job, x), chunks))
			try:
				for i,return_value in enumerate(results):
					print(f"Thread {i} output check: ",return_value)
			except Exception as e:
				print("Exception: ", e)

		# Send end signal
		self.queue.put("finished")
		self.queue.close()