from multiprocessing import Process
import time
import datetime
import os
import json
from celldetective.io import auto_load_number_of_frames, _load_frames_to_measure, locate_labels
from celldetective.utils import extract_experiment_channels, ConfigSectionMap, _get_img_num_per_channel, _mask_intensity_measurements
from pathlib import Path, PurePath
from glob import glob
from tqdm import tqdm
import numpy as np
import gc
import concurrent.futures
import datetime
import os
import json
from celldetective.io import interpret_tracking_configuration
from celldetective.utils import extract_experiment_channels
from celldetective.measure import drop_tonal_features, measure_features
from celldetective.tracking import track
import pandas as pd
from natsort import natsorted
from art import tprint


class TrackingProcess(Process):

	def __init__(self, queue=None, process_args=None, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		
		self.queue = queue

		if process_args is not None:
			for key, value in process_args.items():
				setattr(self, key, value)


		tprint("Track")
		self.timestep_dataframes = []

		# Experiment		
		self.prepare_folders()

		self.locate_experiment_config()
		self.extract_experiment_parameters()
		self.read_tracking_instructions()
		self.detect_movie_and_labels()
		self.detect_channels()

		self.write_log()

		if not self.btrack_option:
			self.features = []
			self.channel_names = None
			self.haralick_options = None

		self.sum_done = 0
		self.t0 = time.time()

	def read_tracking_instructions(self):

		instr_path = PurePath(self.exp_dir,Path(f"{self.instruction_file}"))
		if os.path.exists(instr_path):
			print(f"Tracking instructions for the {self.mode} population have been successfully loaded...")
			with open(instr_path, 'r') as f:
				self.instructions = json.load(f)
			
			self.btrack_config = interpret_tracking_configuration(self.instructions['btrack_config_path'])

			if 'features' in self.instructions:
				self.features = self.instructions['features']
			else:
				self.features = None
			
			if 'mask_channels' in self.instructions:
				self.mask_channels = self.instructions['mask_channels']
			else:
				self.mask_channels = None
			
			if 'haralick_options' in self.instructions:
				self.haralick_options = self.instructions['haralick_options']
			else:
				self.haralick_options = None

			if 'post_processing_options' in self.instructions:
				self.post_processing_options = self.instructions['post_processing_options']
			else:
				self.post_processing_options = None
			
			self.btrack_option = True
			if 'btrack_option' in self.instructions:
				self.btrack_option = self.instructions['btrack_option']
			self.search_range = None
			if 'search_range' in self.instructions:
				self.search_range = self.instructions['search_range']
			self.memory = None
			if 'memory' in self.instructions:
				self.memory = self.instructions['memory']
		else:
			print('Tracking instructions could not be located... Using a standard bTrack motion model instead...')
			self.btrack_config = interpret_tracking_configuration(None)
			self.features = None
			self.mask_channels = None
			self.haralick_options = None
			self.post_processing_options = None
			self.btrack_option = True
			self.memory = None
			self.search_range = None

		if self.features is None:
			self.features = []

	def detect_channels(self):
		self.img_num_channels = _get_img_num_per_channel(self.channel_indices, self.len_movie, self.nbr_channels)

	def write_log(self):

		features_log=f'features: {self.features}'
		mask_channels_log=f'mask_channels: {self.mask_channels}'
		haralick_option_log=f'haralick_options: {self.haralick_options}'
		post_processing_option_log=f'post_processing_options: {self.post_processing_options}'
		log_list=[features_log, mask_channels_log, haralick_option_log, post_processing_option_log]
		log='\n'.join(log_list)

		with open(self.pos+f'log_{self.mode}.txt', 'a') as f:
			f.write(f'{datetime.datetime.now()} TRACK \n')
			f.write(log+"\n")

	def prepare_folders(self):

		if not os.path.exists(self.pos+"output"):
			os.mkdir(self.pos+"output")

		if not os.path.exists(self.pos+os.sep.join(["output","tables"])):
			os.mkdir(self.pos+os.sep.join(["output","tables"]))

		if self.mode.lower()=="target" or self.mode.lower()=="targets":
			self.label_folder = "labels_targets"
			self.instruction_file = os.sep.join(["configs", "tracking_instructions_targets.json"])
			self.napari_name = "napari_target_trajectories.npy"
			self.table_name = "trajectories_targets.csv"

		elif self.mode.lower()=="effector" or self.mode.lower()=="effectors":
			self.label_folder = "labels_effectors"
			self.instruction_file = os.sep.join(["configs","tracking_instructions_effectors.json"])
			self.napari_name = "napari_effector_trajectories.npy"
			self.table_name = "trajectories_effectors.csv"

	def extract_experiment_parameters(self):

		self.movie_prefix = ConfigSectionMap(self.config,"MovieSettings")["movie_prefix"]
		self.spatial_calibration = float(ConfigSectionMap(self.config,"MovieSettings")["pxtoum"])
		self.time_calibration = float(ConfigSectionMap(self.config,"MovieSettings")["frametomin"])
		self.len_movie = float(ConfigSectionMap(self.config,"MovieSettings")["len_movie"])
		self.shape_x = int(ConfigSectionMap(self.config,"MovieSettings")["shape_x"])
		self.shape_y = int(ConfigSectionMap(self.config,"MovieSettings")["shape_y"])

		self.channel_names, self.channel_indices = extract_experiment_channels(self.exp_dir)
		self.nbr_channels = len(self.channel_names)

	def locate_experiment_config(self):

		parent1 = Path(self.pos).parent
		self.exp_dir = parent1.parent
		self.config = PurePath(self.exp_dir,Path("config.ini"))

		if not os.path.exists(self.config):
			print('The configuration file for the experiment was not found...')
			self.abort_process()

	def detect_movie_and_labels(self):

		self.label_path = natsorted(glob(self.pos+f"{self.label_folder}"+os.sep+"*.tif"))
		if len(self.label_path)>0:
			print(f"Found {len(self.label_path)} segmented frames...")
		else:
			print(f"No segmented frames have been found. Please run segmentation first. Abort...")
			self.abort_process()

		try:
			self.file = glob(self.pos+f"movie/{self.movie_prefix}*.tif")[0]
		except IndexError:
			self.file = None
			self.haralick_option = None
			self.features = drop_tonal_features(self.features)
			print('Movie could not be found. Check the prefix.')

		len_movie_auto = auto_load_number_of_frames(self.file)
		if len_movie_auto is not None:
			self.len_movie = len_movie_auto

	def parallel_job(self, indices):

		props = []

		try:

			for t in tqdm(indices,desc="frame"):
				
				# Load channels at time t
				img = _load_frames_to_measure(self.file, indices=self.img_num_channels[:,t])
				lbl = locate_labels(self.pos, population=self.mode, frames=t)
				if lbl is None:
					continue

				df_props = measure_features(img, lbl, features = self.features+['centroid'], border_dist=None, 
												channels=self.channel_names, haralick_options=self.haralick_options, verbose=False)
				df_props.rename(columns={'centroid-1': 'x', 'centroid-0': 'y'},inplace=True)
				df_props['t'] = int(t)

				props.append(df_props)

				self.sum_done+=1/self.len_movie*50
				mean_exec_per_step = (time.time() - self.t0) / (self.sum_done*self.len_movie / 50 + 1)
				pred_time = (self.len_movie - (self.sum_done*self.len_movie / 50 + 1)) * mean_exec_per_step + 30
				self.queue.put([self.sum_done, pred_time])
				

		except Exception as e:
			print(e)

		return props

	def run(self):

		self.indices = list(range(self.img_num_channels.shape[1]))
		chunks = np.array_split(self.indices, self.n_threads)

		self.timestep_dataframes = []
		with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
			results = executor.map(self.parallel_job, chunks)
			try:
				for i,return_value in enumerate(results):
					print(f'Thread {i} completed...')
					#print(f"Thread {i} output check: ",return_value)
					self.timestep_dataframes.extend(return_value)
			except Exception as e:
				print("Exception: ", e)

		print('Features successfully measured...')

		df = pd.concat(self.timestep_dataframes)	
		df.reset_index(inplace=True, drop=True)
		df = _mask_intensity_measurements(df, self.mask_channels)

		# do tracking
		if self.btrack_option:
			tracker = 'bTrack'
		else:
			tracker = 'trackpy'

		# do tracking
		trajectories, napari_data = track(None,
							configuration=self.btrack_config,
							objects=df, 
							spatial_calibration=self.spatial_calibration, 
							channel_names=self.channel_names,
							return_napari_data=True,
				  			optimizer_options = {'tm_lim': int(12e4)}, 
				  			track_kwargs={'step_size': 100}, 
				  			clean_trajectories_kwargs=self.post_processing_options, 
				  			volume=(self.shape_x, self.shape_y),
				  			btrack_option=self.btrack_option, 
				  			search_range=self.search_range,
				  			memory=self.memory,
				  			)
		print(f"Tracking successfully performed...")

		# out trajectory table, create POSITION_X_um, POSITION_Y_um, TIME_min (new ones)
		# Save napari data
		np.save(self.pos+os.sep.join(['output', 'tables', self.napari_name]), napari_data, allow_pickle=True)

		trajectories.to_csv(self.pos+os.sep.join(['output', 'tables', self.table_name]), index=False)
		print(f"Trajectory table successfully exported in {os.sep.join(['output', 'tables'])}...")

		if os.path.exists(self.pos+os.sep.join(['output', 'tables', self.table_name.replace('.csv','.pkl')])):
			os.remove(self.pos+os.sep.join(['output', 'tables', self.table_name.replace('.csv','.pkl')]))

		del trajectories; del napari_data;
		gc.collect()		
		
		# Send end signal
		self.queue.put([100, 0])
		time.sleep(1)
		
		self.queue.put("finished")
		self.queue.close()

	def end_process(self):

		self.terminate()
		self.queue.put("finished")

	def abort_process(self):
		
		self.terminate()
		self.queue.put("error")