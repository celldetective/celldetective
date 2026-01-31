import unittest
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import pandas as pd
import shutil

def sigmoid(t,t0,dt,A,offset):
	return A/(1+np.exp(-(t-t0)/dt)) + offset

def generate_fake_signal_data(n_signals):
	
	timeline = np.linspace(0,100,100)
	amplitudes = list(np.linspace(2000,3000,100))
	slopes = list(np.linspace(0.5,5,100))
	means = list(np.linspace(-100,200,100))
	random_cut = list(np.linspace(25,200,176,dtype=int))
	noise_levels = list(np.linspace(1,100,100,dtype=int))

	trajectories = []
	for i in range(n_signals):
		
		a = random.sample(amplitudes,k=1)[0]
		dt = random.sample(slopes,k=1)[0]
		mu = random.sample(means,k=1)[0]
		cut = random.sample(random_cut,k=1)[0]
		n = random.sample(noise_levels,k=1)[0]
		
		if mu<=0.:
			cclass=2
			t0=-1
		elif (mu>0)*(mu<=100):
			cclass=0
			t0=mu
		else:
			cclass=1
			t0=-1
		
		noise = [random.random()*n for i in range(len(timeline))]
		signal = sigmoid(timeline, mu, dt,a,0)+noise
		signal = signal[:cut]
		if mu>=cut:
			cclass=1
			t0=-1
		
		for j in range(len(signal)):
			trajectories.append({'TRACK_ID': i, 'POSITION_X': 0., 'POSITION_Y': 0., 'FRAME': j,'signal': signal[j], 't0': t0, 'cclass': cclass})  

	trajectories = pd.DataFrame(trajectories)

	return trajectories

def export_set(trajectories, name='set.npy', output_folder='.'):
	
	training_set = []
	cols = trajectories.columns
	tracks = np.unique(trajectories["TRACK_ID"].to_numpy())

	for track in tracks:
		signals = {}
		for c in cols:
			signals.update({c: trajectories.loc[trajectories["TRACK_ID"] == track, c].to_numpy()})
		time_of_interest = trajectories.loc[trajectories["TRACK_ID"] == track, "t0"].to_numpy()[0]
		cclass = trajectories.loc[trajectories["TRACK_ID"] == track, "cclass"].to_numpy()[0]
		signals.update({"time_of_interest": time_of_interest, "class": cclass})
		training_set.append(signals)
		
	np.save(os.sep.join([output_folder,name]), training_set)


class TestCreateSignalModel(unittest.TestCase):

	def test_create_model(self):

		from celldetective.event_detection_models import SignalDetectionModel

        model = SignalDetectionModel(
									 channel_option=["signal"], 
									 model_signal_length=128,
									 n_channels=1,
									 n_conv=2,
									 n_classes=3,
									 dense_collection=512,
									 dropout_rate=0.1,
									 label='test',
									 )


class TestTrainSignalModel(unittest.TestCase):

	@classmethod
	def setUpClass(self):

		from celldetective.event_detection_models import SignalDetectionModel

        self.trajectories = generate_fake_signal_data(300)
		if not os.path.exists('temp'):
			os.mkdir('temp')
		export_set(self.trajectories, name='set.npy', output_folder='temp')
		self.model = SignalDetectionModel(
									 channel_option=["signal"], 
									 model_signal_length=128,
									 n_channels=1,
									 n_conv=2,
									 n_classes=3,
									 dense_collection=512,
									 dropout_rate=0.1,
									 label='test',
									 )

	def test_train_signal_model(self):

		self.model.fit_from_directory(
									 ['temp'],
									 normalize=True,
									 normalization_percentile=None,
									 normalization_values = None,
									 normalization_clip = None,
									 channel_option=["signal"],
									 target_directory='temp',
									 augment=False,
									 model_name='None',
									 validation_split=0.2,
									 test_split=0.1,
									 batch_size = 16,
									 epochs=1,
									 recompile_pretrained=False,
									 learning_rate=0.01,
									 show_plots=False,
		                        )
		shutil.rmtree('temp')

if __name__=="__main__":
	unittest.main()