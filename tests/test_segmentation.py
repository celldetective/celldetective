import unittest
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tifffile import imread
from celldetective.segmentation import segment, segment_frame_from_thresholds
from tensorflow.keras.metrics import BinaryIoU

TEST_IMAGE_FILENAME = os.path.join(os.path.dirname(__file__), os.sep.join(['assets','sample.tif']))
TEST_LABEL_FILENAME = os.path.join(os.path.dirname(__file__), os.sep.join(['assets','sample_labelled.tif']))
TEST_CONFIG_FILENAME = os.path.join(os.path.dirname(__file__), os.sep.join(['assets','sample.json']))

class TestDLMCF7Segmentation(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.img = imread(TEST_IMAGE_FILENAME)
		self.label_true = imread(TEST_LABEL_FILENAME)
		self.stack = np.moveaxis([self.img, self.img, self.img],1,-1)
		with open(TEST_CONFIG_FILENAME) as config_file:
			self.config = json.load(config_file)
		self.channels = self.config['channels']
		print(f'{self.channels=}')
		self.spatial_calibration = self.config['spatial_calibration']

	def test_correct_segmentation_with_multimodal_model(self):
		
		labels = segment(self.stack, "mcf7_nuc_multimodal", channels=self.channels, spatial_calibration=self.spatial_calibration, view_on_napari=False,
						use_gpu=False)
		np.testing.assert_array_equal(labels[0], labels[1])

		self.binary_label_true = self.label_true.copy().astype(float)
		self.binary_label_true[self.binary_label_true>0] = 1.

		label_binary = labels[0].copy().astype(float)
		label_binary[label_binary>0] = 1.

		m = BinaryIoU(target_class_ids=[1])
		m.update_state(self.binary_label_true, label_binary)
		score = m.result().numpy()

		self.assertGreater(score,0.85)

	def test_correct_segmentation_with_transferred_model(self):
		
		labels = segment(self.stack, "mcf7_nuc_stardist_transfer", channels=self.channels, spatial_calibration=self.spatial_calibration, view_on_napari=False,
			use_gpu=True)
		np.testing.assert_array_equal(labels[0], labels[1])

		self.binary_label_true = self.label_true.copy().astype(float)
		self.binary_label_true[self.binary_label_true>0] = 1.

		label_binary = labels[0].copy().astype(float)
		label_binary[label_binary>0] = 1.

		m = BinaryIoU(target_class_ids=[1])
		m.update_state(self.binary_label_true, label_binary)
		score = m.result().numpy()

		self.assertGreater(score,0.85)


class TestThresholdMCF7Segmentation(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		self.img = imread(TEST_IMAGE_FILENAME)
		self.label_true = imread(TEST_LABEL_FILENAME)
		with open(TEST_CONFIG_FILENAME) as config_file:
			self.config = json.load(config_file)
		self.channels = self.config['channels']
		self.spatial_calibration = self.config['spatial_calibration']

	def test_correct_segmentation_with_threshold(self):
		
		label = segment_frame_from_thresholds(np.moveaxis(self.img,0,-1), target_channel=3, thresholds=[8000,1.0E10], equalize_reference=None,
								  filters=[['variance',4],['gauss',2]], marker_min_distance=13, marker_footprint_size=34, marker_footprint=None, feature_queries=["area < 80"], channel_names=None)
		
		self.binary_label_true = self.label_true.copy().astype(float)
		self.binary_label_true[self.binary_label_true>0] = 1.

		label_binary = label.copy().astype(float)
		label_binary[label_binary>0] = 1.

		m = BinaryIoU(target_class_ids=[1])
		m.update_state(self.binary_label_true, label_binary)
		score = m.result().numpy()

		self.assertGreater(score,0.7)

	def test_threshold_segmentation_with_nan_pixels(self):

		# NaN pixels (e.g. left by a background correction) go through interpolate_nan
		frame = np.moveaxis(self.img,0,-1).astype(float)
		frame[:10,:10,3] = np.nan
		label = segment_frame_from_thresholds(frame, target_channel=3, thresholds=[8000,1.0E10], equalize_reference=None,
								  filters=[['variance',4],['gauss',2]], marker_min_distance=13, marker_footprint_size=34, marker_footprint=None, feature_queries=["area < 80"], channel_names=None)

		self.assertEqual(label.shape, frame.shape[:2])
		self.assertGreater(label.max(), 0)


if __name__=="__main__":
	unittest.main()