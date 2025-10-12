import unittest
import matplotlib.pyplot as plt
import numpy as np
import os
from celldetective.preprocessing import fit_background_model, field_correction

import matplotlib.pyplot as plt

class TestFitPlane(unittest.TestCase):

	@classmethod
	def setUpClass(self):
		a = 5.
		self.img = np.full((100,100),5.0)
		self.img_with_cell = self.img.copy()
		self.img_with_cell[:10,:10] = 25.0

	def test_plane_is_well_fit(self):
		mat = np.array(fit_background_model(self.img, cell_masks=None, model='plane', edge_exclusion=None))
		self.assertTrue(np.allclose(self.img, mat))

	def test_plane_is_well_fit_and_applied_with_division(self):
		result = field_correction(self.img, threshold=1.0E05, operation='divide', model='plane', clip=False, return_bg=False, activation_protocol=[])
		self.assertTrue(np.allclose(result, np.full((100,100), 1.0)))

	def test_plane_is_well_fit_and_applied_with_subtraction(self):
		result = field_correction(self.img, threshold=1.0E05, operation='subtract', model='plane', clip=False, return_bg=False, activation_protocol=[])
		self.assertTrue(np.allclose(result, np.zeros((100,100))))

	def test_plane_is_well_fit_with_cell(self):
		cell_masks = np.zeros_like(self.img)
		cell_masks[:10,:10] = 1.0
		mat = np.array(fit_background_model(self.img, cell_masks=cell_masks, model='plane', edge_exclusion=None))
		self.assertTrue(np.allclose(self.img, mat))

if __name__=="__main__":
	unittest.main()