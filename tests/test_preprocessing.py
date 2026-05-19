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

class TestFourierRegistration(unittest.TestCase):

	def test_fourier_registration_single_stack(self):
		import tempfile
		from celldetective.preprocessing import register_stack_fourier_single_stack
		from celldetective.utils.io import save_tiff_imagej_compatible

		# Create a synthetic 100x100 image with a central circle
		y, x = np.ogrid[:100, :100]
		mask = (y - 50)**2 + (x - 50)**2 <= 15**2
		base_img = np.zeros((100, 100))
		base_img[mask] = 100.0

		# Frame 0: unshifted
		# Frame 1: shifted by 3 pixels vertically and -2 horizontally
		# Frame 2: shifted by -1 pixels vertically and 4 horizontally
		frame0 = base_img.copy()
		
		frame1_mask = (y - 53)**2 + (x - 48)**2 <= 15**2
		frame1 = np.zeros((100, 100))
		frame1[frame1_mask] = 100.0

		frame2_mask = (y - 49)**2 + (x - 54)**2 <= 15**2
		frame2 = np.zeros((100, 100))
		frame2[frame2_mask] = 100.0

		# Stack shape needs to be (3, 100, 100)
		stack = np.stack([frame0, frame1, frame2], axis=0)
		
		with tempfile.TemporaryDirectory() as tmpdir:
			tmp_path = os.path.join(tmpdir, "test_stack.tif")
			save_tiff_imagej_compatible(
				tmp_path, stack.astype(np.float32), axes="TYX"
			)

			# Run the fourier registration
			aligned = register_stack_fourier_single_stack(
				tmp_path,
				target_channel_index=0,
				nbr_channels=1,
				stack_length=3,
				reference_frame_idx=0,
				upsample_factor=10, # subpixel alignment
				export=False,
				return_stacks=True
			)

			self.assertIsNotNone(aligned)
			self.assertEqual(aligned.shape, (3, 100, 100, 1))

			aligned_frame0 = aligned[0, :, :, 0]
			aligned_frame1 = aligned[1, :, :, 0]
			aligned_frame2 = aligned[2, :, :, 0]

			# Check center 80x80 region
			self.assertTrue(np.allclose(aligned_frame0[10:90, 10:90], frame0[10:90, 10:90], atol=5))
			self.assertTrue(np.allclose(aligned_frame1[10:90, 10:90], frame0[10:90, 10:90], atol=5))
			self.assertTrue(np.allclose(aligned_frame2[10:90, 10:90], frame0[10:90, 10:90], atol=5))

	def test_fourier_domain_shift_registration(self):
		import tempfile
		from celldetective.preprocessing import register_stack_fourier_single_stack
		from celldetective.utils.io import save_tiff_imagej_compatible

		# Create a synthetic 100x100 image with a central circle
		y, x = np.ogrid[:100, :100]
		mask = (y - 50)**2 + (x - 50)**2 <= 15**2
		base_img = np.zeros((100, 100))
		base_img[mask] = 100.0

		# Frame 0: unshifted
		# Frame 1: shifted by 3 pixels vertically and -2 horizontally
		frame0 = base_img.copy()
		
		frame1_mask = (y - 53)**2 + (x - 48)**2 <= 15**2
		frame1 = np.zeros((100, 100))
		frame1[frame1_mask] = 100.0

		# Add some NaNs to test NaN resistance in fourier shift
		frame0[15:20, 15:20] = np.nan
		frame1[18:23, 13:18] = np.nan  # approximately shifted NaN region

		# Stack shape needs to be (2, 100, 100)
		stack = np.stack([frame0, frame1], axis=0)
		
		with tempfile.TemporaryDirectory() as tmpdir:
			tmp_path = os.path.join(tmpdir, "test_fourier_shift_stack.tif")
			save_tiff_imagej_compatible(
				tmp_path, stack.astype(np.float32), axes="TYX"
			)

			# Run the fourier registration with shift_method="fourier"
			aligned = register_stack_fourier_single_stack(
				tmp_path,
				target_channel_index=0,
				nbr_channels=1,
				stack_length=2,
				reference_frame_idx=0,
				upsample_factor=10, # subpixel alignment
				shift_method="fourier",
				export=False,
				return_stacks=True
			)

			self.assertIsNotNone(aligned)
			self.assertEqual(aligned.shape, (2, 100, 100, 1))

			aligned_frame0 = aligned[0, :, :, 0]
			aligned_frame1 = aligned[1, :, :, 0]

			# Check center 80x80 region, ignoring NaNs
			valid_mask = ~np.isnan(frame0)
			# Ignore boundaries
			valid_mask[:20, :] = False
			valid_mask[-20:, :] = False
			valid_mask[:, :20] = False
			valid_mask[:, -20:] = False

			self.assertTrue(np.allclose(aligned_frame0[valid_mask], frame0[valid_mask], atol=5))
			self.assertTrue(np.allclose(aligned_frame1[valid_mask], frame0[valid_mask], atol=5))

	def test_fourier_registration_single_stack_sliding(self):
		import tempfile
		from celldetective.preprocessing import register_stack_fourier_single_stack
		from celldetective.utils.io import save_tiff_imagej_compatible

		# Create a synthetic 100x100 image with a central circle
		y, x = np.ogrid[:100, :100]
		mask = (y - 50)**2 + (x - 50)**2 <= 15**2
		base_img = np.zeros((100, 100))
		base_img[mask] = 100.0

		# Frame 0: unshifted
		# Frame 1: shifted by 3 pixels vertically and -2 horizontally (relative to frame 0)
		# Frame 2: shifted by 2 pixels vertically and 2 horizontally (relative to frame 1)
		# -> cumulative shift of Frame 2 relative to Frame 0 is (3+2)=5 vertically and (-2+2)=0 horizontally
		frame0 = base_img.copy()
		
		frame1_mask = (y - 53)**2 + (x - 48)**2 <= 15**2
		frame1 = np.zeros((100, 100))
		frame1[frame1_mask] = 100.0

		frame2_mask = (y - 55)**2 + (x - 48)**2 <= 15**2
		frame2 = np.zeros((100, 100))
		frame2[frame2_mask] = 100.0

		# Stack shape needs to be (3, 100, 100)
		stack = np.stack([frame0, frame1, frame2], axis=0)
		
		with tempfile.TemporaryDirectory() as tmpdir:
			tmp_path = os.path.join(tmpdir, "test_stack_sliding.tif")
			save_tiff_imagej_compatible(
				tmp_path, stack.astype(np.float32), axes="TYX"
			)

			# Run the fourier registration with sliding=True
			aligned = register_stack_fourier_single_stack(
				tmp_path,
				target_channel_index=0,
				nbr_channels=1,
				stack_length=3,
				reference_frame_idx=0,
				upsample_factor=10, # subpixel alignment
				sliding=True,
				export=False,
				return_stacks=True
			)

			self.assertIsNotNone(aligned)
			self.assertEqual(aligned.shape, (3, 100, 100, 1))

			aligned_frame0 = aligned[0, :, :, 0]
			aligned_frame1 = aligned[1, :, :, 0]
			aligned_frame2 = aligned[2, :, :, 0]

			# Check center 80x80 region
			self.assertTrue(np.allclose(aligned_frame0[10:90, 10:90], frame0[10:90, 10:90], atol=5))
			self.assertTrue(np.allclose(aligned_frame1[10:90, 10:90], frame0[10:90, 10:90], atol=5))
			self.assertTrue(np.allclose(aligned_frame2[10:90, 10:90], frame0[10:90, 10:90], atol=5))

	def test_fourier_registration_gaussian_smoothing(self):
		import tempfile
		from celldetective.preprocessing import register_stack_fourier_single_stack
		from celldetective.utils.io import save_tiff_imagej_compatible

		# Create a synthetic 100x100 image with a central circle
		y, x = np.ogrid[:100, :100]
		mask = (y - 50)**2 + (x - 50)**2 <= 15**2
		base_img = np.zeros((100, 100))
		base_img[mask] = 100.0

		frame0 = base_img.copy()
		frame1_mask = (y - 52)**2 + (x - 48)**2 <= 15**2
		frame1 = np.zeros((100, 100))
		frame1[frame1_mask] = 100.0

		stack = np.stack([frame0, frame1], axis=0)

		with tempfile.TemporaryDirectory() as tmpdir:
			tmp_path = os.path.join(tmpdir, "test_stack_smooth.tif")
			save_tiff_imagej_compatible(tmp_path, stack.astype(np.float32), axes="TYX")

			# Run fourier registration with Gaussian pre-smoothing
			aligned = register_stack_fourier_single_stack(
				tmp_path,
				target_channel_index=0,
				nbr_channels=1,
				stack_length=2,
				reference_frame_idx=0,
				upsample_factor=1,
				sigma=1.5, # Gaussian smoothing active
				export=False,
				return_stacks=True
			)
			self.assertIsNotNone(aligned)
			self.assertEqual(aligned.shape, (2, 100, 100, 1))

	def test_fourier_registration_max_shift_clipping(self):
		import tempfile
		from celldetective.preprocessing import register_stack_fourier_single_stack
		from celldetective.utils.io import save_tiff_imagej_compatible

		y, x = np.ogrid[:100, :100]
		mask = (y - 50)**2 + (x - 50)**2 <= 10**2
		base_img = np.zeros((100, 100))
		base_img[mask] = 100.0

		# Frame 0: [0, 0] shift
		frame0 = base_img.copy()
		# Frame 1: [2, 2] shift
		frame1_mask = (y - 52)**2 + (x - 52)**2 <= 10**2
		frame1 = np.zeros((100, 100))
		frame1[frame1_mask] = 100.0
		# Frame 2: [30, 30] shift (massive false-positive spike)
		frame2_mask = (y - 80)**2 + (x - 80)**2 <= 10**2
		frame2 = np.zeros((100, 100))
		frame2[frame2_mask] = 100.0

		stack = np.stack([frame0, frame1, frame2], axis=0)

		with tempfile.TemporaryDirectory() as tmpdir:
			tmp_path = os.path.join(tmpdir, "test_stack_clipping.tif")
			save_tiff_imagej_compatible(tmp_path, stack.astype(np.float32), axes="TYX")

			# Run fourier registration with max_shift limit = 5.0 pixels
			aligned = register_stack_fourier_single_stack(
				tmp_path,
				target_channel_index=0,
				nbr_channels=1,
				stack_length=3,
				reference_frame_idx=0,
				upsample_factor=1,
				max_shift=5.0, # shift threshold limit
				export=False,
				return_stacks=True
			)
			self.assertIsNotNone(aligned)
			
			# Check that Frame 2's massive shift was rejected.
			# If it was rejected, it is aligned with the fallback shift [2, 2] instead of [30, 30].
			# Let's verify that the center region of frame 2 is NOT perfectly aligned back to frame 0,
			# showing that the 30px shift was indeed rejected!
			aligned_frame2 = aligned[2, :, :, 0]
			# If [30, 30] was aligned, it would overlap with base_img. But since it's rejected,
			# it should be far away from base_img.
			overlap_with_original_center = np.sum(aligned_frame2[mask])
			self.assertEqual(overlap_with_original_center, 0.0)

	def test_fourier_registration_median_filtering(self):
		import tempfile
		from celldetective.preprocessing import register_stack_fourier_single_stack
		from celldetective.utils.io import save_tiff_imagej_compatible

		y, x = np.ogrid[:100, :100]
		mask = (y - 50)**2 + (x - 50)**2 <= 10**2
		base_img = np.zeros((100, 100))
		base_img[mask] = 100.0

		# 5-frame stack:
		# Frame 0: [0, 0] shift
		frame0 = base_img.copy()
		# Frame 1: [1, 1] shift
		frame1 = np.zeros((100, 100))
		frame1[(y - 51)**2 + (x - 51)**2 <= 10**2] = 100.0
		# Frame 2: [30, 30] shift (massive false-positive spike)
		frame2 = np.zeros((100, 100))
		frame2[(y - 80)**2 + (x - 80)**2 <= 10**2] = 100.0
		# Frame 3: [3, 3] shift
		frame3 = np.zeros((100, 100))
		frame3[(y - 53)**2 + (x - 53)**2 <= 10**2] = 100.0
		# Frame 4: [4, 4] shift
		frame4 = np.zeros((100, 100))
		frame4[(y - 54)**2 + (x - 54)**2 <= 10**2] = 100.0

		stack = np.stack([frame0, frame1, frame2, frame3, frame4], axis=0)

		with tempfile.TemporaryDirectory() as tmpdir:
			tmp_path = os.path.join(tmpdir, "test_stack_median.tif")
			save_tiff_imagej_compatible(tmp_path, stack.astype(np.float32), axes="TYX")

			# Run fourier registration with filter_outliers=True
			aligned = register_stack_fourier_single_stack(
				tmp_path,
				target_channel_index=0,
				nbr_channels=1,
				stack_length=5,
				reference_frame_idx=0,
				upsample_factor=1,
				filter_outliers=True, # Outlier trajectory filtering active
				export=False,
				return_stacks=True
			)
			self.assertIsNotNone(aligned)
			
			# The median filter of window size 3 on shifts [0, 1, 30, 3, 4] will yield:
			# medfilt([0, 1, 30, 3, 4]) -> [0, 1, 3, 3, 4]
			# So frame 2's shift is filtered from 30 to 3.
			# Let's verify that the center region of frame 2 is shifted by 3 instead of 30,
			# so when aligned, it should be close to frame 3's aligned state.
			aligned_frame2 = aligned[2, :, :, 0]
			expected_aligned_frame2_mask = (y - 77)**2 + (x - 77)**2 <= 10**2
			expected_aligned_frame2 = np.zeros((100, 100))
			expected_aligned_frame2[expected_aligned_frame2_mask] = 100.0
			
			# Check that aligned_frame2 matches the expected shift of 3 pixels (meaning it was filtered to 3px)
			self.assertTrue(np.allclose(aligned_frame2[10:90, 10:90], expected_aligned_frame2[10:90, 10:90], atol=5))

	def test_sift_registration_single_stack(self):
		import tempfile
		from celldetective.preprocessing import register_stack_fourier_single_stack
		from celldetective.utils.io import save_tiff_imagej_compatible
		from scipy.ndimage import shift

		# Create a patterned image with multiple dots so SIFT finds plenty of keypoints
		base_img = np.zeros((100, 100))
		y, x = np.ogrid[:100, :100]
		centers = [(30, 30), (70, 30), (30, 70), (70, 70), (50, 50)]
		for cy, cx in centers:
			mask = (y - cy)**2 + (x - cx)**2 <= 6**2
			base_img[mask] = 150.0

		frame0 = base_img.copy()
		frame1 = shift(base_img, [3.0, -2.0])
		frame2 = shift(base_img, [-1.0, 4.0])

		stack = np.stack([frame0, frame1, frame2], axis=0)
		
		with tempfile.TemporaryDirectory() as tmpdir:
			tmp_path = os.path.join(tmpdir, "test_stack_sift.tif")
			save_tiff_imagej_compatible(tmp_path, stack.astype(np.float32), axes="TYX")

			aligned = register_stack_fourier_single_stack(
				tmp_path,
				target_channel_index=0,
				nbr_channels=1,
				stack_length=3,
				reference_frame_idx=0,
				method="sift",
				export=False,
				return_stacks=True
			)

			self.assertIsNotNone(aligned)
			self.assertEqual(aligned.shape, (3, 100, 100, 1))

			aligned_frame0 = aligned[0, :, :, 0]
			aligned_frame1 = aligned[1, :, :, 0]
			aligned_frame2 = aligned[2, :, :, 0]

			# Check center region using Gaussian pre-smoothing to avoid sharp-edge subpixel interpolation mismatch
			from scipy.ndimage import gaussian_filter
			smoothed_f0 = gaussian_filter(frame0, sigma=3.0)
			smoothed_f1 = gaussian_filter(aligned_frame1, sigma=3.0)
			smoothed_f2 = gaussian_filter(aligned_frame2, sigma=3.0)

			self.assertTrue(np.allclose(aligned_frame0[15:85, 15:85], frame0[15:85, 15:85], atol=5))
			self.assertTrue(np.allclose(smoothed_f1[15:85, 15:85], smoothed_f0[15:85, 15:85], atol=5))
			self.assertTrue(np.allclose(smoothed_f2[15:85, 15:85], smoothed_f0[15:85, 15:85], atol=5))

	def test_hybrid_registration_single_stack(self):
		import tempfile
		from celldetective.preprocessing import register_stack_fourier_single_stack
		from celldetective.utils.io import save_tiff_imagej_compatible
		from scipy.ndimage import shift

		# Create a patterned image with multiple dots
		base_img = np.zeros((100, 100))
		y, x = np.ogrid[:100, :100]
		centers = [(30, 30), (70, 30), (30, 70), (70, 70), (50, 50)]
		for cy, cx in centers:
			mask = (y - cy)**2 + (x - cx)**2 <= 6**2
			base_img[mask] = 150.0

		frame0 = base_img.copy()
		frame1 = shift(base_img, [3.0, -2.0])
		# Frame 2 is heavily blurred to force SIFT to fail and fallback to Fourier
		from scipy.ndimage import gaussian_filter
		frame2 = gaussian_filter(shift(base_img, [-1.0, 4.0]), sigma=15.0)

		stack = np.stack([frame0, frame1, frame2], axis=0)
		
		with tempfile.TemporaryDirectory() as tmpdir:
			tmp_path = os.path.join(tmpdir, "test_stack_hybrid.tif")
			save_tiff_imagej_compatible(tmp_path, stack.astype(np.float32), axes="TYX")

			aligned = register_stack_fourier_single_stack(
				tmp_path,
				target_channel_index=0,
				nbr_channels=1,
				stack_length=3,
				reference_frame_idx=0,
				method="hybrid",
				export=False,
				return_stacks=True
			)

			self.assertIsNotNone(aligned)
			self.assertEqual(aligned.shape, (3, 100, 100, 1))

			aligned_frame0 = aligned[0, :, :, 0]
			aligned_frame1 = aligned[1, :, :, 0]
			aligned_frame2 = aligned[2, :, :, 0]

			# Check aligned frames using Gaussian pre-smoothing to avoid sharp-edge subpixel interpolation mismatch
			from scipy.ndimage import gaussian_filter
			smoothed_f0 = gaussian_filter(frame0, sigma=3.0)
			smoothed_f1 = gaussian_filter(aligned_frame1, sigma=3.0)
			smoothed_f2 = gaussian_filter(aligned_frame2, sigma=3.0)

			# Frame 1 (SIFT succeeded) should be aligned
			self.assertTrue(np.allclose(smoothed_f1[15:85, 15:85], smoothed_f0[15:85, 15:85], atol=5))
			
			# Frame 2 (SIFT failed, Fourier fell back) should also be successfully aligned (compare to blurred f0)
			expected_aligned_f2 = gaussian_filter(frame0, sigma=15.0)
			smoothed_expected_f2 = gaussian_filter(expected_aligned_f2, sigma=3.0)
			self.assertTrue(np.allclose(smoothed_f2[15:85, 15:85], smoothed_expected_f2[15:85, 15:85], atol=15))


class TestMultiChannelConsensus(unittest.TestCase):

	def test_multi_channel_consensus(self):
		import tempfile
		from celldetective.preprocessing import register_stack_fourier_single_stack
		from celldetective.utils.io import save_tiff_imagej_compatible

		# Create a synthetic 100x100 image with a central circle for Channel 0
		y, x = np.ogrid[:100, :100]
		mask_ch0 = (y - 50)**2 + (x - 50)**2 <= 15**2
		base_ch0 = np.zeros((100, 100))
		base_ch0[mask_ch0] = 100.0

		# Create a synthetic 100x100 image with a central square for Channel 1
		mask_ch1 = (np.abs(y - 50) <= 12) & (np.abs(x - 50) <= 12)
		base_ch1 = np.zeros((100, 100))
		base_ch1[mask_ch1] = 80.0

		# Shift both by same amount for Frame 1
		frame0_ch0 = base_ch0.copy()
		frame0_ch1 = base_ch1.copy()

		frame1_ch0_mask = (y - 53)**2 + (x - 48)**2 <= 15**2
		frame1_ch0 = np.zeros((100, 100))
		frame1_ch0[frame1_ch0_mask] = 100.0

		frame1_ch1_mask = (np.abs(y - 53) <= 12) & (np.abs(x - 48) <= 12)
		frame1_ch1 = np.zeros((100, 100))
		frame1_ch1[frame1_ch1_mask] = 80.0

		# Stack pages interleaved: [f0_c0, f0_c1, f1_c0, f1_c1]
		stack = np.stack([frame0_ch0, frame0_ch1, frame1_ch0, frame1_ch1], axis=0)

		with tempfile.TemporaryDirectory() as tmpdir:
			tmp_path = os.path.join(tmpdir, "test_consensus_stack.tif")
			save_tiff_imagej_compatible(
				tmp_path, stack.astype(np.float32), axes="TYX"
			)

			# Run registration with both channels as consensus target
			aligned = register_stack_fourier_single_stack(
				tmp_path,
				target_channel_index=[0, 1],
				nbr_channels=2,
				stack_length=2,
				reference_frame_idx=0,
				upsample_factor=10,
				export=False,
				return_stacks=True,
				method="fourier"
			)

			self.assertIsNotNone(aligned)
			# aligned should have shape (stack_length, Y, X, nbr_channels) -> (2, 100, 100, 2)
			self.assertEqual(aligned.shape, (2, 100, 100, 2))

			aligned_f1_ch0 = aligned[1, :, :, 0]
			aligned_f1_ch1 = aligned[1, :, :, 1]

			# Check that both channels are successfully registered back to reference frame0
			self.assertTrue(np.allclose(aligned_f1_ch0[15:85, 15:85], frame0_ch0[15:85, 15:85], atol=5))
			self.assertTrue(np.allclose(aligned_f1_ch1[15:85, 15:85], frame0_ch1[15:85, 15:85], atol=5))


if __name__=="__main__":
	unittest.main()