import unittest
import numpy as np
from celldetective.filters import gauss_filter, abs_filter, filter_image


class TestFilters(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.img = np.ones((256, 256), dtype=int)
        self.img[100:110, 100:110] = 0
        self.gauss_sigma = 1.6

    def test_gauss_filter_is_float(self):
        self.assertIsInstance(gauss_filter(self.img, self.gauss_sigma)[0, 0], float)

    def test_gauss_filter_has_same_shape(self):
        self.assertEqual(gauss_filter(self.img, self.gauss_sigma).shape, self.img.shape)

    def test_abs_filter_is_positive(self):
        self.assertTrue(np.all(abs_filter(self.img) >= 0.0))

    def test_filter_image_none(self):
        # Should return original image if filters is None
        res = filter_image(self.img, filters=None)
        np.testing.assert_array_equal(res, self.img)

    def test_filter_image_single(self):
        # Test with a single filter: e.g. abs
        # Create an image with negatives
        img_neg = self.img.copy() * -1
        res = filter_image(img_neg, filters=[("abs",)])
        self.assertTrue(np.all(res >= 0))
        np.testing.assert_array_almost_equal(res, np.abs(img_neg))

    def test_filter_image_chain(self):
        # Test chaining: subtract 10 then abs
        # Start with ones. Subtract 10 -> -9. Abs -> 9.
        img = np.ones((5, 5), dtype=float)
        filters = [("subtract", 10), ("abs",)]
        res = filter_image(img, filters=filters)
        expected = np.abs(img - 10)
        np.testing.assert_array_almost_equal(res, expected)
        self.assertTrue(np.allclose(res, 9.0))


if __name__ == "__main__":
    unittest.main()
