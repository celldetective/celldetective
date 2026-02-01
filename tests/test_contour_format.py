"""
Unit tests for contour measurement format parsing and interpretation.

Tests the complete pipeline:
1. CellEdgeVisualizer output format: "(min,max)"
2. ListWidget.getItems() parsing
3. contour_of_instance_segmentation() interpretation
4. measure_features() with border_distances
"""

import unittest
import numpy as np
import pandas as pd


class TestContourFormatParsing(unittest.TestCase):
    """
    Test that the contour format "(min,max)" is correctly parsed by list_widget.getItems()
    and contour_of_instance_segmentation().
    """

    def test_tuple_format_parsing_positive(self):
        """Test parsing of positive range like (0,5)"""
        from celldetective.utils.masks import contour_of_instance_segmentation

        # Create a simple label with a square object
        label = np.zeros((50, 50), dtype=int)
        label[15:35, 15:35] = 1  # 20x20 square

        # Test with tuple format string "(0,5)" - inner contour 0-5px
        result = contour_of_instance_segmentation(label, "(0,5)")

        # Should have non-zero pixels (edge region exists)
        self.assertGreater(np.sum(result > 0), 0, "Contour should have pixels")

        # The result should be smaller than the original object
        self.assertLess(
            np.sum(result > 0), np.sum(label > 0), "Edge should be smaller than object"
        )

    def test_tuple_format_parsing_negative(self):
        """Test parsing of negative range like (-5,0) - outer contour"""
        from celldetective.utils.masks import contour_of_instance_segmentation

        label = np.zeros((50, 50), dtype=int)
        label[15:35, 15:35] = 1

        # Test with tuple format string "(-5,0)" - outer contour
        result = contour_of_instance_segmentation(label, "(-5,0)")

        # Should have non-zero pixels in the region outside the original object
        self.assertGreater(np.sum(result > 0), 0, "Outer contour should have pixels")

    def test_tuple_format_parsing_mixed(self):
        """Test parsing of mixed range like (-3,3) - crossing boundary"""
        from celldetective.utils.masks import contour_of_instance_segmentation

        label = np.zeros((50, 50), dtype=int)
        label[15:35, 15:35] = 1

        # Test with tuple format string "(-3,3)" - straddles boundary
        result = contour_of_instance_segmentation(label, "(-3,3)")

        # Should have non-zero pixels
        self.assertGreater(np.sum(result > 0), 0, "Mixed contour should have pixels")

    def test_list_format_direct(self):
        """Test that list format [min, max] works correctly"""
        from celldetective.utils.masks import contour_of_instance_segmentation

        label = np.zeros((50, 50), dtype=int)
        label[15:35, 15:35] = 1

        # Test with list format [-5, 5]
        result = contour_of_instance_segmentation(label, [-5, 5])

        self.assertGreater(np.sum(result > 0), 0, "List format should work")

    def test_tuple_format_direct(self):
        """Test that tuple format (min, max) works correctly"""
        from celldetective.utils.masks import contour_of_instance_segmentation

        label = np.zeros((50, 50), dtype=int)
        label[15:35, 15:35] = 1

        # Test with tuple format (-5, 5)
        result = contour_of_instance_segmentation(label, (-5, 5))

        self.assertGreater(np.sum(result > 0), 0, "Tuple format should work")


class TestListWidgetParsing(unittest.TestCase):
    """
    Test that ListWidget.getItems() correctly parses the "(min,max)" format.
    """

    def setUp(self):
        """Set up a mock list widget for testing."""
        from unittest.mock import MagicMock

        # Create a mock item that returns text
        self.mock_item = MagicMock()
        self.mock_list_widget = MagicMock()

    def test_parse_tuple_format(self):
        """Test that getItems parses (min,max) format correctly."""
        import re

        # Simulate the parsing logic from getItems
        tuple_pattern = re.compile(r"^\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)$")

        test_cases = [
            ("(0,5)", [0, 5]),
            ("(-5,0)", [-5, 0]),
            ("(-10,10)", [-10, 10]),
            ("(-3,3)", [-3, 3]),
            ("(0, 10)", [0, 10]),  # with space
        ]

        for text, expected in test_cases:
            match = tuple_pattern.match(text.strip())
            self.assertIsNotNone(match, f"Should match pattern: {text}")
            minn = int(float(match.group(1)))
            maxx = int(float(match.group(2)))
            self.assertEqual([minn, maxx], expected, f"Failed for: {text}")

    def test_parse_single_value(self):
        """Test that single values are parsed correctly."""
        import re

        tuple_pattern = re.compile(r"^\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)$")

        test_cases = ["5", "10", "-5", "0"]

        for text in test_cases:
            match = tuple_pattern.match(text.strip())
            self.assertIsNone(match, f"Should NOT match tuple pattern: {text}")
            # Should be parseable as int
            val = int(text)
            self.assertIsInstance(val, int)


class TestContourMeasurementIntegration(unittest.TestCase):
    """
    Integration test for the complete contour measurement pipeline.
    Tests that measure_features correctly uses border_distances with the new format.
    """

    @classmethod
    def setUpClass(cls):
        """Create test data for all tests."""
        # Create a larger image with clear intensity gradient
        cls.frame = np.ones((100, 100, 1), dtype=float)
        # Create a gradient - center is brighter
        for i in range(100):
            for j in range(100):
                dist_from_center = np.sqrt((i - 50) ** 2 + (j - 50) ** 2)
                cls.frame[i, j, 0] = max(0, 1 - dist_from_center / 50)

        # Create a single centered object
        cls.labels = np.zeros((100, 100), dtype=int)
        cls.labels[35:65, 35:65] = 1  # 30x30 square centered at (50, 50)

    def test_measure_with_list_border_distance(self):
        """Test that measure_features works with list [min, max] border distance."""
        from celldetective.measure import measure_features

        result = measure_features(
            self.frame,
            self.labels,
            features=["intensity_mean"],
            channels=["test"],
            border_dist=[[-5, 5]],  # Edge region from -5 to +5 around boundary
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        # Check that edge intensity column exists
        edge_cols = [c for c in result.columns if "edge" in c or "slice" in c]
        self.assertGreater(len(edge_cols), 0, "Should have edge measurement columns")

    def test_measure_with_positive_only_border(self):
        """Test inner contour measurement [0, 5]."""
        from celldetective.measure import measure_features

        result = measure_features(
            self.frame,
            self.labels,
            features=["intensity_mean"],
            channels=["test"],
            border_dist=[[0, 5]],  # Inner edge only
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

    def test_measure_with_negative_only_border(self):
        """Test outer contour measurement [-5, 0]."""
        from celldetective.measure import measure_features

        result = measure_features(
            self.frame,
            self.labels,
            features=["intensity_mean"],
            channels=["test"],
            border_dist=[[-5, 0]],  # Outer edge only
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

    def test_measure_with_scalar_border_distance(self):
        """Test that scalar border_dist still works (backwards compatibility)."""
        from celldetective.measure import measure_features

        result = measure_features(
            self.frame,
            self.labels,
            features=["intensity_mean"],
            channels=["test"],
            border_dist=[5],  # Scalar - should be interpreted as [0, 5]
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)


class TestContourWithEdgeCases(unittest.TestCase):
    """Test edge cases and potential issues with contour computation."""

    def test_small_object_with_large_contour(self):
        """Test that small objects with large contour distance don't crash."""
        from celldetective.utils.masks import contour_of_instance_segmentation

        # Very small object
        label = np.zeros((50, 50), dtype=int)
        label[24:26, 24:26] = 1  # 2x2 pixel object

        # Large contour request - may result in empty contour
        result = contour_of_instance_segmentation(label, [0, 10])

        # Should not crash, may be empty or small
        self.assertEqual(result.shape, label.shape)

    def test_empty_label_returns_zeros(self):
        """Test that empty labels return zero array."""
        from celldetective.utils.masks import contour_of_instance_segmentation

        label = np.zeros((50, 50), dtype=int)

        result = contour_of_instance_segmentation(label, [0, 5])

        self.assertTrue(np.all(result == 0), "Empty label should return zeros")

    def test_multiple_objects(self):
        """Test contour with multiple distinct objects."""
        from celldetective.utils.masks import contour_of_instance_segmentation

        label = np.zeros((100, 100), dtype=int)
        label[10:30, 10:30] = 1
        label[60:80, 60:80] = 2

        result = contour_of_instance_segmentation(label, [0, 3])

        # Should have both object IDs in result
        unique_ids = np.unique(result[result > 0])
        self.assertIn(1, unique_ids, "Object 1 should be in result")
        self.assertIn(2, unique_ids, "Object 2 should be in result")


class TestSuffixFormatting(unittest.TestCase):
    """Test that measurement column names are formatted correctly."""

    def test_get_suffix_function(self):
        """Test the get_suffix helper function logic."""

        # Simulate the get_suffix function from measure.py
        def get_suffix(d):
            d_str = str(d)
            d_clean = (
                d_str.replace("(", "")
                .replace(")", "")
                .replace(", ", "_")
                .replace(",", "_")
            )
            if "-" in d_str or "," in d_str:
                return f"_slice_{d_clean.replace('-', 'm')}px"
            else:
                return f"_edge_{d_clean}px"

        # Test cases
        self.assertEqual(get_suffix(5), "_edge_5px")
        self.assertEqual(get_suffix(-5), "_slice_m5px")
        self.assertEqual(get_suffix([0, 5]), "_slice_[0_5]px")
        self.assertEqual(get_suffix([-5, 5]), "_slice_[m5_5]px")


if __name__ == "__main__":
    unittest.main()
