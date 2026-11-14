"""
Merging the masks of several threshold pipelines
================================================

``merge_instance_segmentation`` pairs the objects of each mask by IoU and
merges them. It used StarDist for the pairing, and without StarDist kept the
first mask only (SEG-1 on the wiki page ``Feature-Segmentation``). These tests
block StarDist to check the merge does not depend on it.
"""

import sys
import unittest
from unittest import mock

import numpy as np

from celldetective.segmentation import _match_instances, merge_instance_segmentation


def _square(shape, y, x, size, value):
    mask = np.zeros(shape, dtype=np.int32)
    mask[y : y + size, x : x + size] = value
    return mask


class TestMergeWithoutStarDist(unittest.TestCase):
    """SEG-1: merging several pipelines needs StarDist."""

    def setUp(self):
        blocked = {"stardist": None, "stardist.matching": None}
        patcher = mock.patch.dict(sys.modules, blocked)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_objects_of_every_pipeline_are_kept(self):
        shape = (40, 40)
        first = _square(shape, 2, 2, 8, 1)
        # One object overlapping the first one, one found by this pipeline only.
        second = _square(shape, 4, 4, 8, 5) + _square(shape, 25, 25, 8, 9)
        merged = merge_instance_segmentation([first, second], mode="OR")

        self.assertEqual(len(np.unique(merged[merged > 0])), 2)
        self.assertTrue(np.all(merged[first > 0] > 0), "the first object is lost")
        self.assertTrue(np.all(merged[25:33, 25:33] > 0), "the second pipeline's own object is lost")
        # OR: the overlapping objects become one object covering both.
        union = (first > 0) | (_square(shape, 4, 4, 8, 1) > 0)
        self.assertEqual(len(np.unique(merged[union])), 1)

    def test_inputs_are_not_modified(self):
        shape = (20, 20)
        first = _square(shape, 2, 2, 6, 1)
        second = _square(shape, 3, 3, 6, 2)
        before = second.copy()
        merge_instance_segmentation([first, second])
        np.testing.assert_array_equal(second, before)

    def test_a_single_mask_keeps_its_objects(self):
        # It used to raise UnboundLocalError: nothing to merge it with.
        mask = _square((10, 10), 1, 1, 3, 4)
        merged = merge_instance_segmentation([mask])
        np.testing.assert_array_equal(merged > 0, mask > 0)


class TestMatchInstances(unittest.TestCase):
    def test_one_to_one_by_iou(self):
        shape = (30, 30)
        a = _square(shape, 0, 0, 10, 3) + _square(shape, 15, 15, 10, 8)
        b = _square(shape, 16, 16, 10, 2) + _square(shape, 1, 1, 10, 7)
        pairs = {(r, o) for r, o, _ in _match_instances(a, b)}
        self.assertEqual(pairs, {(3, 7), (8, 2)})

    def test_empty_mask(self):
        a = _square((10, 10), 1, 1, 3, 1)
        self.assertEqual(_match_instances(a, np.zeros_like(a)), [])


if __name__ == "__main__":
    unittest.main()
