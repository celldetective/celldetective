"""
Unit tests for the segmentation robustness work:

- shared per-frame DL inference core (`_run_dl_model_on_frame`)
- fail-loud + atomic label writes (`finalize_position`)
- GPU device resolution
- vectorized mask post-processing (`auto_correct_masks`, `filter_on_property`)

These intentionally avoid loading real StarDist/Cellpose/TF models: the model is
mocked, so the tests are fast and run without a GPU.
"""

import os
import unittest
from unittest.mock import MagicMock

import numpy as np
from tifffile import imwrite

from celldetective.segmentation import (
    _run_dl_model_on_frame,
    filter_on_property,
)
from celldetective.utils.mask_cleaning import auto_correct_masks
from celldetective.utils.resources import resolve_gpu_device
from celldetective.processes.segment_cells import BaseSegmentProcess


class TestResolveGpuDevice(unittest.TestCase):

    def test_default(self):
        os.environ.pop("CELLDETECTIVE_GPU_DEVICE", None)
        self.assertEqual(resolve_gpu_device(), "0")
        self.assertEqual(resolve_gpu_device(default="2"), "2")

    def test_env_override(self):
        os.environ["CELLDETECTIVE_GPU_DEVICE"] = "1"
        try:
            self.assertEqual(resolve_gpu_device(), "1")
        finally:
            del os.environ["CELLDETECTIVE_GPU_DEVICE"]

    def test_blank_env_falls_back(self):
        os.environ["CELLDETECTIVE_GPU_DEVICE"] = "   "
        try:
            self.assertEqual(resolve_gpu_device(), "0")
        finally:
            del os.environ["CELLDETECTIVE_GPU_DEVICE"]


class TestRunDLModelOnFrame(unittest.TestCase):

    def _stardist_model(self, labels):
        model = MagicMock()
        model.predict_instances.return_value = (labels, {})
        return model

    def test_stardist_dispatch_no_rescale(self):
        labels = np.zeros((8, 8), dtype=np.uint16)
        labels[2:5, 2:5] = 1
        model = self._stardist_model(labels)
        frame = np.zeros((8, 8, 1), dtype=float)

        out = _run_dl_model_on_frame(frame, model, "stardist", scale_model=None)
        self.assertEqual(out.shape, (8, 8))
        np.testing.assert_array_equal(out, labels)

    def test_rescale_applied_when_scale_model_set(self):
        # Input frame is on a 4x4 grid; scale_model=2 means it was zoomed 2x, so
        # an 8x8 prediction must be rescaled back to 4x4.
        labels = np.ones((8, 8), dtype=np.uint16)
        model = self._stardist_model(labels)
        frame = np.zeros((8, 8, 1), dtype=float)

        out = _run_dl_model_on_frame(frame, model, "stardist", scale_model=2.0)
        self.assertEqual(out.shape, (4, 4))

    def test_cellpose_dispatch(self):
        labels = np.zeros((6, 6), dtype=np.uint16)
        labels[1:3, 1:3] = 7
        model = MagicMock()
        model.eval.return_value = (labels, None, None)
        frame = np.zeros((6, 6, 2), dtype=float)

        out = _run_dl_model_on_frame(
            frame, model, "cellpose", scale_model=None, diameter=30
        )
        np.testing.assert_array_equal(out, labels)
        self.assertTrue(model.eval.called)

    def test_unknown_model_type_raises(self):
        with self.assertRaises(ValueError):
            _run_dl_model_on_frame(np.zeros((4, 4, 1)), MagicMock(), "bogus")


class TestFinalizePosition(unittest.TestCase):
    """Verify completeness gating + backup handling (direct-write model)."""

    def _make_proc(self, tmp_path, n_frames):
        proc = BaseSegmentProcess.__new__(BaseSegmentProcess)
        proc.pos = str(tmp_path) + os.sep
        proc.label_folder = "labels_targets"
        proc.backup_label_folder = "labels_targets.bak"
        proc.len_movie = n_frames
        return proc

    def _write_masks(self, folder, n):
        os.makedirs(folder, exist_ok=True)
        for t in range(n):
            imwrite(
                os.path.join(folder, f"{str(t).zfill(4)}.tif"),
                np.zeros((4, 4), dtype=np.uint16),
            )

    def _count_tif(self, folder):
        if not os.path.isdir(folder):
            return 0
        return len([f for f in os.listdir(folder) if f.endswith(".tif")])

    def test_complete_drops_backup(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            proc = self._make_proc(d, n_frames=3)
            final = proc.pos + proc.label_folder
            backup = proc.pos + proc.backup_label_folder
            # New masks live directly in labels_targets; backup holds the old ones.
            self._write_masks(final, 3)
            self._write_masks(backup, 2)

            proc.finalize_position()

            self.assertEqual(self._count_tif(final), 3)
            self.assertFalse(os.path.isdir(backup), "backup should be dropped")

    def test_incomplete_keeps_partial_visible_and_backup(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            proc = self._make_proc(d, n_frames=5)
            final = proc.pos + proc.label_folder
            backup = proc.pos + proc.backup_label_folder
            # 2 of 5 new frames produced before the abort; old masks in backup.
            self._write_masks(final, 2)
            self._write_masks(backup, 5)

            with self.assertRaises(RuntimeError):
                proc.finalize_position()

            # Partial new masks remain inspectable in the real folder...
            self.assertEqual(self._count_tif(final), 2)
            # ...and the previous masks are still recoverable.
            self.assertEqual(self._count_tif(backup), 5)


class TestAutoCorrectMasks(unittest.TestCase):

    def test_no_in_place_mutation_and_compaction(self):
        m = np.array(
            [[0, 0, 1, 1], [0, 2, 2, 1], [0, 2, 0, 9]], dtype=np.int32
        )
        orig = m.copy()
        out = auto_correct_masks(m, min_area=1)
        self.assertTrue(np.array_equal(m, orig), "input must not be mutated")
        # Labels compacted to a contiguous 1..N range.
        self.assertEqual(sorted(np.unique(out).tolist()), [0, 1, 2, 3])

    def test_small_objects_removed(self):
        m = np.zeros((20, 20), dtype=np.int32)
        m[2:12, 2:12] = 1  # large
        m[15, 15] = 2  # 1 px, below default min_area
        out = auto_correct_masks(m)
        self.assertEqual(sorted(np.unique(out).tolist()), [0, 1])

    def test_empty_mask(self):
        out = auto_correct_masks(np.zeros((10, 10), dtype=np.int32))
        self.assertEqual(np.unique(out).tolist(), [0])


class TestFilterOnProperty(unittest.TestCase):

    def _three_cells(self):
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[2:4, 2:4] = 1  # area 4
        labels[5:12, 5:12] = 2  # area 49
        labels[15, 15] = 3  # area 1
        return labels

    def test_query_filters_cells(self):
        out = filter_on_property(self._three_cells(), queries=["area < 10"])
        self.assertEqual(sorted(np.unique(out).tolist()), [0, 2])

    def test_none_query_passthrough(self):
        labels = self._three_cells()
        out = filter_on_property(labels.copy(), queries=None)
        np.testing.assert_array_equal(out, labels)


if __name__ == "__main__":
    unittest.main()
