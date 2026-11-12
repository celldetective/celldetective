"""
Unit tests for the mask clean-up helpers behind the napari correction viewers.

`auto_correct_masks` runs on every frame when corrected masks are saved, and
`relabel_segmentation` decides which track a mask is drawn as in the track
viewer, so a mistake in either changes what is written to disk or which cell a
correction lands on.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from celldetective.utils.mask_cleaning import auto_correct_masks, relabel_segmentation


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _square(frame, top, left, size, value):
    frame[top : top + size, left : left + size] = value


class TestAutoCorrectMasks:

    def test_labels_are_renumbered_from_one(self):
        masks = np.zeros((30, 30), dtype=int)
        _square(masks, 0, 0, 4, 3)
        _square(masks, 10, 10, 4, 7)
        out = auto_correct_masks(masks)
        assert set(np.unique(out)) == {0, 1, 2}
        assert np.all(out[masks == 3] == 1)
        assert np.all(out[masks == 7] == 2)

    def test_small_objects_are_removed(self):
        masks = np.zeros((30, 30), dtype=int)
        _square(masks, 0, 0, 2, 1)  # 4 px
        _square(masks, 10, 10, 4, 2)  # 16 px
        out = auto_correct_masks(masks, min_area=9)
        assert np.all(out[masks == 1] == 0)
        assert np.all(out[masks == 2] == 1)

    def test_a_min_area_of_zero_keeps_everything_and_still_renumbers(self):
        masks = np.zeros((30, 30), dtype=int)
        _square(masks, 0, 0, 1, 5)
        _square(masks, 10, 10, 4, 9)
        out = auto_correct_masks(masks, min_area=0)
        assert set(np.unique(out)) == {0, 1, 2}

    def test_two_separate_objects_sharing_a_label_are_split(self):
        masks = np.zeros((40, 40), dtype=int)
        _square(masks, 0, 0, 5, 1)
        _square(masks, 30, 30, 5, 1)
        out = auto_correct_masks(masks)
        assert len(np.unique(out[masks == 1])) == 2
        assert 0 not in np.unique(out[masks == 1])

    def test_holes_are_filled_only_when_asked(self):
        masks = np.zeros((20, 20), dtype=int)
        _square(masks, 2, 2, 9, 1)
        masks[6, 6] = 0
        assert auto_correct_masks(masks, fill_labels=False)[6, 6] == 0
        assert auto_correct_masks(masks, fill_labels=True)[6, 6] == 1

    def test_negative_labels_are_made_positive(self):
        masks = np.zeros((20, 20), dtype=int)
        _square(masks, 2, 2, 4, -2)
        out = auto_correct_masks(masks)
        assert np.all(out[masks == -2] == 1)

    def test_the_input_is_not_modified(self):
        masks = np.zeros((20, 20), dtype=int)
        _square(masks, 2, 2, 4, 5)
        before = masks.copy()
        auto_correct_masks(masks)
        np.testing.assert_array_equal(masks, before)

    def test_only_2d_masks_are_accepted(self):
        with pytest.raises(ValueError):
            auto_correct_masks(np.zeros((2, 5, 5), dtype=int))

    def test_an_unedited_diagonal_cell_changes_label(self):
        # Documents why saving masks invalidates the tables: a single connected
        # cell whose bounding box is much larger than itself is relabelled, so
        # its ID moves even though nobody edited it.
        masks = np.zeros((40, 40), dtype=int)
        _square(masks, 2, 2, 6, 1)
        for i in range(20):
            masks[15 + i, 10 + i : 13 + i] = 2
        _square(masks, 30, 30, 6, 3)
        out = auto_correct_masks(masks)
        assert len(np.unique(out[masks == 2])) == 1
        assert np.unique(out[masks == 2])[0] != 2

    def test_keep_labels_leaves_an_unedited_diagonal_cell_alone(self):
        masks = np.zeros((40, 40), dtype=int)
        _square(masks, 2, 2, 6, 1)
        for i in range(20):
            masks[15 + i, 10 + i : 13 + i] = 2
        _square(masks, 30, 30, 6, 3)
        out = auto_correct_masks(masks, keep_labels=True)
        np.testing.assert_array_equal(out, masks)

    def test_keep_labels_splits_off_the_smaller_pieces_only(self):
        masks = np.zeros((30, 30), dtype=int)
        _square(masks, 1, 1, 6, 4)  # largest piece of label 4
        _square(masks, 20, 20, 4, 4)  # a second, separate piece
        _square(masks, 10, 10, 5, 7)
        out = auto_correct_masks(masks, keep_labels=True)
        assert (out[1:7, 1:7] == 4).all()
        assert (out[10:15, 10:15] == 7).all()
        split = np.unique(out[20:24, 20:24])
        assert len(split) == 1 and split[0] > 7

    def test_keep_labels_drops_small_objects_without_renumbering(self):
        masks = np.zeros((20, 20), dtype=int)
        _square(masks, 1, 1, 2, 3)  # 4 px, dropped
        _square(masks, 10, 10, 4, 9)
        out = auto_correct_masks(masks, keep_labels=True)
        assert set(np.unique(out)) == {0, 9}


class TestRelabelSegmentation:

    @staticmethod
    def _labels():
        labels = np.zeros((2, 20, 20), dtype=int)
        _square(labels[0], 1, 1, 3, 1)
        _square(labels[0], 10, 10, 4, 2)
        _square(labels[0], 5, 14, 3, 3)  # not in the table
        _square(labels[1], 1, 1, 3, 1)
        _square(labels[1], 12, 2, 3, 4)  # not in the table
        return labels

    @staticmethod
    def _table():
        return pd.DataFrame(
            {
                "TRACK_ID": [4.0, 9.0, 4.0],
                "FRAME": [0, 0, 1],
                "class_id": [1.0, 2.0, 1.0],
                "POSITION_X": [2.0, 12.0, 2.0],
                "POSITION_Y": [2.0, 12.0, 2.0],
            }
        )

    def test_tracked_masks_take_their_track_id(self):
        labels = self._labels()
        out = relabel_segmentation(labels, self._table())
        assert np.all(out[0][labels[0] == 1] == 4)
        assert np.all(out[0][labels[0] == 2] == 9)
        assert np.all(out[1][labels[1] == 1] == 4)

    def test_an_untracked_mask_never_takes_an_existing_track_id(self):
        labels = self._labels()
        df = self._table()
        out = relabel_segmentation(labels, df)
        untracked = {
            int(np.unique(out[0][labels[0] == 3])[0]),
            int(np.unique(out[1][labels[1] == 4])[0]),
        }
        assert untracked.isdisjoint(set(df["TRACK_ID"].astype(int)))
        assert all(i > df["TRACK_ID"].max() for i in untracked)
        assert len(untracked) == 2

    def test_a_track_without_any_mask_still_reserves_its_id(self):
        labels = self._labels()
        df = pd.concat(
            [
                self._table(),
                pd.DataFrame(
                    {"TRACK_ID": [12.0], "FRAME": [0], "class_id": [np.nan]}
                ),
            ],
            ignore_index=True,
        )
        out = relabel_segmentation(labels, df, exclude_nans=True)
        assert int(np.unique(out[0][labels[0] == 3])[0]) > 12

    def test_untracked_ids_stay_distinct_across_threads(self):
        labels = np.zeros((6, 20, 20), dtype=int)
        rows = []
        for t in range(6):
            _square(labels[t], 1, 1, 3, 1)
            _square(labels[t], 10, 10, 3, 2)  # untracked on every frame
            rows.append({"TRACK_ID": 1.0, "FRAME": t, "class_id": 1.0})
        out = relabel_segmentation(labels, pd.DataFrame(rows), threads=3)
        ids = [int(np.unique(out[t][labels[t] == 2])[0]) for t in range(6)]
        assert len(set(ids)) == 6
        assert 1 not in ids

    def test_a_cancelled_relabelling_returns_none(self):
        out = relabel_segmentation(
            self._labels(), self._table(), progress_callback=lambda p: False
        )
        assert out is None
