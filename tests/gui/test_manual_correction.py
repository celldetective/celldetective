"""
Unit tests for the napari correction viewers.

The viewers themselves need a display and a position on disk, so what is tested
here is what they are built from: the track propagation, the export of the
corrected tracks, the annotation crop bounds, the warning before masks are saved
over a tracked position, and the guard against closing with unsaved edits.
"""

import gc
import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PyQt5.QtWidgets import QMessageBox, QWidget

from celldetective.napari import frame_segmentation as fs
from celldetective.napari import utils as nu


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable all logging to avoid Windows OSError with pytest capture."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _two_cells(n_frames=4):
    """Two cells, tracks 1 and 2, on every frame, shown under their track IDs."""
    labels = np.zeros((n_frames, 20, 20), dtype=np.int32)
    rows = []
    for t in range(n_frames):
        labels[t, 1:4, 1:4] = 1
        labels[t, 10:14, 10:14] = 2
        rows.append({"TRACK_ID": 1, "FRAME": t, "class_id": 1.0})
        rows.append({"TRACK_ID": 2, "FRAME": t, "class_id": 2.0})
    return labels, pd.DataFrame(rows)


def _track_of(df, frame, class_id):
    row = df[(df["FRAME"] == frame) & (df["class_id"] == class_id)]
    return int(row["TRACK_ID"].iloc[0])


class TestPropagateIdentity:

    def test_the_cell_clicked_takes_the_identity_from_the_frame_on(self):
        labels, df = _two_cells()
        out = nu.propagate_identity(labels, df, 2, value_under=2, target_track_id=1)
        # Before the frame, nothing changes.
        assert _track_of(out, 1, 2.0) == 2
        assert _track_of(out, 1, 1.0) == 1
        # From the frame on, the clicked cell is track 1 ...
        assert _track_of(out, 2, 2.0) == 1
        assert _track_of(out, 3, 2.0) == 1
        assert np.all(labels[3, 10:14, 10:14] == 1)
        # ... and the old track 1 continues under a new ID.
        new_id = _track_of(out, 2, 1.0)
        assert new_id not in (0, 1, 2)
        assert np.all(labels[3, 1:4, 1:4] == new_id)
        assert np.all(labels[1, 10:14, 10:14] == 2)

    def test_a_swap_is_fixed_in_two_double_clicks(self):
        labels, df = _two_cells()
        df = nu.propagate_identity(labels, df, 2, value_under=2, target_track_id=1)
        new_id = _track_of(df, 2, 1.0)
        df = nu.propagate_identity(labels, df, 2, value_under=new_id, target_track_id=2)
        for t in (2, 3):
            assert _track_of(df, t, 2.0) == 1
            assert _track_of(df, t, 1.0) == 2
        assert set(df["TRACK_ID"]) == {1, 2}

    def test_label_zero_removes_the_cell_from_the_frame_on(self):
        labels, df = _two_cells()
        out = nu.propagate_identity(labels, df, 2, value_under=2, target_track_id=0)
        assert set(out.loc[out["TRACK_ID"] == 2, "FRAME"]) == {0, 1}
        assert np.all(labels[2:, 10:14, 10:14] == 0)
        assert np.all(labels[:2, 10:14, 10:14] == 2)

    def test_the_same_track_is_split(self):
        labels, df = _two_cells()
        out = nu.propagate_identity(labels, df, 2, value_under=1, target_track_id=1)
        assert set(out.loc[out["TRACK_ID"] == 1, "FRAME"]) == {0, 1}
        new_id = _track_of(out, 2, 1.0)
        assert new_id not in (0, 1, 2)
        assert set(out.loc[out["TRACK_ID"] == new_id, "FRAME"]) == {2, 3}

    def test_the_new_id_is_past_the_table_as_well_as_the_masks(self):
        labels, df = _two_cells()
        # A track with rows but no mask on screen.
        df = pd.concat(
            [df, pd.DataFrame({"TRACK_ID": [40], "FRAME": [0], "class_id": [np.nan]})],
            ignore_index=True,
        )
        out = nu.propagate_identity(labels, df, 2, value_under=2, target_track_id=1)
        assert _track_of(out, 2, 1.0) == 41

    def test_the_input_table_is_left_alone(self):
        labels, df = _two_cells()
        before = df.copy()
        nu.propagate_identity(labels, df, 2, value_under=2, target_track_id=1)
        pd.testing.assert_frame_equal(df, before)


class TestPropagationQuestion:

    def test_a_propagation_says_what_happens_to_both_tracks(self):
        text = nu.propagation_question(3, 7, 5)
        assert "track 3" in text and "track 7" in text and "frame 5" in text
        assert "new track" in text

    def test_label_zero_says_the_cell_is_removed(self):
        assert "Remove" in nu.propagation_question(0, 7, 5)

    def test_the_same_track_says_it_is_split(self):
        assert "Split" in nu.propagation_question(7, 7, 5)


class TestDetectionRows:

    def test_rows_carry_the_saved_label_and_the_centroid(self):
        original = np.zeros((20, 20), dtype=np.int32)
        original[2:5, 6:9] = 12
        shown = np.zeros_like(original)
        shown[2:5, 6:9] = 30
        rows = nu._detection_rows(original, shown, 30, 3)
        assert len(rows) == 1
        row = rows.iloc[0]
        assert row["class_id"] == 12
        assert row["TRACK_ID"] == 30
        assert row["FRAME"] == 3
        assert row["POSITION_X"] == pytest.approx(7.0)
        assert row["POSITION_Y"] == pytest.approx(3.0)

    def test_the_saved_frame_is_not_modified(self):
        original = np.full((5, 5), 4, dtype=np.int32)
        shown = np.zeros_like(original)
        nu._detection_rows(original, shown, 1, 0)
        assert np.all(original == 4)


class TestPrepareTrackExport:

    @staticmethod
    def _table():
        rows = []
        for t in range(5):
            rows.append(
                {"TRACK_ID": 1, "FRAME": t, "POSITION_X": 50.0 + t,
                 "POSITION_Y": 50.0, "class_id": 1.0, "Unnamed: 0": t}
            )
            # A ghost track: rows but never a mask.
            rows.append(
                {"TRACK_ID": 2, "FRAME": t, "POSITION_X": 80.0,
                 "POSITION_Y": 80.0, "class_id": np.nan, "Unnamed: 0": t}
            )
        return pd.DataFrame(rows)

    def test_ghost_tracks_and_index_leftovers_are_dropped(self):
        out = nu.prepare_track_export(self._table(), (200, 200), {})
        assert set(out["TRACK_ID"]) == {1}
        assert not [c for c in out.columns if c.startswith("Unnamed")]

    def test_first_detection_and_velocity_are_recomputed(self):
        out = nu.prepare_track_export(self._table(), (200, 200), {})
        assert "class_firstdetection" in out.columns
        assert "t_firstdetection" in out.columns
        assert any("velocity" in c for c in out.columns)

    def test_the_post_processing_is_applied(self):
        out = nu.prepare_track_export(
            self._table(), (200, 200), {"minimum_tracklength": 10}
        )
        assert out.empty or 1 not in set(out["TRACK_ID"])


class TestEventColumns:

    def test_event_columns_are_listed_but_not_the_recomputed_ones(self):
        df = pd.DataFrame(
            columns=[
                "TRACK_ID", "class_id", "class_firstdetection", "t_firstdetection",
                "class_death", "t_death", "status_death", "area",
            ]
        )
        assert nu._event_columns(df) == ["class_death", "t_death", "status_death"]

    def test_the_colour_column_is_not_an_event(self):
        df = pd.DataFrame(columns=["class_color", "class_death"])
        assert nu._event_columns(df) == ["class_death"]


class TestRoiBounds:

    @staticmethod
    def _rect(r0, c0, r1, c1, t=0):
        return np.array([[t, r0, c0], [t, r0, c1], [t, r1, c1], [t, r1, c0]], float)

    def test_a_rectangle_inside_the_frame(self):
        assert nu._roi_bounds(self._rect(10, 20, 50, 70), (100, 100)) == (10, 50, 20, 70)

    def test_the_vertex_order_does_not_matter(self):
        rect = self._rect(10, 20, 50, 70)[::-1]
        assert nu._roi_bounds(rect, (100, 100)) == (10, 50, 20, 70)

    def test_a_rectangle_past_the_top_left_is_clipped(self):
        # Sliced with its raw corners, this crop came out empty.
        assert nu._roi_bounds(self._rect(-5, -8, 40, 30), (100, 100)) == (0, 40, 0, 30)

    def test_a_rectangle_past_the_bottom_right_is_clipped(self):
        assert nu._roi_bounds(self._rect(60, 70, 130, 120), (100, 100)) == (60, 100, 70, 100)

    def test_a_rectangle_outside_the_frame_gives_nothing(self):
        assert nu._roi_bounds(self._rect(-50, -50, -10, -10), (100, 100)) is None

    def test_a_rotated_rectangle_gives_its_bounding_box(self):
        diamond = np.array([[0, 10, 50], [0, 50, 90], [0, 90, 50], [0, 50, 10]], float)
        assert nu._roi_bounds(diamond, (100, 100)) == (10, 90, 10, 90)


class TestStaleTables:

    def test_a_tracked_position_is_detected(self, tmp_path):
        tables = tmp_path / "output" / "tables"
        tables.mkdir(parents=True)
        (tables / "trajectories_targets.csv").write_text("TRACK_ID\n1\n")
        assert nu._has_trajectory_table(str(tmp_path), "targets")
        assert nu._has_trajectory_table(str(tmp_path) + os.sep, "targets")
        assert not nu._has_trajectory_table(str(tmp_path), "effectors")

    def test_an_untracked_position_needs_no_warning(self, tmp_path):
        assert not nu._has_trajectory_table(str(tmp_path), "targets")

    def test_the_question_names_the_population_the_frames_and_the_remedy(self):
        text = nu.stale_tables_question("effectors", [3, 7])
        assert "effectors" in text
        assert "2 edited frame(s) (3, 7)" in text
        assert "tracking" in text


class TestFramesToSave:

    @staticmethod
    def _folder(tmp_path, labels, frames):
        from tifffile import imwrite

        for t in frames:
            imwrite(str(tmp_path / f"{str(t).zfill(4)}.tif"), labels[t])
        return str(tmp_path) + os.sep

    def test_only_the_edited_frames_are_written(self, tmp_path):
        labels = np.zeros((4, 8, 8), dtype=np.int16)
        folder = self._folder(tmp_path, labels, range(4))
        labels[2, 1, 1] = 5
        assert nu.frames_to_save(labels, {2}, folder) == [2]

    def test_an_edit_undone_is_not_written(self, tmp_path):
        labels = np.zeros((3, 8, 8), dtype=np.int16)
        folder = self._folder(tmp_path, labels, range(3))
        # Flagged, but back to what is on disk.
        assert nu.frames_to_save(labels, {1}, folder) == []

    def test_a_frame_without_a_file_is_always_written(self, tmp_path):
        labels = np.zeros((3, 8, 8), dtype=np.int16)
        folder = self._folder(tmp_path, labels, [0, 2])
        assert nu.frames_to_save(labels, set(), folder) == [1]


class TestAnnotationCrops:

    @staticmethod
    def _layer(shapes, kinds=None):
        data = [np.asarray(s, dtype=float) for s in shapes]
        return SimpleNamespace(data=data, shape_type=kinds or ["rectangle"] * len(data))

    @staticmethod
    def _rect(t, r0, c0, r1, c1):
        return [[t, r0, c0], [t, r0, c1], [t, r1, c1], [t, r1, c0]]

    def test_a_rectangle_on_the_frame_is_a_crop(self):
        layer = self._layer([self._rect(2, 10, 20, 50, 70)])
        assert nu.annotation_crops([layer], 2, (100, 100)) == (1, [(10, 50, 20, 70)])

    def test_regions_carry_on_from_an_earlier_frame(self):
        layer = self._layer([self._rect(2, 10, 20, 50, 70)])
        assert nu.annotation_crops([layer], 5, (100, 100))[1] == [(10, 50, 20, 70)]
        assert nu.annotation_crops([layer], 1, (100, 100)) == (0, [])

    def test_any_shapes_layer_counts_including_the_threshold_rois(self):
        shapes = self._layer([self._rect(0, 0, 0, 10, 10)])
        rois = self._layer([self._rect(0, 20, 20, 40, 40)])
        rois.name = "ROIs"
        assert nu.annotation_crops([shapes, rois], 0, (100, 100))[1] == [
            (0, 10, 0, 10),
            (20, 40, 20, 40),
        ]

    def test_a_polygon_gives_its_bounding_box_and_a_line_nothing(self):
        polygon = [[0, 10, 10], [0, 30, 50], [0, 60, 20], [0, 40, 5], [0, 15, 2]]
        line = [[0, 0, 0], [0, 90, 90]]
        layer = self._layer([polygon, line], kinds=["polygon", "line"])
        assert nu.annotation_crops([layer], 0, (100, 100)) == (1, [(10, 60, 2, 50)])

    def test_a_2d_region_applies_to_every_frame(self):
        layer = self._layer([[[10, 20], [10, 70], [50, 70], [50, 20]]])
        assert nu.annotation_crops([layer], 9, (100, 100))[1] == [(10, 50, 20, 70)]

    def test_a_region_outside_the_frame_is_counted_but_not_cropped(self):
        layer = self._layer([self._rect(0, -50, -50, -10, -10)])
        assert nu.annotation_crops([layer], 0, (100, 100)) == (1, [])


class TestUnsavedFlag:

    @staticmethod
    def _layer():
        from napari.layers import Labels

        return Labels(np.zeros((2, 10, 10), dtype=np.int32))

    def test_a_new_layer_has_no_unsaved_edits(self):
        assert not fs.has_unsaved(self._layer())

    def test_painting_marks_the_frame_painted_on(self):
        layer = self._layer()
        fs.track_unsaved_edits(layer)
        layer.paint((1, 5, 5), 3)
        assert fs.unsaved_frames(layer) == {1}
        fs.clear_unsaved(layer)
        assert not fs.has_unsaved(layer)

    def test_clearing_one_frame_keeps_the_others(self):
        layer = self._layer()
        fs.mark_unsaved(layer, [0, 1])
        fs.clear_unsaved(layer, [0])
        assert fs.unsaved_frames(layer) == {1}

    def test_a_refresh_does_not_mark_the_layer(self):
        layer = self._layer()
        fs.track_unsaved_edits(layer)
        layer.refresh()
        assert not fs.has_unsaved(layer)

    def test_a_bulk_write_is_marked_explicitly(self):
        layer = self._layer()
        fs.mark_unsaved(layer, [1])
        assert fs.has_unsaved(layer)

    def test_recording_an_undo_step_marks_its_frames(self):
        layer = self._layer()
        before = layer.data[1].copy()
        after = before.copy()
        after[2, 2] = 7
        fs.record_undo(layer, [fs.history_atom(1, before, after)])
        assert fs.unsaved_frames(layer) == {1}

    def test_an_empty_undo_step_does_not(self):
        layer = self._layer()
        fs.record_undo(layer, [None])
        assert not fs.has_unsaved(layer)


class TestGuardUnsavedClose:

    @staticmethod
    def _viewer(qtbot):
        window = QWidget()
        qtbot.addWidget(window)
        window.show()
        return SimpleNamespace(window=SimpleNamespace(_qt_window=window)), window

    @staticmethod
    def _layer():
        from napari.layers import Labels

        return Labels(np.zeros((2, 10, 10), dtype=np.int32))

    def test_a_clean_viewer_closes_without_asking(self, qtbot, monkeypatch):
        asked = []
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: asked.append(a) or QMessageBox.No
        )
        viewer, window = self._viewer(qtbot)
        layer = self._layer()
        nu.guard_unsaved_close(viewer, layer, "lose them?")
        assert window.close()
        assert not asked

    def test_declining_keeps_the_viewer_open(self, qtbot, monkeypatch):
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
        viewer, window = self._viewer(qtbot)
        layer = self._layer()
        nu.guard_unsaved_close(viewer, layer, "lose them?")
        layer.paint((0, 5, 5), 3)
        assert not window.close()
        assert window.isVisible()

    def test_accepting_closes_the_viewer(self, qtbot, monkeypatch):
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
        viewer, window = self._viewer(qtbot)
        layer = self._layer()
        nu.guard_unsaved_close(viewer, layer, "lose them?")
        fs.mark_unsaved(layer, [0])
        assert window.close()

    def test_the_guard_does_not_keep_the_layer_alive(self, qtbot, monkeypatch):
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
        viewer, window = self._viewer(qtbot)
        layer = self._layer()
        nu.guard_unsaved_close(viewer, layer, "lose them?")
        fs.mark_unsaved(layer, [0])
        del layer
        gc.collect()
        assert window.close()

    def test_an_unreachable_window_is_not_an_error(self):
        assert nu.guard_unsaved_close(SimpleNamespace(), self._layer(), "x") is None
