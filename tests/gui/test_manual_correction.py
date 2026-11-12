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
from PyQt5.QtCore import QEvent
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


class TestWriteLabelFrames:

    def test_only_the_given_frames_are_written_and_cells_keep_their_numbers(self, tmp_path):
        from tifffile import imread

        labels = np.zeros((3, 30, 30), dtype=np.int32)
        labels[:, 2:8, 2:8] = 5
        labels[:, 15:20, 15:20] = 9
        labels[1, 25:27, 25:27] = 12  # 4 px: removed by the auto-fixes
        written = []
        n = nu.write_label_frames(
            labels,
            [1],
            str(tmp_path) + os.sep,
            {"correct_anomalies": True, "fill_labels": False, "min_area": 9},
            written.append,
        )
        assert sorted(os.listdir(tmp_path)) == ["0001.tif"]
        on_disk = imread(str(tmp_path / "0001.tif"))
        assert set(np.unique(on_disk)) == {0, 5, 9}
        # The viewer shows what was saved.
        np.testing.assert_array_equal(labels[1], on_disk)
        assert n == 2 and written == [1]

    def test_numbers_past_int16_are_renumbered(self):
        im = np.zeros((20, 20), dtype=np.int64)
        im[2:8, 2:8] = 40_000
        im[10:16, 10:16] = 3
        out = nu.correct_saved_frame(im, {"min_area": 0})
        assert set(np.unique(out)) == {0, 1, 2}


class TestWriteFrameAnnotation:

    @staticmethod
    def _frame():
        channels = [np.full((300, 400), 1.0), np.full((300, 400), 2.0)]
        labels = np.zeros((300, 400), dtype=int)
        labels[10:20, 10:20] = 1
        return channels, labels

    def test_the_whole_field_is_written_with_its_sidecar(self, tmp_path):
        import json
        from tifffile import imread

        channels, labels = self._frame()
        stems = nu.write_frame_annotation(
            str(tmp_path), "exp_pos_0003", channels, labels, None, {"frame": 3}
        )
        assert stems == ["exp_pos_0003"]
        assert imread(str(tmp_path / "exp_pos_0003.tif")).shape == (2, 300, 400)
        lab = imread(str(tmp_path / "exp_pos_0003_labelled.tif"))
        assert lab.dtype == np.int16 and lab.shape == (300, 400)
        assert json.loads((tmp_path / "exp_pos_0003.json").read_text()) == {"frame": 3}

    def test_a_small_crop_is_padded_to_exactly_256(self, tmp_path):
        from tifffile import imread

        channels, labels = self._frame()
        stems = nu.write_frame_annotation(
            str(tmp_path), "p", channels, labels, [(5, 106, 0, 255)], {}
        )
        assert stems == ["p_roi_5_106_0_255"]
        img = imread(str(tmp_path / "p_roi_5_106_0_255.tif"))
        lab = imread(str(tmp_path / "p_roi_5_106_0_255_labelled.tif"))
        assert img.shape == (2, 256, 256) and lab.shape == (256, 256)
        # The cell stays in the crop, shifted by the padding.
        assert (lab == 1).sum() == 100

    def test_a_large_crop_is_not_padded(self, tmp_path):
        from tifffile import imread

        channels, labels = self._frame()
        nu.write_frame_annotation(str(tmp_path), "p", channels, labels, [(0, 280, 0, 300)], {})
        assert imread(str(tmp_path / "p_roi_0_280_0_300.tif")).shape == (2, 280, 300)


class TestTrackCorrections:

    def test_undo_restores_the_masks_and_the_table(self):
        labels, df = _two_cells()
        masks, table = labels.copy(), df.copy()
        history = nu.TrackCorrections()
        out = history.apply(labels, df, 2, 2, 1)
        assert not np.array_equal(labels, masks)
        undone, frame = history.undo(labels)
        np.testing.assert_array_equal(labels, masks)
        pd.testing.assert_frame_equal(undone, table)
        assert frame == 2 and history.is_clean
        assert history.undo(labels) is None
        assert len(out) == len(df)

    def test_steps_are_undone_in_reverse_order(self):
        labels, df = _two_cells()
        masks = labels.copy()
        history = nu.TrackCorrections()
        df1 = history.apply(labels, df, 1, 2, 1)
        after_first = labels.copy()
        history.apply(labels, df1, 3, 1, 0)  # label 0: remove from frame 3
        assert history.undo(labels)[0] is df1
        np.testing.assert_array_equal(labels, after_first)
        history.undo(labels)
        np.testing.assert_array_equal(labels, masks)

    def test_label_0_does_not_record_the_background(self):
        labels, df = _two_cells()
        history = nu.TrackCorrections()
        history.apply(labels, df, 0, 2, 0)
        changed = history._steps[0][1]
        assert len(changed[0]) == (labels.shape[0]) * 16

    def test_forgotten_steps_keep_the_viewer_dirty(self):
        labels, df = _two_cells()
        history = nu.TrackCorrections(max_steps=1)
        df = history.apply(labels, df, 1, 2, 1)
        history.apply(labels, df, 2, 1, 2)
        history.undo(labels)
        assert len(history) == 0 and not history.is_clean
        history.clear()
        assert history.is_clean

    def test_added_detections_are_counted_in_the_exported_table(self):
        labels, df = _two_cells()
        labels[:, 16:19, 2:5] = 7  # an untracked mask, class_id 3 on disk
        history = nu.TrackCorrections()
        added = pd.DataFrame(
            {"POSITION_X": [3.0], "POSITION_Y": [17.0], "class_id": [3], "FRAME": [1], "TRACK_ID": [7]}
        )
        out = history.apply(labels, df, 1, 7, 1, [added])
        assert history.count_added(out) == 1
        # Removed again by a later correction: no longer in the table.
        out2 = history.apply(labels, out, 1, 1, 0)
        assert history.count_added(out2) == 0
        # Undoing the addition forgets it.
        history.undo(labels)
        history.undo(labels)
        assert history.added == set()


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

    def test_a_close_from_viewer_close_is_not_asked(self, qtbot, monkeypatch):
        # viewer.close() removes the layers before closing the window: declining
        # then would leave a viewer with nothing in it.
        asked = []
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: asked.append(a) or QMessageBox.No
        )
        viewer, window = self._viewer(qtbot)
        window._qt_viewer = SimpleNamespace(viewer=SimpleNamespace(layers=[]))
        layer = self._layer()
        nu.guard_unsaved_close(viewer, layer, "lose them?")
        fs.mark_unsaved(layer, [0])
        assert window.close()
        assert not asked

    def test_an_accepted_close_from_the_frame_is_not_asked_again_by_napari(
        self, qtbot, monkeypatch
    ):
        asked, timers = [], []
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: asked.append(a) or QMessageBox.Yes
        )
        monkeypatch.setattr(nu, "_napari_confirms_close", lambda: True)
        monkeypatch.setattr(nu.QTimer, "singleShot", lambda ms, fn: timers.append(fn))
        viewer, window = self._viewer(qtbot)
        layer = self._layer()
        guard = nu.guard_unsaved_close(viewer, layer, "lose them?")
        fs.mark_unsaved(layer, [0])
        ignored = []
        event = SimpleNamespace(
            type=lambda: QEvent.Close,
            spontaneous=lambda: True,
            ignore=lambda: ignored.append(True),
        )
        # The frame's close is replaced by one from code, which napari does not
        # ask about; that one is let through without asking again.
        assert guard.eventFilter(window, event)
        assert ignored and timers == [window.close] and len(asked) == 1
        assert window.close()
        assert len(asked) == 1

    def test_napari_menu_close_asks_once(self, qtbot, monkeypatch):
        asked = []
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, **k: asked.append(a) or QMessageBox.Yes
        )

        class NapariWindow(QWidget):
            _is_close_dialog = {False: True, True: True}

            def __init__(self):
                super().__init__()
                self.calls = []

            def close(self, quit_app=False, confirm_need=False):
                self.calls.append((quit_app, confirm_need))
                return super().close()

        window = NapariWindow()
        qtbot.addWidget(window)
        viewer = SimpleNamespace(window=SimpleNamespace(_qt_window=window))
        layer = self._layer()
        nu.guard_unsaved_close(viewer, layer, "lose them?")

        window.close(confirm_need=True)  # clean: napari's own question stands
        assert window.calls == [(False, True)] and not asked

        fs.mark_unsaved(layer, [0])
        window.close(quit_app=True, confirm_need=True)
        # Asked here, so napari is told not to ask.
        assert window.calls[-1] == (True, False) and len(asked) == 1

    def test_declining_at_the_menu_does_not_close(self, qtbot, monkeypatch):
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)

        class NapariWindow(QWidget):
            _is_close_dialog = {False: True, True: True}
            calls = []

            def close(self, quit_app=False, confirm_need=False):
                self.calls.append(confirm_need)
                return super().close()

        window = NapariWindow()
        qtbot.addWidget(window)
        layer = self._layer()
        nu.guard_unsaved_close(
            SimpleNamespace(window=SimpleNamespace(_qt_window=window)), layer, "x"
        )
        fs.mark_unsaved(layer, [0])
        assert window.close(confirm_need=True) is None
        assert window.calls == []


class TestReportFailure:

    def test_a_failure_is_shown_and_swallowed(self, monkeypatch):
        shown = []
        monkeypatch.setattr(
            QMessageBox, "warning", lambda *a, **k: shown.append(a[1:3])
        )

        def fail():
            raise OSError("disk full")

        assert nu.report_failure(None, "Save the modified labels", fail) is None
        assert shown[0][0] == "Save the modified labels"
        assert "disk full" in shown[0][1]

    def test_success_returns_the_result_without_a_box(self, monkeypatch):
        shown = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a))
        assert nu.report_failure(None, "x", lambda: 3) == 3
        assert not shown
