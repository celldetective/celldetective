import os

import numpy as np
import pytest
import tifffile
from scipy.ndimage import gaussian_filter

from celldetective.preprocessing import register_single_stack, register_stacks
from celldetective.utils.registration import (
    downscale_frame,
    estimate_drift,
    estimate_shift,
    tukey_window,
)

FIELD = 128
MARGIN = 20


def _texture(seed=0, size=FIELD + 2 * MARGIN):
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.random((size, size)), 2.0) * 1000.0


def _drifting_crops(drifts, seed=0):
    """Crops of a large texture whose content moves by `drifts[t] = (dy, dx)`."""
    texture = _texture(seed)
    return [
        texture[MARGIN + dy : MARGIN + dy + FIELD, MARGIN + dx : MARGIN + dx + FIELD]
        for dy, dx in drifts
    ]


DRIFTS = [(0, 0), (2, -1), (5, -3), (7, 1), (10, 4)]


def test_tukey_window_full_frame_and_radius():
    full = tukey_window((64, 80), alpha=0.5)
    assert full.shape == (64, 80)
    assert full.max() == pytest.approx(1.0)
    assert full[0, 0] == pytest.approx(0.0)

    disk = tukey_window((64, 64), alpha=0.5, radius=20)
    assert disk[32, 32] == pytest.approx(1.0)
    assert disk[0, 0] == 0.0
    assert np.all(disk[32, 32 + 21 :] == 0.0)

    with pytest.raises(ValueError):
        tukey_window((64, 64), alpha=1.5)


@pytest.mark.parametrize("reference", ["previous", "first"])
def test_estimate_drift_recovers_known_translation(reference):
    frames = _drifting_crops(DRIFTS)
    window = tukey_window((FIELD, FIELD), alpha=0.25)
    shifts = estimate_drift(frames, window, reference=reference, upsample_factor=10)
    # A crop taken at offset +d shows the content displaced by -d, so the registering shift is +d.
    np.testing.assert_allclose(shifts, np.array(DRIFTS, dtype=float), atol=0.3)


@pytest.mark.parametrize("reference", ["previous", "first"])
def test_estimate_drift_bridges_blank_frame(reference):
    frames = [f.copy() for f in _drifting_crops(DRIFTS)]
    frames[2] = np.zeros_like(frames[2])
    window = tukey_window((FIELD, FIELD), alpha=0.25)
    shifts = estimate_drift(frames, window, reference=reference, upsample_factor=10)
    expected = np.array(DRIFTS, dtype=float)
    # The blank frame keeps the shift before it; later frames keep the full drift.
    expected[2] = expected[1]
    np.testing.assert_allclose(shifts, expected, atol=0.3)


@pytest.mark.parametrize("reference", ["previous", "first"])
def test_estimate_drift_skips_blank_first_frame(reference):
    frames = [f.copy() for f in _drifting_crops(DRIFTS)]
    frames[0] = np.zeros_like(frames[0])
    window = tukey_window((FIELD, FIELD), alpha=0.25)
    shifts = estimate_drift(frames, window, reference=reference, upsample_factor=10)
    # Shifts are relative to the first frame with signal.
    expected = np.array(DRIFTS, dtype=float) - np.array(DRIFTS[1], dtype=float)
    expected[0] = 0.0
    np.testing.assert_allclose(shifts, expected, atol=0.3)


def test_radius_ignores_static_edge_artefact():
    reference, moving = _drifting_crops([(0, 0), (6, -4)], seed=1)
    reference, moving = reference.copy(), moving.copy()
    # A very bright artefact fixed to the camera, near a corner, outside the ROI.
    for img in (reference, moving):
        img[:25, :25] += 5e5

    window = tukey_window((FIELD, FIELD), alpha=0.25, radius=45)
    shift = estimate_shift(reference, moving, window, upsample_factor=10)
    np.testing.assert_allclose(shift, [6, -4], atol=0.3)


def test_estimate_shift_handles_nan_and_empty_frames():
    reference, moving = _drifting_crops([(0, 0), (3, 2)], seed=2)
    moving = moving.copy()
    moving[:5, :] = np.nan
    window = tukey_window((FIELD, FIELD), alpha=0.25, radius=55)
    np.testing.assert_allclose(estimate_shift(reference, moving, window), [3, 2], atol=0.3)
    np.testing.assert_array_equal(
        estimate_shift(reference, np.zeros_like(reference), window), [0, 0]
    )


def _write_experiment(tmp_path, nbr_channels=2):
    exp_dir = tmp_path / "Experiment"
    movie_dir = exp_dir / "W1" / "100" / "movie"
    movie_dir.mkdir(parents=True)
    (exp_dir / "W1" / "100" / "output" / "tables").mkdir(parents=True)

    registration = np.stack(_drifting_crops(DRIFTS, seed=3))
    other = np.stack(_drifting_crops(DRIFTS, seed=4))
    stack = np.stack([registration, other][:nbr_channels], axis=1).astype(np.float32)
    tifffile.imwrite(
        movie_dir / "sample.tif", stack, imagej=True, metadata={"axes": "TCYX"}
    )

    channels = "\n".join(f"Channel{c + 1} = {c}" for c in range(nbr_channels))
    (exp_dir / "config.ini").write_text(
        f"[MovieSettings]\nmovie_prefix = sample\nlen_movie = {len(DRIFTS)}\n"
        f"shape_x = {FIELD}\nshape_y = {FIELD}\npxtoum = 1.0\nframetomin = 1.0\n"
        "[Labels]\nconcentrations = 0\ncell_types = dummy\nantibodies = none\n"
        f"pharmaceutical_agents = none\n[Channels]\n{channels}\n"
    )
    return str(exp_dir) + os.sep, movie_dir


def test_register_single_stack_aligns_all_channels(tmp_path):
    _, movie_dir = _write_experiment(tmp_path)
    registered = register_single_stack(
        str(movie_dir / "sample.tif"),
        registration_channel_index=0,
        nbr_channels=2,
        radius=50,
        return_stacks=True,
    )
    assert registered.shape == (len(DRIFTS), FIELD, FIELD, 2)

    # Away from the borders filled with zeros, every frame matches the first in both channels.
    inner = slice(MARGIN, FIELD - MARGIN)
    for c in range(2):
        for t in range(1, len(DRIFTS)):
            np.testing.assert_allclose(
                registered[t, inner, inner, c], registered[0, inner, inner, c], rtol=0.05
            )

    shifts = np.loadtxt(
        movie_dir / "Corrected_sample_registration_shifts.csv", delimiter=",", skiprows=1
    )
    np.testing.assert_allclose(shifts[:, 1:], np.array(DRIFTS, dtype=float), atol=0.3)


def test_downscale_frame_block_average():
    frame = np.arange(7 * 9, dtype=float).reshape(7, 9)
    reduced = downscale_frame(frame, 2)
    assert reduced.shape == (3, 4)
    assert reduced[0, 0] == pytest.approx(frame[:2, :2].mean())
    assert downscale_frame(frame, 1) is frame
    with pytest.raises(ValueError):
        downscale_frame(frame, 0)


@pytest.mark.parametrize("downscale", [2, 4])
def test_downscaled_registration_applies_full_scale_shifts(tmp_path, downscale):
    _, movie_dir = _write_experiment(tmp_path)
    registered = register_single_stack(
        str(movie_dir / "sample.tif"),
        registration_channel_index=0,
        nbr_channels=2,
        radius=50,
        downscale=downscale,
        return_stacks=True,
    )
    assert registered.shape == (len(DRIFTS), FIELD, FIELD, 2)

    shifts = np.loadtxt(
        movie_dir / "Corrected_sample_registration_shifts.csv", delimiter=",", skiprows=1
    )
    np.testing.assert_allclose(shifts[:, 1:], np.array(DRIFTS, dtype=float), atol=0.5)


def test_register_stacks_exports_corrected_movie(tmp_path):
    exp_dir, movie_dir = _write_experiment(tmp_path)
    register_stacks(
        exp_dir,
        well_option="*",
        position_option="*",
        target_channel="Channel1",
        radius=50,
        export=True,
        show_progress_per_well=False,
        show_progress_per_pos=False,
    )
    exported = movie_dir / "Corrected_sample.tif"
    assert exported.exists()
    assert tifffile.imread(exported).shape == (len(DRIFTS), 2, FIELD, FIELD)
    assert (tmp_path / "Experiment" / "W1" / "100" / "log_preprocessing.txt").exists()


def test_register_stacks_skips_position_without_movie(tmp_path):
    exp_dir, movie_dir = _write_experiment(tmp_path)
    # A position listed before the valid one, with no movie matching the prefix.
    (tmp_path / "Experiment" / "W1" / "099" / "movie").mkdir(parents=True)
    register_stacks(
        exp_dir,
        well_option="*",
        position_option="*",
        target_channel="Channel1",
        radius=50,
        export=True,
        show_progress_per_well=False,
        show_progress_per_pos=False,
    )
    assert (movie_dir / "Corrected_sample.tif").exists()
