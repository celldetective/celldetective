import shutil

import numpy as np
import pytest
import tifffile
from scipy.ndimage import shift

from celldetective.preprocessing import (
    _best_l1_coefficient,
    apply_background_to_stack,
    correct_background_model,
    correct_background_model_free,
    correct_channel_offset,
    estimate_background_per_condition,
    register_stacks,
)
from tests.test_registration import DRIFTS, FIELD, _write_experiment

QUIET = dict(show_progress_per_well=False, show_progress_per_pos=False)


@pytest.fixture
def experiment(tmp_path):
    exp_dir, movie_dir = _write_experiment(tmp_path)
    # A position without a movie, listed before the valid one: every correction skips it.
    (tmp_path / "Experiment" / "W1" / "10" / "movie").mkdir(parents=True)
    source = tifffile.imread(movie_dir / "sample.tif").astype(float)
    return exp_dir, movie_dir, source


def test_channel_offset_shifts_target_channel_only(experiment):
    exp_dir, movie_dir, source = experiment
    stacks = correct_channel_offset(
        exp_dir,
        target_channel="Channel2",
        correction_vertical=-2,
        correction_horizontal=3,
        export=True,
        return_stacks=True,
        **QUIET,
    )

    (corrected,) = stacks
    assert corrected.shape == (len(DRIFTS), FIELD, FIELD, 2)
    np.testing.assert_allclose(corrected[..., 0], source[:, 0], rtol=1e-6)
    np.testing.assert_allclose(
        corrected[2, ..., 1], shift(source[2, 1], [-2, 3]), rtol=1e-5
    )
    exported = tifffile.imread(movie_dir / "Corrected_sample.tif")
    np.testing.assert_allclose(exported, np.moveaxis(corrected, -1, 1))


@pytest.mark.parametrize(
    "correct",
    [
        lambda exp, **kw: correct_background_model(
            exp, target_channel="Channel1", model="plane", operation="subtract", **kw
        ),
        lambda exp, **kw: correct_background_model_free(
            exp, target_channel="Channel1", mode="tiles", operation="subtract", **kw
        ),
        lambda exp, **kw: correct_channel_offset(
            exp, target_channel="Channel1", correction_horizontal=1, **kw
        ),
        lambda exp, **kw: register_stacks(
            exp, target_channel="Channel1", radius=50, **kw
        ),
    ],
    ids=["fit", "model-free", "offset", "registration"],
)
def test_overwriting_correction_replaces_source_without_leftovers(experiment, correct):
    exp_dir, movie_dir, source = experiment
    correct(exp_dir, export=True, export_prefix=None, **QUIET)

    assert not list(movie_dir.glob("temp_*"))
    assert not list(movie_dir.glob("Corrected_*"))
    overwritten = tifffile.imread(movie_dir / "sample.tif")
    assert overwritten.shape == source.shape
    assert overwritten.dtype == np.float32
    # The other channel of the first frame is left untouched (registration has no shift there).
    np.testing.assert_allclose(overwritten[0, 1], source[0, 1], rtol=1e-5)
    assert (movie_dir.parent / "log_preprocessing.txt").exists()


def test_corrections_report_well_and_position_progress(experiment):
    exp_dir, _, _ = experiment
    calls = []
    correct_channel_offset(
        exp_dir,
        target_channel="Channel2",
        correction_horizontal=1,
        progress_callback=lambda **kw: calls.append(kw),
        **QUIET,
    )

    wells = [c["iter"] for c in calls if c.get("level") == "well"]
    positions = [c for c in calls if c.get("level") == "position"]
    assert wells == [0, 1]
    # Reported once each position is done, including the one skipped for lack of a movie.
    assert [c["iter"] for c in positions] == [0, 1]
    assert all(c["total"] == 2 for c in positions)


def test_model_free_coefficient_fit_ignores_pixels_outside_radius(tmp_path):
    # Background at 100 everywhere; the frame is 5 % brighter in the field, but a diaphragm
    # darkens it to the camera black level outside a centred disk.
    size, field_radius = 64, 20
    yy, xx = np.mgrid[:size, :size]
    inside = np.hypot(yy - (size - 1) / 2, xx - (size - 1) / 2) <= field_radius
    background = np.full((size, size), 100.0)
    frame = np.where(inside, 105.0, 0.0).astype(np.float32)
    stack_path = tmp_path / "sample.tif"
    tifffile.imwrite(stack_path, frame[np.newaxis], imagej=True, metadata={"axes": "TYX"})

    def fitted_coefficient(opt_radius):
        (corrected,) = apply_background_to_stack(
            str(stack_path),
            background,
            stack_length=1,
            optimize_option=True,
            opt_coef_range=(0.9, 1.1),
            opt_radius=opt_radius,
            operation="divide",
        )
        return 1.05 / corrected[size // 2, size // 2, 0]

    # Over the full frame, the high-variance diaphragm edge is a closed ring whose inside is
    # masked by the hole filling: only the dark diaphragm is left and drags the fit down to
    # the lower bound of the range.
    assert fitted_coefficient(None) == pytest.approx(0.9)
    assert fitted_coefficient(field_radius - 5) == pytest.approx(1.05)


def test_coefficient_is_the_exact_l1_optimum():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(1, 300)
        background = np.round(rng.normal(100, 60, n))
        target = np.round(
            background * rng.uniform(0.8, 1.2) + rng.normal(0, rng.uniform(0, 50), n)
        )

        def loss(c):
            return np.sum(np.abs(target - c * background))

        # The piecewise linear loss reaches its minimum at one of its breakpoints.
        nonzero = background != 0
        breakpoints = target[nonzero] / background[nonzero] if np.any(nonzero) else [1.0]
        found = _best_l1_coefficient(target, background)
        assert loss(found) == pytest.approx(min(loss(c) for c in breakpoints))


def test_coefficient_of_a_large_frame_is_the_weighted_median():
    # Enough pixels for the binning, with outliers stretching the range of the ratios.
    rng = np.random.default_rng(1)
    background = rng.normal(1000, 50, 200_000)
    target = background * 1.02 + rng.normal(0, 20, background.size)
    target[:100] = 1e9

    ratio = target / background
    order = np.argsort(ratio)
    cumulative = np.cumsum(background[order])
    expected = ratio[order[np.searchsorted(cumulative, 0.5 * cumulative[-1])]]

    assert _best_l1_coefficient(target, background) == expected


def test_model_free_preview_corrects_only_the_subset(experiment):
    exp_dir, movie_dir, source = experiment
    stack_path = str(movie_dir / "sample.tif")
    background = np.full((FIELD, FIELD), 2.0)
    options = dict(nbr_channels=2, stack_length=len(DRIFTS), optimize_option=False)

    full = apply_background_to_stack(stack_path, background, **options)
    subset = apply_background_to_stack(
        stack_path, background, subset_indices=[2 * 2], **options
    )

    assert subset.shape == (1, FIELD, FIELD, 2)
    np.testing.assert_allclose(subset[0], full[2])
    np.testing.assert_allclose(subset[0, ..., 0], source[2, 0] / 2.0, rtol=1e-6)


def test_model_free_export_returns_nothing_unless_asked(experiment):
    exp_dir, movie_dir, source = experiment
    stacks = correct_background_model_free(
        exp_dir, target_channel="Channel1", mode="tiles", export=True, **QUIET
    )

    assert stacks is None
    assert tifffile.imread(movie_dir / "Corrected_sample.tif").shape == source.shape


def test_chained_model_free_estimates_the_background_on_the_movie_it_corrects(
    experiment,
):
    exp_dir, movie_dir, source = experiment
    # A previous step of the protocol left a flat corrected movie next to the raw one.
    flat = np.full(source.shape, 7.0, dtype=np.float32)
    tifffile.imwrite(
        movie_dir / "Corrected_sample.tif", flat, imagej=True, metadata={"axes": "TCYX"}
    )

    (corrected,) = correct_background_model_free(
        exp_dir,
        target_channel="Channel1",
        mode="tiles",
        operation="subtract",
        movie_prefix="Corrected",
        return_stacks=True,
        **QUIET,
    )

    # Background from the raw movie, the textured one, would leave its texture behind.
    np.testing.assert_allclose(corrected[..., 0], 0.0, atol=1e-5)


def test_timeseries_background_leaves_out_a_transient_object(experiment):
    exp_dir, movie_dir, source = experiment
    rng = np.random.default_rng(0)
    movie = 100.0 + rng.normal(0, 1, source.shape)
    # A bright object crossing the field in one frame of the range only.
    movie[2, 0, 50:80, 50:80] = 1000.0
    tifffile.imwrite(
        movie_dir / "sample.tif",
        movie.astype(np.float32),
        imagej=True,
        metadata={"axes": "TCYX"},
    )

    (background,) = estimate_background_per_condition(
        exp_dir,
        threshold_on_std=5,
        target_channel="Channel1",
        frame_range=[0, len(DRIFTS)],
        mode="timeseries",
        show_progress_per_well=False,
    )

    # Masked in the frame showing it, the object leaves the background of the others.
    assert background["bg"][65, 65] == pytest.approx(100.0, abs=2.0)


def test_cancelled_estimation_stops_the_model_free_correction(experiment):
    exp_dir, movie_dir, _ = experiment
    wells = movie_dir.parents[2]
    shutil.copytree(wells / "W1", wells / "W2")
    calls = []

    def cancel(**kwargs):
        calls.append(kwargs)
        return False

    stacks = correct_background_model_free(
        exp_dir,
        target_channel="Channel1",
        export=True,
        return_stacks=True,
        progress_callback=cancel,
        **QUIET,
    )

    assert stacks == []
    # Stopped in the first well rather than skipping on to the next one.
    assert [c["iter"] for c in calls if c.get("level") == "well"] == [0]
    assert not list(wells.rglob("Corrected_*"))




def test_background_of_a_well_without_frames_is_none(experiment):
    exp_dir, movie_dir, source = experiment
    # A time range past the end of the movie leaves no frame in any position.
    start = len(DRIFTS) + 10

    backgrounds = estimate_background_per_condition(
        exp_dir,
        target_channel="Channel1",
        frame_range=[start, start + 5],
        mode="timeseries",
        show_progress_per_well=False,
    )

    # Rather than a scalar NaN, which would correct every pixel to NaN.
    assert backgrounds == [None]
