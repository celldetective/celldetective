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
            opt_coef_nbr=21,
            opt_radius=opt_radius,
            operation="divide",
        )
        return 1.05 / corrected[size // 2, size // 2, 0]

    # Over the full frame, the high-variance diaphragm edge is a closed ring whose inside is
    # masked by the hole filling: only the dark diaphragm is left and drags the fit down.
    assert fitted_coefficient(None) == pytest.approx(0.9)
    assert fitted_coefficient(field_radius - 5) == pytest.approx(1.05)


def test_coefficient_search_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(1, 300)
        background = np.round(rng.normal(100, 60, n))
        target = np.round(
            background * rng.uniform(0.8, 1.2) + rng.normal(0, rng.uniform(0, 50), n)
        )
        grid = np.append(np.linspace(0.9, 1.1, rng.integers(1, 120)), [1.0])

        losses = [np.sum(np.abs(target - c * background)) for c in grid]
        found = _best_l1_coefficient(target, background, grid)
        assert np.sum(np.abs(target - found * background)) == pytest.approx(min(losses))

