"""Small experiments on disk for the preprocessing tests."""

import numpy as np
import tifffile

FIELD = 64


def write_config(experiment_dir, prefix="sample", frames=3):
    """Write the ``config.ini`` of a one-channel experiment."""
    (experiment_dir / "config.ini").write_text(
        f"[MovieSettings]\nmovie_prefix = {prefix}\nlen_movie = {frames}\n"
        f"shape_x = {FIELD}\nshape_y = {FIELD}\npxtoum = 1.0\nframetomin = 1.0\n"
        "[Labels]\nconcentrations = 0\ncell_types = dummy\nantibodies = none\n"
        "pharmaceutical_agents = none\n[Channels]\nChannel1 = 0\n"
    )


def write_movie(path, stack):
    """Write a ``TYX`` stack as ImageJ does, the format of the movies."""
    tifffile.imwrite(
        path, np.asarray(stack, dtype=np.float32), imagej=True, metadata={"axes": "TYX"}
    )


def write_experiment(tmp_path, positions_per_well, frames=3, prefix="sample"):
    """
    An experiment with noisy flat movies named ``sample.tif``.

    Returns
    -------
    tuple
        The experiment path as a string, and the movie folder of each position.
    """
    rng = np.random.default_rng(0)
    experiment_dir = tmp_path / "Experiment"
    movie_dirs = []
    for w, n_positions in enumerate(positions_per_well, start=1):
        for p in range(n_positions):
            movie_dir = experiment_dir / f"W{w}" / f"{w}{p:02d}" / "movie"
            movie_dir.mkdir(parents=True)
            stack = 100 + rng.normal(0, 1, (frames, FIELD, FIELD))
            write_movie(movie_dir / "sample.tif", stack)
            movie_dirs.append(movie_dir)
    write_config(experiment_dir, prefix, frames)
    return str(experiment_dir) + "/", movie_dirs
