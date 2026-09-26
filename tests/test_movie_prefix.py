"""
Tests for the movie prefixes derived from the stacks an experiment holds.

The prefix stored in ``config.ini`` selects the stack of a position folder
(``prefix*.tif``): these helpers read the names that are actually there and
turn them into the prefixes worth proposing.
"""

import os

from celldetective.utils import experiment
from celldetective.utils.experiment import (
    _prefixes_of_name,
    count_movies_matching_prefix,
    get_movie_prefix_candidates,
    list_movies_per_position,
)


def by_position(movies):
    """Key the stacks of each position by the name of its folder."""
    return {os.path.basename(os.path.normpath(p)): names for p, names in movies.items()}


def test_prefixes_of_name_cuts_on_separators_and_numbering():
    assert _prefixes_of_name("Well1_t01.tif") == [
        "Well",
        "Well1",
        "Well1_",
        "Well1_t",
        "Well1_t01",
    ]


def test_prefixes_of_name_ignores_the_extension_and_empty_pieces():
    assert _prefixes_of_name("_A.tif") == ["_A"]
    assert "stack.tif" not in _prefixes_of_name("stack.tif")


def test_prefixes_of_name_never_end_on_a_space():
    # The prefix is saved stripped: 'Well ' would be saved as 'Well'.
    assert all(p == p.strip() for p in _prefixes_of_name("Well 1.tif"))


def test_prefixes_of_name_stop_before_the_wildcards_of_glob():
    # 'img[1]_' would be globbed as the character class [1], matching 'img1_'.
    assert _prefixes_of_name("img[1]_t01.tif") == ["img"]


def test_list_movies_per_position_reads_the_movie_folders(tmp_path, write_stacks):
    folder = write_stacks(
        tmp_path,
        {
            ("W1", "101"): ["Alexa488_stack.tif", "BF_stack.tif"],
            ("W1", "102"): [],
            ("W2", "201"): ["Alexa488_stack.tif", "notes.txt"],
        },
    )
    movies = by_position(list_movies_per_position(folder))

    assert sorted(movies) == ["101", "102", "201"]
    assert movies["101"] == ["Alexa488_stack.tif", "BF_stack.tif"]
    # Only the stacks the software can load.
    assert movies["201"] == ["Alexa488_stack.tif"]


def test_list_movies_per_position_keeps_positions_without_movie_folder(tmp_path, write_stacks):
    folder = write_stacks(tmp_path, {("W1", "101"): ["a.tif"]})
    (tmp_path / "W1" / "102").mkdir(parents=True)
    assert by_position(list_movies_per_position(folder)) == {"101": ["a.tif"], "102": []}


def test_count_movies_matching_prefix():
    movies = {
        "101": ["Alexa488_stack.tif", "BF_stack.tif"],
        "102": ["Alexa488_stack.tif"],
        "103": [],
    }
    assert count_movies_matching_prefix(movies, "Alexa488_") == (2, 2)
    assert count_movies_matching_prefix(movies, "") == (2, 3)
    assert count_movies_matching_prefix(movies, "Hoechst") == (0, 0)


def test_candidates_separate_the_channels_of_a_position():
    movies = {
        "101": ["Alexa488_stack.tif", "BF_stack.tif"],
        "102": ["Alexa488_stack.tif", "BF_stack.tif"],
    }
    assert get_movie_prefix_candidates(movies) == ["Alexa488_", "BF_"]


def test_candidates_put_the_covering_prefix_first():
    movies = {f"10{i}": [f"sample_W1_10{i}.tif"] for i in range(3)}
    candidates = get_movie_prefix_candidates(movies)

    assert candidates[0] == "sample_"
    # The name of one position is not the prefix of the experiment.
    assert all(count_movies_matching_prefix(movies, p)[0] > 1 for p in candidates)


def test_the_names_of_single_positions_are_a_last_resort():
    movies = {"101": ["alpha.tif"], "102": ["beta.tif"]}
    assert get_movie_prefix_candidates(movies) == ["alpha", "beta"]


def test_candidates_prefer_the_prefix_ending_on_a_separator():
    # 'Alexa', 'Alexa488' and 'Alexa488_' match the same stack.
    movies = {"101": ["Alexa488_stack.tif"]}
    assert get_movie_prefix_candidates(movies) == ["Alexa488_"]


def test_candidates_compare_names_the_way_glob_does(monkeypatch):
    # On Windows glob ignores the case: 'BF_' and 'bf_' select the same stacks.
    monkeypatch.setattr(experiment.os.path, "normcase", str.lower)
    movies = {"101": ["BF_1.tif"], "102": ["BF_1.tif"], "103": ["bf_1.tif"]}
    assert get_movie_prefix_candidates(movies)[0].lower() == "bf_"
    assert len(get_movie_prefix_candidates(movies)) == 1


def test_candidates_are_limited():
    movies = {f"1{i:02d}": [f"c{c}_{i}_t00.tif" for c in range(6)] for i in range(20)}
    assert len(get_movie_prefix_candidates(movies, limit=5)) == 5


def test_no_candidate_without_stacks():
    assert get_movie_prefix_candidates({}) == []
    assert get_movie_prefix_candidates({"101": []}) == []
