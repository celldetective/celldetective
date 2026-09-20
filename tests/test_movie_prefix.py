"""
Tests for the movie prefixes derived from the stacks an experiment holds.

The prefix stored in ``config.ini`` selects the stack of a position folder
(``prefix*.tif``): these helpers read the names that are actually there and
turn them into the prefixes worth proposing.
"""

import os

from celldetective.utils.experiment import (
    _prefixes_of_name,
    count_movies_matching_prefix,
    get_movie_prefix_candidates,
    list_movies_per_position,
)


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


def test_list_movies_per_position_reads_the_movie_folders(tmp_path, write_stacks):
    folder = write_stacks(
        tmp_path,
        {
            ("W1", "101"): ["Alexa488_stack.tif", "BF_stack.tif"],
            ("W1", "102"): [],
            ("W2", "201"): ["Alexa488_stack.tif", "notes.txt"],
        },
    )
    movies = list_movies_per_position(folder)

    assert sorted(os.path.basename(os.path.normpath(p)) for p in movies) == [
        "101",
        "102",
        "201",
    ]
    assert sorted(next(v for k, v in movies.items() if k.rstrip(os.sep).endswith("101"))) == [
        "Alexa488_stack.tif",
        "BF_stack.tif",
    ]
    # Only the stacks the software can load.
    assert all("notes.txt" not in names for names in movies.values())


def test_list_movies_per_position_skips_positions_without_movie_folder(tmp_path, write_stacks):
    folder = write_stacks(tmp_path, {("W1", "101"): ["a.tif"]})
    (tmp_path / "W1" / "102").mkdir(parents=True)
    assert len(list_movies_per_position(folder)) == 1


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
    assert get_movie_prefix_candidates(movies) == [
        ("Alexa488_", 2, 2),
        ("BF_", 2, 2),
    ]


def test_candidates_put_the_covering_prefix_first():
    movies = {f"10{i}": [f"sample_W1_10{i}.tif"] for i in range(3)}
    candidates = get_movie_prefix_candidates(movies)

    assert candidates[0] == ("sample_", 3, 3)
    # The name of one position is not the prefix of the experiment.
    assert all(positions > 1 for _, positions, _ in candidates)


def test_the_names_of_single_positions_are_a_last_resort():
    movies = {"101": ["alpha.tif"], "102": ["beta.tif"]}
    assert get_movie_prefix_candidates(movies) == [("alpha", 1, 1), ("beta", 1, 1)]


def test_candidates_are_the_shortest_of_the_prefixes_matching_the_same_stacks():
    movies = {"101": ["Alexa488_stack.tif"]}
    prefixes = [c[0] for c in get_movie_prefix_candidates(movies)]
    assert prefixes == ["Alexa488_"]


def test_candidates_are_limited():
    movies = {f"1{i:02d}": [f"c{c}_{i}_t00.tif" for c in range(6)] for i in range(20)}
    assert len(get_movie_prefix_candidates(movies, limit=5)) == 5


def test_no_candidate_without_stacks():
    assert get_movie_prefix_candidates({}) == []
    assert get_movie_prefix_candidates({"101": []}) == []
