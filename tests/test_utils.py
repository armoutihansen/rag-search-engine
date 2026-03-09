"""Tests for the load_movies_data utility."""

import json
import pytest
from cli.lib.utils import load_movies_data


class TestLoadMoviesData:
    def test_returns_list(self):
        movies = load_movies_data()
        assert isinstance(movies, list)

    def test_list_is_non_empty(self):
        movies = load_movies_data()
        assert len(movies) > 0

    def test_each_movie_has_id_and_title(self):
        movies = load_movies_data()
        for movie in movies:
            assert "id" in movie, f"Missing 'id' in: {movie}"
            assert "title" in movie, f"Missing 'title' in: {movie}"

    def test_ids_are_unique(self):
        movies = load_movies_data()
        ids = [m["id"] for m in movies]
        assert len(ids) == len(set(ids))

    def test_custom_path_loads_correct_data(self, tmp_path):
        payload = {"movies": [{"id": 99, "title": "Test Movie", "description": "Test"}]}
        custom_file = tmp_path / "test_movies.json"
        custom_file.write_text(json.dumps(payload), encoding="utf-8")

        movies = load_movies_data(str(custom_file))
        assert len(movies) == 1
        assert movies[0]["title"] == "Test Movie"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_movies_data("/nonexistent/path/movies.json")
