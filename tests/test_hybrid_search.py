"""Tests for the normalize_scores utility in hybrid_search."""

import pytest
from cli.lib.hybrid_search import normalize_scores


class TestNormalizeScores:
    def test_min_becomes_zero(self):
        result = normalize_scores([1.0, 2.0, 3.0])
        assert result[0] == 0.0

    def test_max_becomes_one(self):
        result = normalize_scores([1.0, 2.0, 3.0])
        assert result[-1] == 1.0

    def test_middle_value_is_scaled(self):
        result = normalize_scores([0.0, 5.0, 10.0])
        assert abs(result[1] - 0.5) < 1e-9

    def test_all_same_values_return_ones(self):
        result = normalize_scores([7.0, 7.0, 7.0])
        assert result == [1.0, 1.0, 1.0]

    def test_single_element_returns_one(self):
        result = normalize_scores([42.0])
        assert result == [1.0]

    def test_negative_values_handled(self):
        result = normalize_scores([-3.0, 0.0, 3.0])
        assert result[0] == 0.0
        assert result[-1] == 1.0
        assert abs(result[1] - 0.5) < 1e-9

    def test_output_length_matches_input(self):
        scores = [1.0, 3.0, 2.0, 5.0, 4.0]
        result = normalize_scores(scores)
        assert len(result) == len(scores)

    def test_order_preserved(self):
        scores = [3.0, 1.0, 2.0]
        result = normalize_scores(scores)
        assert result[0] > result[1]
        assert result[0] > result[2]
