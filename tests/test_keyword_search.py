"""Tests for the InvertedIndex keyword search module."""

import math
import os
import pytest

from cli.lib.keyword_search import InvertedIndex
from tests.conftest import SAMPLE_MOVIES


@pytest.fixture
def index():
    idx = InvertedIndex()
    idx.build(SAMPLE_MOVIES)
    return idx


# ---------------------------------------------------------------------------
# Preprocessing and tokenization
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_removes_punctuation(self, index):
        assert index._preprocess("Hello, world!") == "hello world"

    def test_converts_to_lowercase(self, index):
        assert index._preprocess("BEAR") == "bear"

    def test_handles_mixed_input(self, index):
        assert index._preprocess("The Quick-Brown Fox.") == "the quickbrown fox"


class TestTokenize:
    def test_stems_words(self, index):
        tokens = index._tokenize("running")
        assert tokens == ["run"]

    def test_removes_stopwords(self, index):
        tokens = index._tokenize("the bear in a forest")
        assert "the" not in tokens
        assert "in" not in tokens
        assert "a" not in tokens

    def test_returns_list_of_strings(self, index):
        tokens = index._tokenize("bears wilderness")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_empty_string_returns_empty(self, index):
        assert index._tokenize("") == []


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

class TestBuild:
    def test_docmap_populated(self, index):
        assert len(index.docmap) == len(SAMPLE_MOVIES)

    def test_docmap_contains_correct_ids(self, index):
        ids = {m["id"] for m in SAMPLE_MOVIES}
        assert set(index.docmap.keys()) == ids

    def test_index_contains_terms(self, index):
        # "bear" should appear after stemming
        assert any("bear" in term for term in index.index)

    def test_doc_lengths_populated(self, index):
        for movie in SAMPLE_MOVIES:
            assert index.doc_lengths[movie["id"]] > 0


# ---------------------------------------------------------------------------
# get_documents
# ---------------------------------------------------------------------------

class TestGetDocuments:
    def test_returns_docs_containing_term(self, index):
        docs = index.get_documents("bear")
        assert 1 in docs  # "The Bear"
        assert 4 in docs  # "Jungle Book" (bears)

    def test_unknown_term_returns_empty(self, index):
        assert index.get_documents("zzznonsenseterm") == []

    def test_result_is_sorted(self, index):
        docs = index.get_documents("bear")
        assert docs == sorted(docs)


# ---------------------------------------------------------------------------
# TF / IDF / TF-IDF
# ---------------------------------------------------------------------------

class TestTermFrequency:
    def test_known_term_has_positive_tf(self, index):
        tf = index.get_tf(1, "bear")
        assert tf > 0

    def test_absent_term_has_zero_tf(self, index):
        tf = index.get_tf(1, "ocean")
        assert tf == 0

    def test_raises_for_multi_token_term(self, index):
        with pytest.raises(ValueError):
            index.get_tf(1, "grizzly bear")


class TestIDF:
    def test_idf_is_positive(self, index):
        idf = index.get_idf("bear")
        assert idf > 0

    def test_rare_term_has_higher_idf(self, index):
        # "ocean" appears in one doc, "bear" appears in more docs
        idf_ocean = index.get_idf("ocean")
        idf_bear = index.get_idf("bear")
        assert idf_ocean > idf_bear

    def test_idf_formula(self, index):
        """Verify the IDF formula: log((N+1) / (df+1))."""
        term_stemmed = index._tokenize("ocean")[0]
        n = len(index.docmap)
        df = len(index.index.get(term_stemmed, set()))
        expected = math.log((n + 1) / (df + 1))
        assert abs(index.get_idf("ocean") - expected) < 1e-9

    def test_raises_for_multi_token_term(self, index):
        with pytest.raises(ValueError):
            index.get_idf("ocean fish")


class TestTFIDF:
    def test_tfidf_positive_for_present_term(self, index):
        score = index.get_tfidf(1, "bear")
        assert score > 0

    def test_tfidf_zero_for_absent_term(self, index):
        score = index.get_tfidf(3, "bear")
        assert score == 0


# ---------------------------------------------------------------------------
# BM25 IDF / TF / score
# ---------------------------------------------------------------------------

class TestBM25IDF:
    def test_bm25_idf_positive(self, index):
        idf = index.get_bm25_idf("bear")
        assert idf > 0

    def test_bm25_idf_formula(self, index):
        """Verify BM25 IDF formula: log((N - df + 0.5) / (df + 0.5) + 1)."""
        term_stemmed = index._tokenize("bear")[0]
        n = len(index.docmap)
        df = len(index.index.get(term_stemmed, set()))
        expected = math.log((n - df + 0.5) / (df + 0.5) + 1)
        assert abs(index.get_bm25_idf("bear") - expected) < 1e-9

    def test_raises_for_multi_token_term(self, index):
        with pytest.raises(ValueError):
            index.get_bm25_idf("grizzly bear")


class TestBM25TF:
    def test_bm25_tf_between_zero_and_one(self, index):
        # BM25 TF is positive when the term is present; it is NOT bounded by 1 —
        # the asymptotic upper bound is (k1 + 1).
        tf = index.get_bm25_tf(1, "bear")
        assert tf > 0

    def test_bm25_tf_zero_for_absent_term(self, index):
        tf = index.get_bm25_tf(3, "bear")
        assert tf == 0


class TestBM25Score:
    def test_bm25_positive_for_present_term(self, index):
        score = index.bm25(1, "bear")
        assert score > 0

    def test_bm25_zero_for_absent_term(self, index):
        score = index.bm25(3, "bear")
        assert score == 0


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------

class TestBM25Search:
    def test_returns_list_of_dicts(self, index):
        results = index.bm25_search("bear", limit=3)
        assert isinstance(results, list)
        for r in results:
            assert {"id", "title", "document", "score"}.issubset(r.keys())

    def test_respects_limit(self, index):
        results = index.bm25_search("bear", limit=2)
        assert len(results) <= 2

    def test_bear_query_ranks_bear_movie_first(self, index):
        results = index.bm25_search("bear", limit=5)
        top_title = results[0]["title"]
        assert "Bear" in top_title or "Jungle" in top_title

    def test_results_sorted_by_score_descending(self, index):
        results = index.bm25_search("ocean fish", limit=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_irrelevant_query_still_returns_results(self, index):
        results = index.bm25_search("zzznonsense", limit=3)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    @pytest.fixture
    def tmp_project(self, tmp_path):
        """Mirror the minimal data directory so InvertedIndex can instantiate from tmp_path."""
        import shutil
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        (tmp_path / "data").mkdir()
        shutil.copy(os.path.join(project_root, "data", "stopwords.txt"), tmp_path / "data" / "stopwords.txt")
        return tmp_path

    def test_save_creates_files(self, index, tmp_project, monkeypatch):
        monkeypatch.chdir(tmp_project)
        index.save()
        assert (tmp_project / "cache" / "index.pkl").exists()
        assert (tmp_project / "cache" / "docmap.pkl").exists()
        assert (tmp_project / "cache" / "term_frequencies.pkl").exists()
        assert (tmp_project / "cache" / "doc_lengths.pkl").exists()

    def test_round_trip_preserves_index(self, index, tmp_project, monkeypatch):
        monkeypatch.chdir(tmp_project)
        index.save()

        new_idx = InvertedIndex()
        new_idx.load()

        assert new_idx.index == index.index
        assert new_idx.docmap == index.docmap

    def test_load_raises_when_no_cache(self, tmp_project, monkeypatch):
        monkeypatch.chdir(tmp_project)
        idx = InvertedIndex()
        with pytest.raises(FileNotFoundError):
            idx.load()
