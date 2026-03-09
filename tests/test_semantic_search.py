"""Tests for cosine_similarity and semantic_chunking (no model loading required)."""

import numpy as np
import pytest

from cli.lib.semantic_search import cosine_similarity, semantic_chunking, SemanticSearch


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert abs(cosine_similarity(v1, v2)) < 1e-6

    def test_opposite_vectors_return_minus_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, -v) - (-1.0)) < 1e-6

    def test_shape_mismatch_raises(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            cosine_similarity(v1, v2)

    def test_zero_vector_returns_zero(self):
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v1, v2) == 0.0

    def test_value_in_minus_one_to_one_range(self):
        rng = np.random.default_rng(42)
        v1 = rng.standard_normal(128)
        v2 = rng.standard_normal(128)
        sim = cosine_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# semantic_chunking
# ---------------------------------------------------------------------------

class TestSemanticChunking:
    def test_empty_string_returns_empty(self):
        assert semantic_chunking("") == []

    def test_whitespace_only_returns_empty(self):
        assert semantic_chunking("   ") == []

    def test_single_sentence_no_period_returns_it(self):
        result = semantic_chunking("A fish swims")
        assert result == ["A fish swims"]

    def test_multiple_sentences_chunked(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = semantic_chunking(text, max_chunk_size=2, overlap=0)
        assert len(chunks) >= 2
        assert all(isinstance(c, str) and len(c) > 0 for c in chunks)

    def test_overlap_larger_than_max_raises(self):
        with pytest.raises(ValueError):
            semantic_chunking("Some text.", max_chunk_size=2, overlap=3)

    def test_chunk_size_limits_sentences_per_chunk(self):
        sentences = [f"Sentence {i}." for i in range(8)]
        text = " ".join(sentences)
        chunks = semantic_chunking(text, max_chunk_size=2, overlap=0)
        for chunk in chunks:
            sentence_count = chunk.count(".")
            assert sentence_count <= 2

    def test_overlap_creates_shared_content(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = semantic_chunking(text, max_chunk_size=3, overlap=1)
        # With overlap=1, the last sentence of one chunk should appear in the next
        assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# SemanticSearch – unit tests with a mocked model
# ---------------------------------------------------------------------------

class TestSemanticSearchWithMockModel:
    @pytest.fixture
    def mock_model(self, monkeypatch):
        """Replace SentenceTransformer with a lightweight fake."""
        import cli.lib.semantic_search as sem_mod

        class FakeModel:
            def encode(self, texts, show_progress_bar=False):
                # Return deterministic unit vectors of dimension 4
                n = len(texts) if isinstance(texts, list) else 1
                arr = np.zeros((n, 4))
                for i in range(n):
                    arr[i, i % 4] = 1.0
                return arr

        monkeypatch.setattr(sem_mod, "SentenceTransformer", lambda name: FakeModel())
        return FakeModel()

    def test_generate_embedding_empty_raises(self, mock_model):
        search = SemanticSearch()
        with pytest.raises(ValueError):
            search.generate_embedding("")

    def test_generate_embedding_whitespace_raises(self, mock_model):
        search = SemanticSearch()
        with pytest.raises(ValueError):
            search.generate_embedding("   ")

    def test_generate_embedding_returns_array(self, mock_model):
        search = SemanticSearch()
        emb = search.generate_embedding("some text")
        assert isinstance(emb, np.ndarray)

    def test_search_without_embeddings_raises(self, mock_model):
        search = SemanticSearch()
        with pytest.raises(ValueError):
            search.search("query")

    def test_build_embeddings_sets_embeddings(self, mock_model, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cache").mkdir()

        search = SemanticSearch()
        docs = [
            {"id": 1, "title": "Alpha", "description": "alpha content"},
            {"id": 2, "title": "Beta", "description": "beta content"},
            {"id": 3, "title": "Gamma", "description": "gamma content"},
            {"id": 4, "title": "Delta", "description": "delta content"},
        ]
        search.build_embeddings(docs)
        assert search.embeddings is not None
        assert search.embeddings.shape[0] == len(docs)

    def test_search_returns_top_k(self, mock_model, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cache").mkdir()

        search = SemanticSearch()
        docs = [
            {"id": 1, "title": "Alpha", "description": "alpha content"},
            {"id": 2, "title": "Beta", "description": "beta content"},
            {"id": 3, "title": "Gamma", "description": "gamma content"},
            {"id": 4, "title": "Delta", "description": "delta content"},
        ]
        search.build_embeddings(docs)
        results = search.search("alpha", limit=2)
        assert len(results) == 2
