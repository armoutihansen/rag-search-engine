import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from .constants import SEMANTIC_MODEL
from .utils import load_movies_data


class SemanticSearch:
    def __init__(self, model_name: str = SEMANTIC_MODEL):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if text is None or len(text.strip()) == 0:
            raise ValueError("Input text cannot be empty.")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list):
        self.documents = documents
        document_descriptions = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            description = f"{doc['title']}: {doc.get('description', '')}"
            document_descriptions.append(description)
        self.embeddings = self.model.encode(
            document_descriptions, show_progress_bar=True
        )
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        try:
            self.embeddings = np.load("cache/movie_embeddings.npy")
            print("Loaded cached embeddings.")
            if self.embeddings.shape[0] != len(documents):
                print(
                    "Warning: Number of cached embeddings does not match number of documents. Rebuilding embeddings..."
                )
                self.build_embeddings(documents)
            else:
                return self.embeddings
        except FileNotFoundError:
            print("No cached embeddings found. Building new embeddings...")
            self.build_embeddings(documents)
        return self.embeddings

    def search(self, query: str, limit: int = 5):
        results = {}  # Dictionary to store results with score, title, and description
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "Embeddings and documents must be loaded or created before searching."
            )
        query_embedding = self.generate_embedding(query)
        similarities = np.array(
            [
                cosine_similarity(query_embedding, doc_embedding)
                for doc_embedding in self.embeddings
            ]
        )
        top_indices = np.argsort(similarities)[::-1][:limit]
        for idx in top_indices:
            doc_id = self.documents[idx]["id"]
            doc = self.document_map[doc_id]
            results[doc_id] = {
                "score": similarities[idx],
                "title": doc["title"],
                "description": doc.get("description", ""),
            }
        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = SEMANTIC_MODEL):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list):
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        chunks = []
        metadata = []  # List of dicts to hold metadata for each chunk
        for j, doc in enumerate(documents):
            if doc.get("description") is None or len(doc["description"].strip()) == 0:
                continue
            description = doc["description"]
            doc_chunks = semantic_chunking(description, max_chunk_size=4, overlap=1)
            for i, chunk in enumerate(doc_chunks):
                chunks.append(chunk)
                metadata.append(
                    {
                        "movie_idx": j,
                        "chunk_idx": i,
                        "total_chunks": len(doc_chunks),
                    }
                )
        self.chunk_metadata = metadata
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": metadata, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}
        try:
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            print("Loaded cached chunk embeddings and metadata.")
            if self.chunk_embeddings.shape[0] != len(self.chunk_metadata):
                print(
                    "Warning: Number of cached chunk embeddings does not match number of metadata entries. Rebuilding chunk embeddings..."
                )
                self.build_chunk_embeddings(documents)
            else:
                return self.chunk_embeddings
        except FileNotFoundError:
            print("No cached chunk embeddings found. Building new chunk embeddings...")
            self.build_chunk_embeddings(documents)
        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "Chunk embeddings and metadata must be loaded or created before searching."
            )
        query_embedding = self.generate_embedding(query)
        chunks_scores = []
        chunk_embeddings = self.chunk_embeddings
        for i, chunk_embedding in enumerate(chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunks_scores.append(
                {
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": score,
                }
            )
        movie_scores = {}
        for chunk_score in chunks_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]
        results = []
        for movie_idx, score in sorted_movies:
            doc_id = self.documents[movie_idx]["id"]
            doc = self.document_map[doc_id]
            results.append(
                {
                    "id": doc_id,
                    "title": doc["title"],
                    "document": doc.get("description", "")[:100],
                    "score": round(score, 4),
                    "metadata": doc.get("metadata") or {},
                }
            )
        return results


def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")


def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search = SemanticSearch()
    documents = load_movies_data()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same shape for cosine similarity.")
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def semantic_chunking(text: str, max_chunk_size: int = 4, overlap: int = 1) -> list:
    if overlap > max_chunk_size:
        raise ValueError("Overlap cannot be greater than max chunk size.")
    text = text.strip()
    if len(text) == 0:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        return sentences
    i = 0
    chunks = []
    while i < len(sentences):
        chunk = sentences[i : i + max_chunk_size]
        if len(chunks) > 0 and len(chunk) <= overlap:
            break
        chunks.append(" ".join(chunk))
        i += max_chunk_size - overlap

    return chunks
