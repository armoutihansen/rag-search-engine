import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

# Expansion factor for initial search results before re-ranking
SEARCH_EXPANSION_FACTOR = 500


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build(documents)
            self.idx.save()
        else:
            self.idx.load()

    def _bm25_search(self, query, limit):
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        """Perform hybrid search using weighted combination of BM25 and semantic scores.
        
        Args:
            query: Search query text
            alpha: Weight for BM25 score (0-1), semantic weight is (1-alpha)
            limit: Number of final results to return
            
        Returns:
            List of result dictionaries with scores and metadata
        """
        # Get expanded results from both search methods
        bm25_results = self._bm25_search(query, limit * SEARCH_EXPANSION_FACTOR)
        semantic_results = self.semantic_search.search_chunks(query, limit * SEARCH_EXPANSION_FACTOR)
        
        # Normalize scores using list comprehensions
        bm25_scores = [result["score"] for result in bm25_results]
        normalized_bm25_scores = normalize_scores(bm25_scores)
        for i, result in enumerate(bm25_results):
            result["score"] = normalized_bm25_scores[i]
            
        semantic_scores = [result["score"] for result in semantic_results]
        normalized_semantic_scores = normalize_scores(semantic_scores)
        for i, result in enumerate(semantic_results):
            result["score"] = normalized_semantic_scores[i]
        # Combine scores from both methods
        combined_scores = {}
        for result in bm25_results:
            doc_id = result["id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                    "title": result["title"],
                    "document": result["document"],
                }
            combined_scores[doc_id]["bm25_score"] = max(combined_scores[doc_id]["bm25_score"], result["score"])
            
        for result in semantic_results:
            doc_id = result["id"]
            if doc_id in combined_scores:
                combined_scores[doc_id]["semantic_score"] = max(result["score"], combined_scores[doc_id]["semantic_score"])
                if combined_scores[doc_id].get("document") == "":
                    combined_scores[doc_id]["document"] = result["document"]
            else:
                combined_scores[doc_id] = {
                    "bm25_score": 0.0,
                    "semantic_score": result["score"],
                    "title": result["title"],
                    "document": result["document"]
                }
        
        # Calculate combined scores
        for data in combined_scores.values():
            data["combined_score"] = alpha * data["bm25_score"] + (1 - alpha) * data["semantic_score"]
            
        # Sort and return top results
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x]["combined_score"], reverse=True)
        return [
            {
                "id": doc_id,
                "title": combined_scores[doc_id]["title"],
                "document": combined_scores[doc_id]["document"],
                "score": round(combined_scores[doc_id]["combined_score"], 3),
                "metadata": {
                    "bm25_score": round(combined_scores[doc_id]["bm25_score"], 3),
                    "semantic_score": round(combined_scores[doc_id]["semantic_score"], 3),
                },
            }
            for doc_id in sorted_ids[:limit]
        ]

    def rrf_search(self, query, k=60, limit=10):
        """Perform hybrid search using Reciprocal Rank Fusion.
        
        Args:
            query: Search query text
            k: RRF constant (typically 60)
            limit: Number of final results to return
            
        Returns:
            List of result dictionaries with ranks and RRF scores
        """
        bm25_results = self._bm25_search(query, limit * SEARCH_EXPANSION_FACTOR)
        semantic_results = self.semantic_search.search_chunks(query, limit * SEARCH_EXPANSION_FACTOR)
        
        combined_scores = {}
        
        # Process BM25 results - already sorted by relevance (best first)
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "bm25_rank": rank,
                    "semantic_rank": None,
                    "title": result["title"],
                    "document": result.get("description", ""),
                }
            else:
                # Take the best (smallest) rank if duplicate
                combined_scores[doc_id]["bm25_rank"] = min(combined_scores[doc_id]["bm25_rank"], rank)
        
        # Process semantic results - already sorted by relevance (best first)
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "title": result["title"],
                    "document": result["document"],
                }
            else:
                # Take the best (smallest) rank if duplicate
                if combined_scores[doc_id]["semantic_rank"] is None:
                    combined_scores[doc_id]["semantic_rank"] = rank
                else:
                    combined_scores[doc_id]["semantic_rank"] = min(combined_scores[doc_id]["semantic_rank"], rank)
                # Update document if it was empty
                if combined_scores[doc_id].get("document") == "":
                    combined_scores[doc_id]["document"] = result["document"]
        
        # Calculate RRF scores from ranks
        for doc_id, data in combined_scores.items():
            bm25_rank = data["bm25_rank"]
            semantic_rank = data["semantic_rank"]
            
            bm25_rrf = 1 / (k + bm25_rank) if bm25_rank is not None else 0
            semantic_rrf = 1 / (k + semantic_rank) if semantic_rank is not None else 0
            
            data["rrf_score"] = bm25_rrf + semantic_rrf
        
        # Sort by RRF score and return top results
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x]["rrf_score"], reverse=True)
        return [
            {
                "id": doc_id,
                "title": combined_scores[doc_id]["title"],
                "document": combined_scores[doc_id]["document"],
                "score": round(combined_scores[doc_id]["rrf_score"], 3),
                "metadata": {
                    "bm25_rank": combined_scores[doc_id]["bm25_rank"],
                    "semantic_rank": combined_scores[doc_id]["semantic_rank"],
                },
            }
            for doc_id in sorted_ids[:limit]
        ]
        
        
    
def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    else:
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
