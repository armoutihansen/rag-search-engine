import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
import numpy as np


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_score = self._bm25_search(query, limit*500)
        semantic_score = self.semantic_search.search_chunks(query, limit*500)
        bm25_scores = []
        for it in bm25_score:
            bm25_scores.append(it["score"])
        normalized_bm25_scores = normalize_scores(bm25_scores)
        for i, it in enumerate(bm25_score):
            it["score"] = normalized_bm25_scores[i]
        semantic_scores = []
        for it in semantic_score:
            semantic_scores.append(it["score"])
        normalized_semantic_scores = normalize_scores(semantic_scores)
        for i, it in enumerate(semantic_score):
            it["score"] = normalized_semantic_scores[i]
        combined_scores = {}
        for it in bm25_score:
            doc_id = it["id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                    "title": it["title"],
                    "document": it["document"],
                }
            combined_scores[doc_id]["bm25_score"] = max(combined_scores[doc_id]["bm25_score"], it["score"])
            # combined_scores[it["id"]] = {"bm25_score": it["score"], "semantic_score": 0.0, "title": it["title"], "document": it["document"]}
        for it in semantic_score:
            if it["id"] in combined_scores:
                combined_scores[it["id"]]["semantic_score"] = max(it["score"], combined_scores[it["id"]]["semantic_score"])
                if combined_scores[it["id"]].get("document") == "":
                    combined_scores[it["id"]]["document"] = it["document"]
            else:
                combined_scores[it["id"]] = {"bm25_score": 0.0, "semantic_score": it["score"], "title": it["title"], "document": it["document"]}
        for it in combined_scores.values():
            it["combined_score"] = alpha * it["bm25_score"] + (1 - alpha) * it["semantic_score"]
            
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x]["combined_score"], reverse=True)
        out = []
        for doc_id in sorted_ids[:limit]:
            it = combined_scores[doc_id]
            out.append({
                "id": doc_id,
                "title": it["title"],
                "document": it["document"],
                "score": round(it["combined_score"], 3),
                "metadata": {
                    "bm25_score": round(it["bm25_score"], 3),
                    "semantic_score": round(it["semantic_score"], 3),
                },
            })
        return out

    def rrf_search(self, query, k=60, limit=10):
        bm25_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)
        
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
        
        # Sort by RRF score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x]["rrf_score"], reverse=True)
        
        out = []
        for doc_id in sorted_ids[:limit]:
            data = combined_scores[doc_id]
            out.append({
                "id": doc_id,
                "title": data["title"],
                "document": data["document"],
                "score": round(data["rrf_score"], 3),
                "metadata": {
                    "bm25_rank": data["bm25_rank"],
                    "semantic_rank": data["semantic_rank"],
                },
            })
        
        return out
        
        
    
def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    else:
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
