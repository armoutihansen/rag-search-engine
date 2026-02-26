import argparse

from lib.hybrid_search import HybridSearch
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="")
    normalize_parser.add_argument("scores", nargs='+', type=float, help="Text to normalize")
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Search movies using a weighted combination of BM25 and semantic search scores")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, nargs="?", default=0.5, help="Weight for BM25 score (0.0 to 1.0)")
    weighted_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return")
    rrf_search_parser = subparsers.add_parser("rrf-search", help="Search movies using Reciprocal Rank Fusion (RRF) to combine BM25 and semantic search results")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, nargs="?", default=60, help="RRF k parameter")
    rrf_search_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return")
    

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            if len(scores) == 0:
                return
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                normalized_scores = [1.0 for _ in scores]
            else:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            data_path = Path("./data/movies.json")
            data = json.loads(data_path.read_text(encoding="utf-8"))
            documents = data["movies"]            
            search = HybridSearch(documents)
            results = search.weighted_search(args.query, alpha=args.alpha, limit=args.limit)
            print(f"Weighted search results for query: '{args.query}' with alpha={args.alpha}")
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}\n Hybrid Score: {result['score']:.4f})\n BM25: {result['metadata']['bm25_score']:.4f}, Semantic: {result['metadata']['semantic_score']:.4f}\n  {result['document'][:100]}")
        case "rrf-search":
            data_path = Path("./data/movies.json")
            data = json.loads(data_path.read_text(encoding="utf-8"))
            documents = data["movies"]            
            search = HybridSearch(documents)
            results = search.rrf_search(args.query, k=args.k, limit=args.limit)
            print(f"RRF search results for query: '{args.query}' with k={args.k}")
            for i, result in enumerate(results, start=1):
                print(f"{i}. {result['title']}\n RRF Score: {result['score']:.4f})\n  {result['document'][:100]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()