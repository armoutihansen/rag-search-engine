import argparse
import os

from dotenv import load_dotenv
from lib.hybrid_search import (
    HybridSearch,
    cross_encode_matches,
    expand_query,
    rewrite_query,
    spell_correct_query,
    rate_matches_with_query,
    rate_matches_with_query_batch,
    evaluate_rrf,
)
from lib.utils import load_movies_data


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help="")
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="Text to normalize"
    )
    weighted_search_parser = subparsers.add_parser(
        "weighted-search",
        help="Search movies using a weighted combination of BM25 and semantic search scores",
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        default=0.5,
        help="Weight for BM25 score (0.0 to 1.0)",
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of results to return"
    )
    rrf_search_parser = subparsers.add_parser(
        "rrf-search",
        help="Search movies using Reciprocal Rank Fusion (RRF) to combine BM25 and semantic search results",
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument(
        "--k", type=int, nargs="?", default=60, help="RRF k parameter"
    )
    rrf_search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of results to return"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Optional enhancement to apply to the query before searching",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Method to use for reranking the results",
    )
    rrf_search_parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging for search pipeline"
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation of RRF using and LLM-based evaluation",
    )

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
                normalized_scores = [
                    (score - min_score) / (max_score - min_score) for score in scores
                ]
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            documents = load_movies_data()
            search = HybridSearch(documents)
            results = search.weighted_search(
                args.query, alpha=args.alpha, limit=args.limit
            )
            print(
                f"Weighted search results for query: '{args.query}' with alpha={args.alpha}"
            )
            for i, result in enumerate(results, start=1):
                print(
                    f"{i}. {result['title']}\n Hybrid Score: {result['score']:.4f})\n BM25: {result['metadata']['bm25_score']:.4f}, Semantic: {result['metadata']['semantic_score']:.4f}\n  {result['document'][:100]}"
                )

        case "rrf-search":
            documents = load_movies_data()
            search = HybridSearch(documents, debug=args.debug)
            query = args.query
            limit = args.limit

            if args.enhance:
                if args.enhance == "spell":
                    enhanced_query = spell_correct_query(query, api_key)
                elif args.enhance == "rewrite":
                    enhanced_query = rewrite_query(query, api_key)
                elif args.enhance == "expand":
                    enhanced_query = expand_query(query, api_key)
                query = enhanced_query

            if args.rerank_method:
                if not api_key and not args.rerank_method == "cross_encoder":
                    print("GEMINI_API_KEY is required for this reranking method.")
                    return
                limit = limit * 5

            results = search.rrf_search(query, k=args.k, limit=limit)

            if args.rerank_method:
                if args.rerank_method == "individual":
                    results = rate_matches_with_query(query, results, api_key)
                    if args.debug:
                        import logging as log_module

                        logger_obj = log_module.getLogger(__name__)
                        logger_obj.info("Reranking complete (individual method)")
                        for i, result in enumerate(results[: args.limit], 1):
                            logger_obj.info(
                                f"  {i}. {result['title']} (score: {result.get('match_score', 0.0):.1f})"
                            )
                elif args.rerank_method == "batch":
                    results = rate_matches_with_query_batch(query, results, api_key)
                    if args.debug:
                        import logging as log_module

                        logger_obj = log_module.getLogger(__name__)
                        logger_obj.info("Reranking complete (batch method)")
                        for i, result in enumerate(results[: args.limit], 1):
                            logger_obj.info(
                                f"  {i}. {result['title']} (rank: {int(result.get('match_score', 999))})"
                            )
                elif args.rerank_method == "cross_encoder":
                    results = cross_encode_matches(query, results)

            print(f"Results for query: '{query}' with k={args.k}")
            for i, result in enumerate(results, start=1):
                if i > args.limit:
                    break
                if args.rerank_method == "individual":
                    print(
                        f"{i}. {result['title']}\n Rerank Score: {result.get('match_score', 0.0):.1f}/10\n RRF Score: {result['score']:.4f})\n BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}\n {result['document'][:100]}"
                    )
                elif args.rerank_method == "batch":
                    print(
                        f"{i}. {result['title']}\n Rerank Rank: {int(result.get('match_score', 999))}\n RRF Score: {result['score']:.4f})\n BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}\n {result['document'][:100]}"
                    )
                elif args.rerank_method == "cross_encoder":
                    print(
                        f"{i}. {result['title']}\n Cross-Encoder Score: {result.get('match_score', 0.0):.4f}\n RRF Score: {result['score']:.4f})\n BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}\n {result['document'][:100]}"
                    )
                else:
                    print(
                        f"{i}. {result['title']}\n RRF Score: {result['score']:.4f})\n BM25 Rank: {result['metadata']['bm25_rank']}, Semantic Rank: {result['metadata']['semantic_rank']}\n {result['document'][:100]}"
                    )
            if args.evaluate:
                results = evaluate_rrf(query, results, api_key)
                print("\nLLM Evaluation of results:")
                for i, result in enumerate(results, start=1):
                    if i > args.limit:
                        break
                    print(
                        f"{i}. {result['title']}: {result.get('evaluation_score', 0)}/3"
                    )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
