import argparse
import os

from dotenv import load_dotenv
from lib.hybrid_search import HybridSearch
from lib.utils import load_movies_data
from lib.augmented_generation import (
    rag,
    summarize,
    citations,
    question_answering
)


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize a long text document using the LLM"
    )
    summarize_parser.add_argument("query", type=str, help="Text to summarize")
    summarize_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of search results to use for generation"
    )
    
    citations_parser = subparsers.add_parser(
        "citations", help="Generate a summary with citations to the retrieved documents"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG with citations")
    citations_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of search results to use for generation"
    )
    
    question_parser = subparsers.add_parser(
        "question", help="Answer a question using RAG with retrieved documents"
    )
    question_parser.add_argument("question", type=str, help="Question to answer using RAG")
    question_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of search results to use for answering the question"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            documents = load_movies_data()
            search = HybridSearch(documents)
            results = search.rrf_search(query, limit=5)
            rag(query, results, api_key)
            
        case "summarize":
            query = args.query
            limit = args.limit
            documents = load_movies_data()
            search = HybridSearch(documents)
            results = search.rrf_search(query, limit=limit)
            summarize(query, results, api_key)
        case "citations":
            query = args.query
            limit = args.limit
            documents = load_movies_data()
            search = HybridSearch(documents)
            results = search.rrf_search(query, limit=limit)
            citations(query, results, api_key)
        case "question":
            question = args.question
            limit = args.limit
            documents = load_movies_data()
            search = HybridSearch(documents)
            results = search.rrf_search(question, limit=limit)
            question_answering(question, results, api_key)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()