#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    embed_query_text,
    embed_text,
    semantic_chunking,
    verify_embeddings,
    verify_model,
)
from lib.utils import load_movies_data


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser(
        "verify", help="Verify that the semantic search model loads correctly"
    )
    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for input text"
    )
    embed_parser.add_argument("text", type=str, help="Text to generate embedding for")
    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings",
        help="Verify that embeddings can be generated and cached for the movie dataset",
    )
    embedquery_parser = subparsers.add_parser(
        "embedquery", help="Generate embedding for a search query"
    )
    embedquery_parser.add_argument(
        "query", type=str, help="Search query to generate embedding for"
    )
    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of results to return"
    )
    chunk_parser = subparsers.add_parser(
        "chunk", help="Test text chunking for long documents"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, nargs="?", default=200, help="Chunk size in tokens"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=50, help="Chunk overlap in tokens"
    )
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Test semantic chunking for long documents"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        nargs="?",
        default=4,
        help="Maximum chunk size in tokens",
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="Chunk overlap in tokens"
    )
    embed_chunks_parser = subparsers.add_parser(
        "embed_chunks", help="Test embedding generation for text chunks"
    )
    search_chunks_parser = subparsers.add_parser(
        "search_chunked", help="Test searching over chunked documents"
    )
    search_chunks_parser.add_argument("query", type=str, help="Search query")
    search_chunks_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search = SemanticSearch()
            documents = load_movies_data()
            search.load_or_create_embeddings(documents)
            results = search.search(args.query, limit=args.limit)
            print(f"Search results for query: '{args.query}'")
            for i, (doc_id, result) in enumerate(results.items(), start=1):
                print(
                    f"{i}. {result['title']} (score: {result['score']:.4f})\n  {result['description']}"
                )
        case "chunk":
            text_split = args.text.split()
            chunk_size = args.chunk_size
            overlap = args.overlap
            chunked_text = [
                text_split[max(i - overlap, 0) : i + chunk_size]
                for i in range(0, len(text_split), chunk_size)
            ]
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunked_text, start=1):
                print(f"{i}. {' '.join(chunk)}")
        case "semantic_chunk":
            chunked_text = semantic_chunking(
                args.text, max_chunk_size=args.max_chunk_size, overlap=args.overlap
            )
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunked_text, start=1):
                print(f"{i}. {chunk}")
        case "embed_chunks":
            search = ChunkedSemanticSearch()
            documents = load_movies_data()
            embeddings = search.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search_chunked":
            search = ChunkedSemanticSearch()
            documents = load_movies_data()
            search.load_or_create_chunk_embeddings(documents)
            results = search.search_chunks(args.query, limit=args.limit)
            for i, result in enumerate(results, start=1):
                print(
                    f"\n{i}. {result['title']} (score: {result['score']:.4f})\n  {result['document']}"
                )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
