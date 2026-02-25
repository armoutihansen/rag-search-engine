#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
import string
from nltk.stem import PorterStemmer
from lib.keyword_search import InvertedIndex
from lib.constants import BM25_B, BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")
    
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")
    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a document and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("--k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("--b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter")
    
    bm25_parser = subparsers.add_parser("bm25search", help="Search movies using BM25")
    bm25_parser.add_argument("query", type=str, help="Search query")
    bm25_parser.add_argument("--limit", type=int, nargs="?", default=5, help="Number of results to return")
    bm25_parser.add_argument("--k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_parser.add_argument("--b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter")

    args = parser.parse_args()
    
    data_path = Path("./data/movies.json")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    movies = data["movies"]
    
    with open("./data/stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = list(f.read().splitlines())
        
    stemmer = PorterStemmer()

    match args.command:
        case "search":
            index = InvertedIndex()
            try: 
                index.load()
            except FileNotFoundError:
                print("Index files not found. Please run the 'build' command to create the index before searching.")
                sys.exit(1)
            
            translator = str.maketrans("", "", string.punctuation)
            query_tokens = args.query.lower().translate(translator).split()
            query_tokens = [stemmer.stem(token) for token in query_tokens if len(token) > 0 and token not in stopwords]
            
            limit = 5
            seen_ids = set()
            found_movies = []
            for token in query_tokens:
                doc_ids = index.get_documents(token)
                for doc_id in doc_ids:
                    if doc_id in seen_ids:
                        continue
                    if len(found_movies) >= limit:
                        break
                    found_movies.append(index.docmap[doc_id])
                    seen_ids.add(doc_id)
            
            for movie in found_movies:
                print(movie["title"], movie["id"])
            
        case "build":
            index = InvertedIndex()
            index.build(movies)
            index.save()
            
        case "tf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index files not found. Please run the 'build' command to create the index before searching.")
                sys.exit(1)
            doc_id = args.doc_id
            term = args.term
            try:
                tf = index.get_tf(doc_id, term)
                print(f"Term frequency of '{term}' in document {doc_id}: {tf}")
            except ValueError as e:
                print(0)
        case "idf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index files not found. Please run the 'build' command to create the index before searching.")
                sys.exit(1)
            term = args.term
            try:
                idf = index.get_idf(term)
                print(f"Inverse document frequency of '{term}': {idf: .2f}")
            except ValueError as e:
                print(0)
                
        case "tfidf":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index files not found. Please run the 'build' command to create the index before searching.")
                sys.exit(1)
            doc_id = args.doc_id
            term = args.term
            try:
                tfidf = index.get_tfidf(doc_id, term)
                print(f"TF-IDF score of '{term}' in document {doc_id}: {tfidf: .2f}")
            except ValueError as e:
                print(0)
        
        case "bm25idf":
            index = InvertedIndex()
            bm25_idf = index.bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf: .2f}")
            
        case "bm25tf":
            index = InvertedIndex()
            bm25tf = index.bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            
        case "bm25search":
            index = InvertedIndex()
            try:
                index.load()
            except FileNotFoundError:
                print("Index files not found. Please run the 'build' command to create the index before searching.")
                sys.exit(1)
            results = index.bm25_search(args.query, args.limit, args.k1, args.b)
            for movie, score in results:
                print(f"({movie['id']}) {movie['title']} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()