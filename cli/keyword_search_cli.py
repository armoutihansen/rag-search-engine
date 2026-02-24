#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
import string
from nltk.stem import PorterStemmer
from lib.keyword_search import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

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
            
            
            # print the search query here
            # print(f"Searching for: {args.query}")
            # query_movies = []

            # translator = str.maketrans("", "", string.punctuation)
            # query_tokens = args.query.lower().translate(translator).split()
            # query_tokens = [stemmer.stem(token) for token in query_tokens if len(token) > 0 and token not in stopwords]
            
            # for movie in movies:
            #     movie_tokens = movie["title"].lower().translate(translator).split()
            #     movie_tokens = [stemmer.stem(token) for token in movie_tokens if len(token) > 0 and token not in stopwords]
            #     if any(token in movie_token for movie_token in movie_tokens for token in query_tokens):
            #         query_movies.append(movie)
            # query_movies_sorted = sorted(query_movies, key=lambda x: x["id"])
            
            # for i, movie in enumerate(query_movies_sorted, start=1):
            #     if i > 5:
            #         break
            #     print(f"{i}. {movie['title']}")
                
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()