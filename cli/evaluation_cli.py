import argparse
from pathlib import Path
import json
from lib.hybrid_search import (
    HybridSearch
    )


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open(Path("./data/golden_dataset.json"), "r") as f:
        golden_dataset = json.load(f)
    test_cases = golden_dataset["test_cases"]
    data_path = Path("./data/movies.json")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    documents = data["movies"]
    search = HybridSearch(documents)
    
    print(f"Evaluating RRF search with k={limit} on {len(test_cases)} test cases...")
    for case in test_cases:
        query = case["query"]
        docs = case["relevant_docs"]
        results = search.rrf_search(query, limit=limit, k=60)
        relevant_docs = 0
        retrived_titles = [result["title"] for result in results]
        for result in results:
            if result["title"] in docs:
                relevant_docs += 1
        precision = relevant_docs / limit if limit > 0 else 0
        recall = relevant_docs / len(docs) if len(docs) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        output = f"\n- Query: {query}\n"
        output += f"  - Precision@{limit}: {precision:.4f}\n"
        output += f"  - Recall@{limit}: {recall:.4f}\n"
        output += f"  - F1 Score: {f1:.4f}\n"
        output += f"  - Retrieved: {', '.join(retrived_titles)}\n"
        output += f"  - Relevant: {', '.join(docs)}"
        print(output, flush=True)


if __name__ == "__main__":
    main()