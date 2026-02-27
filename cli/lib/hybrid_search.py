import os
import time
import json
import logging

from .constants import GEMINI_MODEL
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from sentence_transformers import CrossEncoder

# Expansion factor for initial search results before re-ranking
SEARCH_EXPANSION_FACTOR = 500

# Configure logging - suppress library logs, only show application logs
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from libraries
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class HybridSearch:
    def __init__(self, documents, debug=False):
        self.documents = documents
        self.debug = debug
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
        semantic_results = self.semantic_search.search_chunks(
            query, limit * SEARCH_EXPANSION_FACTOR
        )

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
            combined_scores[doc_id]["bm25_score"] = max(
                combined_scores[doc_id]["bm25_score"], result["score"]
            )

        for result in semantic_results:
            doc_id = result["id"]
            if doc_id in combined_scores:
                combined_scores[doc_id]["semantic_score"] = max(
                    result["score"], combined_scores[doc_id]["semantic_score"]
                )
                if combined_scores[doc_id].get("document") == "":
                    combined_scores[doc_id]["document"] = result["document"]
            else:
                combined_scores[doc_id] = {
                    "bm25_score": 0.0,
                    "semantic_score": result["score"],
                    "title": result["title"],
                    "document": result["document"],
                }

        # Calculate combined scores
        for data in combined_scores.values():
            data["combined_score"] = (
                alpha * data["bm25_score"] + (1 - alpha) * data["semantic_score"]
            )

        # Sort and return top results
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x]["combined_score"],
            reverse=True,
        )
        return [
            {
                "id": doc_id,
                "title": combined_scores[doc_id]["title"],
                "document": combined_scores[doc_id]["document"],
                "score": round(combined_scores[doc_id]["combined_score"], 3),
                "metadata": {
                    "bm25_score": round(combined_scores[doc_id]["bm25_score"], 3),
                    "semantic_score": round(
                        combined_scores[doc_id]["semantic_score"], 3
                    ),
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
        if self.debug:
            logger.info(f"Original query: '{query}'")
        
        bm25_results = self._bm25_search(query, limit * SEARCH_EXPANSION_FACTOR)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * SEARCH_EXPANSION_FACTOR
        )

        combined_scores = {}

        # Process BM25 results - already sorted by relevance (best first)
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result["id"]
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    "bm25_rank": rank,
                    "semantic_rank": None,
                    "title": result["title"],
                    "document": result.get("document", ""),
                }
            else:
                # Take the best (smallest) rank if duplicate
                combined_scores[doc_id]["bm25_rank"] = min(
                    combined_scores[doc_id]["bm25_rank"], rank
                )

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
                    combined_scores[doc_id]["semantic_rank"] = min(
                        combined_scores[doc_id]["semantic_rank"], rank
                    )
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
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x]["rrf_score"],
            reverse=True,
        )
        
        final_results = [
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
        
        if self.debug:
            logger.info(f"RRF search returned {len(final_results)} results:")
            for i, result in enumerate(final_results, 1):
                logger.info(f"  {i}. {result['title']} (RRF score: {result['score']})")
        
        return final_results


def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    else:
        return [(score - min_score) / (max_score - min_score) for score in scores]


def spell_correct_query(query: str, api_key: str) -> str:
    """Fix spelling errors in a search query using Gemini AI.

    Args:
        query: The original search query
        api_key: Google GenAI API key

    Returns:
        Corrected query string
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    prompt = f"""Fix any spelling errors in this movie search query.
Only correct obvious typos. Don't change correctly spelled words.
Query: "{query}"
If no errors, return the original query.
Corrected:"""  

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    corrected = response.text.strip()
    if corrected != query:
        logger.info(f"[ENHANCE] Spell correct: '{query}' -> '{corrected}'")
    return corrected


def rewrite_query(query: str, api_key: str) -> str:
    """Rewrite a search query to be more specific and searchable using Gemini AI.

    Args:
        query: The original search query
        api_key: Google GenAI API key

    Returns:
        Rewritten query string
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    rewritten = response.text.strip()
    if rewritten != query:
        logger.info(f"[ENHANCE] Rewrite: '{query}' -> '{rewritten}'")
    return rewritten


def expand_query(query: str, api_key: str) -> str:
    """Expand a search query with related terms using Gemini AI.

    Args:
        query: The original search query
        api_key: Google GenAI API key

    Returns:
        Expanded query string
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    expanded = response.text.strip()
    if expanded != query:
        logger.info(f"[ENHANCE] Expand: '{query}' -> '{expanded}'")
    return expanded

def rate_matches_with_query(query: str, documents: list, api_key: str) -> list:
    """Use Gemini AI to rate how well each document matches the search query.

    Args:
        query: The search query
        documents: List of document dictionaries with 'title' and 'description'
        api_key: Google GenAI API key

    Returns:
        List of documents with added 'match_score' (0-1) indicating relevance to the query
    """
    from google import genai

    client = genai.Client(api_key=api_key)
    
    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        try:
            score = float(response.text.strip())
            doc["match_score"] = round(score, 3)
        except ValueError:
            doc["match_score"] = 0.0  # If parsing fails, assign lowest score

        time.sleep(1)  # Small delay to avoid hitting rate limits

    # Sort documents by match_score in descending order
    documents.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return documents

def rate_matches_with_query_batch(query: str, documents: list, api_key: str) -> list:
    """Use Gemini AI to rate how well each document matches the search query in batch.

    Args:
        query: The search query
        documents: List of document dictionaries with 'title' and 'description'
        api_key: Google GenAI API key
    Returns:
        List of documents with added 'match_score' (0-1) indicating relevance to the query
    """
    from google import genai
    
    doc_list_str = "\n".join(
        [f"ID {doc.get('id')}: {doc.get('title', '')} - {doc.get('document', '')}" for doc in documents]
    )

    client = genai.Client(api_key=api_key)

    prompt =f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
""" 

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    
    try:
        # Strip markdown code fences if present
        response_text = response.text.strip()
        if response_text.startswith("```"):
            # Remove markdown code fences
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])  # Remove first and last lines
        
        ranked_ids = json.loads(response_text.strip())
        id_to_score = {doc["id"]: doc for doc in documents}
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in id_to_score:
                id_to_score[doc_id]["match_score"] = rank  # Store actual rank (1, 2, 3...)
            else:
                logger.warning(f"Warning: ID {doc_id} from Gemini response not found in documents.")
        
        # Assign worst rank (999) to any documents that weren't ranked
        for doc in documents:
            if "match_score" not in doc:
                doc["match_score"] = 999
    except json.JSONDecodeError:
        logger.error("Error parsing Gemini response for batch ranking. Response was:")
        logger.error(response.text)
        for doc in documents:
            doc["match_score"] = 999  # Assign worst rank if parsing fails
    
    # Sort documents by match_score in ascending order (lower rank = better)
    documents.sort(key=lambda x: x.get("match_score", 999))
    return documents

def cross_encode_matches(query: str, documents: list) -> list:
    """Use a CrossEncoder model to score relevance of documents to the query.

    Args:
        query: The search query
        documents: List of document dictionaries with 'title' and 'description'
        api_key: Google GenAI API key

    Returns:
        List of documents with added 'match_score' (0-1) indicating relevance to the query
    """
    model_name = "cross-encoder/ms-marco-TinyBERT-L2-v2"
    cross_encoder = CrossEncoder(model_name)

    pairs = [(query, f"{doc.get('title', '')} - {doc.get('document', '')}") for doc in documents]
    scores = cross_encoder.predict(pairs)

    for i, doc in enumerate(documents):
        doc["match_score"] = round(scores[i], 4)

    # Sort documents by match_score in descending order
    documents.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return documents

def evaluate_rrf(query: str, results: list, api_key: str):
    """Evaluate RRF search results using Gemini AI.

    Args:
        query: The search query
        results: List of result dictionaries with 'title' and 'document'
        api_key: Google GenAI API key

    Prints evaluation metrics to console.
    """
    from google import genai

    client = genai.Client(api_key=api_key)
    
    formatted_results = [f"{i+1}. {result['title']} - {result['document'][:250]}" for i, result in enumerate(results)]

    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{"\n".join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    
    try:
        # Strip markdown code fences if present
        response_text = response.text.strip()
        if response_text.startswith("```"):
            # Remove markdown code fences
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])  # Remove first and last lines
        
        scores = json.loads(response_text.strip())
        for i, score in enumerate(scores):
            results[i]["evaluation_score"] = score
    except json.JSONDecodeError:
        logger.error("Error parsing Gemini response for evaluation. Response was:")
        logger.error(response.text)
        for result in results:
            result["evaluation_score"] = 0  # Assign worst score if parsing fails
            
    return results 