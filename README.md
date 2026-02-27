# RAG Search Engine

A hybrid search engine combining keyword-based (BM25) and semantic search for movie data. Implements multiple search strategies including weighted combination and Reciprocal Rank Fusion (RRF), with AI-powered query enhancements and reranking capabilities.

## Features

- **Keyword Search**: BM25 algorithm with stemming and stopword removal
- **Semantic Search**: Sentence transformer embeddings with chunked document support
- **Hybrid Search**: Combines both approaches using:
  - Weighted score combination
  - Reciprocal Rank Fusion (RRF)
- **AI-Powered Query Enhancement** (via Gemini API):
  - Spell correction
  - Query rewriting for better search results
  - Query expansion with related terms
- **AI-Powered Reranking** (via Gemini API or Cross-Encoder):
  - Individual document scoring (0-10 scale)
  - Batch ranking (more efficient)
  - Cross-encoder reranking
- **Debug Pipeline**: Track query transformations and results through each stage
- **Evaluation Framework**: Test search quality with precision, recall, and F1 metrics
- **Efficient Caching**: Embeddings and indexes are cached for fast retrieval

## Installation

```bash
# Create virtual environment and install dependencies
uv sync
```

## Environment Setup

For AI-powered features (query enhancement and reranking), create a `.env` file:

```bash
# .env
GEMINI_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. (Optional) Set up API key for AI features
echo "GEMINI_API_KEY=your_api_key" > .env

# 3. Basic search
uv run cli/hybrid_search_cli.py rrf-search "family comedy" --limit 5

# 4. With query enhancement
uv run cli/hybrid_search_cli.py rrf-search "family comedy" --enhance expand --limit 5

# 5. With reranking
uv run cli/hybrid_search_cli.py rrf-search "family comedy" --rerank-method batch --limit 5

# 6. Full pipeline with debug logging
uv run cli/hybrid_search_cli.py rrf-search "family comedy" \
  --enhance expand \
  --rerank-method batch \
  --limit 3 \
  --debug
```

## Project Structure

```
rag-search-engine/
├── cli/
│   ├── lib/
│   │   ├── keyword_search.py    # BM25 inverted index
│   │   ├── semantic_search.py   # Embedding-based search
│   │   ├── hybrid_search.py     # Combined search strategies
│   │   └── constants.py         # BM25 parameters
│   ├── keyword_search_cli.py    # Keyword search CLI
│   ├── semantic_search_cli.py   # Semantic search CLI
│   ├── hybrid_search_cli.py     # Hybrid search CLI
│   └── evaluation_cli.py        # Search evaluation CLI
├── data/
│   ├── movies.json              # Movie dataset
│   ├── stopwords.txt            # Stopwords list
│   └── golden_dataset.json      # Test cases for evaluation
├── cache/                       # Generated indexes and embeddings
├── .env                         # API keys (not in repo)
└── pyproject.toml              # Project configuration
```

## Usage

### Keyword Search (BM25)

**Build the inverted index:**
```bash
uv run cli/keyword_search_cli.py build
```

**Search for movies:**
```bash
uv run cli/keyword_search_cli.py bm25search "animated adventure" --limit 5
```

**Get term statistics:**
```bash
# Term frequency
uv run cli/keyword_search_cli.py tf 123 "adventure"

# Inverse document frequency
uv run cli/keyword_search_cli.py idf "adventure"

# TF-IDF score
uv run cli/keyword_search_cli.py tfidf 123 "adventure"

# BM25 IDF
uv run cli/keyword_search_cli.py bm25idf "adventure"

# BM25 TF
uv run cli/keyword_search_cli.py bm25tf 123 "adventure" --k1 1.5 --b 0.75
```

### Semantic Search

**Verify model:**
```bash
uv run cli/semantic_search_cli.py verify
```

**Build embeddings:**
```bash
uv run cli/semantic_search_cli.py verify_embeddings
```

**Search using semantic similarity:**
```bash
uv run cli/semantic_search_cli.py search "family friendly animation" --limit 5
```

**Embed text:**
```bash
uv run cli/semantic_search_cli.py embed_text "hero's journey"
```

### Hybrid Search

**Weighted search** (combines normalized BM25 and semantic scores):
```bash
# Alpha = 0.5 means equal weight to both methods
uv run cli/hybrid_search_cli.py weighted-search "action adventure" --alpha 0.5 --limit 10

# Alpha = 0.8 prioritizes keyword matching
uv run cli/hybrid_search_cli.py weighted-search "specific term" --alpha 0.8 --limit 10

# Alpha = 0.2 prioritizes semantic similarity
uv run cli/hybrid_search_cli.py weighted-search "conceptual query" --alpha 0.2 --limit 10
```

**RRF search** (Reciprocal Rank Fusion):
```bash
# Basic RRF search
uv run cli/hybrid_search_cli.py rrf-search "british comedy" --limit 10 --k 60

# With debug logging to see pipeline stages
uv run cli/hybrid_search_cli.py rrf-search "family movie" --limit 5 --debug
```

**Query Enhancement** (requires GEMINI_API_KEY):
```bash
# Spell correction
uv run cli/hybrid_search_cli.py rrf-search "famly movi" --enhance spell --limit 5

# Query rewriting for better results
uv run cli/hybrid_search_cli.py rrf-search "scary bear" --enhance rewrite --limit 5

# Query expansion with related terms
uv run cli/hybrid_search_cli.py rrf-search "bear adventure" --enhance expand --limit 5
```

**Reranking Results** (requires GEMINI_API_KEY for individual/batch):
```bash
# Individual scoring (0-10 for each document)
uv run cli/hybrid_search_cli.py rrf-search "family comedy" --rerank-method individual --limit 5

# Batch ranking (faster, ranks all at once)
uv run cli/hybrid_search_cli.py rrf-search "action movie" --rerank-method batch --limit 5

# Cross-encoder reranking (no API key needed)
uv run cli/hybrid_search_cli.py rrf-search "romantic comedy" --rerank-method cross_encoder --limit 5
```

**Combined Pipeline** (enhancement + reranking):
```bash
# Expand query, then rerank with batch method
uv run cli/hybrid_search_cli.py rrf-search "bear" \
  --enhance expand \
  --rerank-method batch \
  --limit 3 \
  --debug
```

### Search Evaluation

Evaluate search quality using precision, recall, and F1 metrics:

```bash
# Run evaluation with k=5 (Precision@5, Recall@5)
uv run cli/evaluation_cli.py --limit 5

# Run with different k values
uv run cli/evaluation_cli.py --limit 10
```

Output format:
```
- Query: dangerous bear wilderness survival
  - Precision@5: 0.8000
  - Recall@5: 0.4000
  - F1 Score: 0.5333
  - Retrieved: The Edge, Man in the Wilderness, Claws, Unnatural, Into the Grizzly Maze
  - Relevant: Unnatural, Alaska, The Edge, Into the Grizzly Maze, Claws, Man in the Wilderness, The Revenant
```

## Search Strategies

### BM25 (Best Matching 25)
- Probabilistic ranking function for keyword search
- Accounts for term frequency, document length, and inverse document frequency
- Parameters: `k1` (term saturation, default: 1.5), `b` (length normalization, default: 0.75)

### Semantic Search
- Uses sentence transformers (`all-MiniLM-L6-v2`) to encode text into embeddings
- Measures cosine similarity between query and document embeddings
- Supports chunked search for long documents

### Weighted Hybrid
- Normalizes both BM25 and semantic scores to [0, 1]
- Combines using: `score = alpha * bm25_score + (1 - alpha) * semantic_score`
- Alpha parameter controls the balance (0 = pure semantic, 1 = pure BM25)

### RRF (Reciprocal Rank Fusion)
- Combines rankings from multiple search methods
- Score formula: `1 / (k + rank)` where k is typically 60
- Robust to score scale differences
- Effective when different methods find complementary results

## AI-Powered Features

### Query Enhancement

**Spell Correction** (`--enhance spell`)
- Fixes typos and spelling errors
- Example: "famly movi" → "family movie"

**Query Rewriting** (`--enhance rewrite`)
- Rewrites queries to be more specific and searchable
- Considers movie knowledge, genres, and conventions
- Example: "scary bear movie" → "bear horror thriller movie terrifying"

**Query Expansion** (`--enhance expand`)
- Adds related terms and synonyms
- Expands semantic coverage
- Example: "family bear" → "family bear parents children kids siblings heartwarming cub grizzly polar"

### Reranking Methods

**Individual Reranking** (`--rerank-method individual`)
- Uses Gemini AI to score each document individually (0-10 scale)
- Provides relevance score for each match
- More thorough but slower (1 API call per document)

**Batch Reranking** (`--rerank-method batch`)
- Sends all documents to Gemini AI at once for ranking
- Returns documents sorted by relevance (rank 1, 2, 3, ...)
- More efficient (1 API call total)
- Recommended for production use

**Cross-Encoder Reranking** (`--rerank-method cross_encoder`)
- Uses a cross-encoder model for relevance scoring
- No API key required (runs locally)
- Good balance of speed and accuracy

### Debug Logging

Enable with `--debug` flag to see the search pipeline:

1. **Original Query**: The input query
2. **Enhanced Query**: Query after enhancement (if using `--enhance`)
3. **RRF Results**: Top results after RRF fusion with scores
4. **Reranked Results**: Final results after reranking (if using `--rerank-method`)

Example output:
```
[INFO] Original query: 'family bear'
[INFO] [ENHANCE] Expand: 'family bear' -> 'family bear parents children kids...'
[INFO] RRF search returned 10 results:
[INFO]   1. Alaska (RRF score: 0.033)
[INFO]   2. The Bear (RRF score: 0.029)
[INFO] Reranking complete (batch method)
[INFO]   1. The Berenstain Bears' Christmas Tree (rank: 1)
[INFO]   2. The Little Polar Bear (rank: 2)
```

## Configuration

### Environment Variables
Add to `.env` file:
```bash
GEMINI_API_KEY=your_api_key_here  # Required for query enhancement and AI reranking
```

### BM25 Parameters
Edit `cli/lib/constants.py`:
```python
BM25_K1 = 1.5  # Term saturation parameter
BM25_B = 0.75  # Length normalization parameter
```

### Search Expansion Factor
Edit `cli/lib/hybrid_search.py`:
```python
SEARCH_EXPANSION_FACTOR = 500  # Number of candidates to retrieve before re-ranking
```

### Logging
The system automatically suppresses verbose library logs and only shows application-level logs when using `--debug`.

## API Examples

### Using InvertedIndex directly

```python
from cli.lib.keyword_search import InvertedIndex

# Load or build index
idx = InvertedIndex()
idx.load()

# Search
results = idx.bm25_search("adventure movie", limit=5)
for result in results:
    print(f"{result['title']}: {result['score']}")
```

### Using HybridSearch

```python
from cli.lib.hybrid_search import HybridSearch
import json

# Load documents
with open("data/movies.json") as f:
    documents = json.load(f)["movies"]

# Initialize (loads/builds indexes automatically)
search = HybridSearch(documents)

# Weighted search
results = search.weighted_search("family movie", alpha=0.5, limit=5)

# RRF search
results = search.rrf_search("british comedy", k=60, limit=5)

# RRF search with debug logging
search_debug = HybridSearch(documents, debug=True)
results = search_debug.rrf_search("action movie", k=60, limit=5)

# Access results
for result in results:
    print(f"{result['title']}")
    print(f"  Score: {result['score']}")
    print(f"  Metadata: {result['metadata']}")
```

### Using Query Enhancement

```python
from cli.lib.hybrid_search import spell_correct_query, rewrite_query, expand_query
import os

api_key = os.environ.get("GEMINI_API_KEY")

# Spell correction
corrected = spell_correct_query("famly movi", api_key)
print(f"Corrected: {corrected}")

# Query rewriting
rewritten = rewrite_query("scary bear movie", api_key)
print(f"Rewritten: {rewritten}")

# Query expansion
expanded = expand_query("family adventure", api_key)
print(f"Expanded: {expanded}")
```

### Using Reranking

```python
from cli.lib.hybrid_search import rate_matches_with_query, rate_matches_with_query_batch
import os

api_key = os.environ.get("GEMINI_API_KEY")
query = "family comedy movie"

# Get initial results
search = HybridSearch(documents)
results = search.rrf_search(query, k=60, limit=20)

# Individual reranking (scores each document 0-10)
reranked = rate_matches_with_query(query, results, api_key)
for result in reranked[:5]:
    print(f"{result['title']}: {result['match_score']}/10")

# Batch reranking (ranks all documents at once)
reranked = rate_matches_with_query_batch(query, results, api_key)
for result in reranked[:5]:
    print(f"{result['title']}: Rank {result['match_score']}")
```

## Performance Considerations

- **First run**: Builds indexes and embeddings (slower)
- **Subsequent runs**: Loads from cache (fast)
- **Cache location**: `cache/` directory
- **Clear cache**: Delete `cache/` folder to rebuild

### API Call Optimization

**Query Enhancement**:
- Each enhancement method makes 1 API call
- Use `--enhance` only when query quality is critical

**Reranking Methods**:
- `individual`: 1 API call per document (slower but thorough)
- `batch`: 1 API call total (faster, recommended)
- `cross_encoder`: No API calls (runs locally)

**Best Practices**:
- Use `batch` reranking for production
- Limit initial results (e.g., `--limit 5`) to reduce API costs
- Cache enhanced queries when possible
- Use `cross_encoder` for API-free reranking

## Evaluation Metrics

The evaluation framework measures search quality using:

- **Precision@k**: Proportion of retrieved documents that are relevant
  - Formula: `relevant_retrieved / k`
  - High precision = fewer false positives

- **Recall@k**: Proportion of relevant documents that are retrieved
  - Formula: `relevant_retrieved / total_relevant`
  - High recall = fewer false negatives

- **F1 Score**: Harmonic mean of precision and recall
  - Formula: `2 * (precision * recall) / (precision + recall)`
  - Balances precision and recall

Test cases are defined in `data/golden_dataset.json`.

## Data Format

### Movie Data
Movies should be in `data/movies.json`:
```json
{
  "movies": [
    {
      "id": 1,
      "title": "Movie Title",
      "description": "Movie description..."
    }
  ]
}
```

### Golden Dataset (for evaluation)
Test cases in `data/golden_dataset.json`:
```json
{
  "test_cases": [
    {
      "query": "dangerous bear wilderness survival",
      "relevant_docs": [
        "The Edge",
        "Into the Grizzly Maze",
        "Alaska",
        "Claws"
      ]
    }
  ]
}
```

Each test case includes:
- `query`: The search query to test
- `relevant_docs`: List of movie titles that should be retrieved for this query

## Common Use Cases

### Use Case 1: Basic Movie Search
```bash
# Best for: General queries, no API cost
uv run cli/hybrid_search_cli.py rrf-search "action adventure" --limit 10
```

### Use Case 2: Misspelled or Vague Queries
```bash
# Best for: User-generated queries with typos
uv run cli/hybrid_search_cli.py rrf-search "famly annimation" --enhance spell --limit 5
```

### Use Case 3: High-Quality Results
```bash
# Best for: When you need the most relevant results
uv run cli/hybrid_search_cli.py rrf-search "family comedy" \
  --enhance expand \
  --rerank-method batch \
  --limit 5
```

### Use Case 4: Development and Debugging
```bash
# Best for: Understanding the search pipeline
uv run cli/hybrid_search_cli.py rrf-search "bear movie" \
  --enhance rewrite \
  --rerank-method batch \
  --debug \
  --limit 3
```

### Use Case 5: Local-Only (No API)
```bash
# Best for: When you don't have an API key
uv run cli/hybrid_search_cli.py rrf-search "romantic comedy" \
  --rerank-method cross_encoder \
  --limit 5
```

## Troubleshooting

### "GEMINI_API_KEY is required"
- Create a `.env` file with your API key
- Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Or use `cross_encoder` reranking which doesn't need an API key

### Slow First Run
- First run builds indexes and downloads models
- Subsequent runs load from `cache/` directory
- Expected: 10-30 seconds for first run

### Library Logs Cluttering Output
- Use `--debug` flag only when needed
- Library logs (HTTP requests, model loading) are automatically suppressed in debug mode
- Only application-level logs are shown

### Rate Limiting
- Individual reranking makes 1 API call per document
- Use `batch` reranking (1 API call total) for better performance
- Add delays between requests if needed

### Cache Issues
- Delete `cache/` directory to rebuild from scratch
- Useful when updating document data or models
```bash
rm -rf cache/
```

## License

(Add your license here)

## Contributing

(Add contribution guidelines here)
