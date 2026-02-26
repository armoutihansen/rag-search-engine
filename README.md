# RAG Search Engine

A hybrid search engine combining keyword-based (BM25) and semantic search for movie data. Implements multiple search strategies including weighted combination and Reciprocal Rank Fusion (RRF).

## Features

- **Keyword Search**: BM25 algorithm with stemming and stopword removal
- **Semantic Search**: Sentence transformer embeddings with chunked document support
- **Hybrid Search**: Combines both approaches using:
  - Weighted score combination
  - Reciprocal Rank Fusion (RRF)
- **Efficient Caching**: Embeddings and indexes are cached for fast retrieval

## Installation

```bash
# Create virtual environment and install dependencies
uv sync
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
│   └── hybrid_search_cli.py     # Hybrid search CLI
├── data/
│   ├── movies.json              # Movie dataset
│   └── stopwords.txt            # Stopwords list
├── cache/                       # Generated indexes and embeddings
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
uv run cli/hybrid_search_cli.py rrf-search "british comedy" --limit 10 --k 60
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

## Configuration

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

# Access results
for result in results:
    print(f"{result['title']}")
    print(f"  Score: {result['score']}")
    print(f"  Metadata: {result['metadata']}")
```

## Performance Considerations

- **First run**: Builds indexes and embeddings (slower)
- **Subsequent runs**: Loads from cache (fast)
- **Cache location**: `cache/` directory
- **Clear cache**: Delete `cache/` folder to rebuild

## Data Format

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

## License

(Add your license here)

## Contributing

(Add contribution guidelines here)
