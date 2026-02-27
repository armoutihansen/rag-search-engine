# RAG Search Engine

A hybrid search engine combining keyword-based (BM25) and semantic search for movie data, with AI-powered query enhancement, reranking, multimodal search, and retrieval-augmented generation.

## Features

- **Keyword Search (BM25)**: Probabilistic ranking with stemming and stopword removal
- **Semantic Search**: Sentence transformer embeddings with chunked document support
- **Hybrid Search**: Combines keyword and semantic via weighted combination or RRF
- **Multimodal Search**: Image-based search using CLIP vision embeddings
- **Query Enhancement**: Spell correction, rewriting, expansion, image-to-query transformation
- **Reranking**: Individual scoring (0-10), batch ranking, or cross-encoder methods
- **RAG Pipeline**: Generate Q&A, summaries, and citations from retrieved results
- **Debug Mode**: Track query transformations through the search pipeline
- **Evaluation Framework**: Precision@k, Recall@k, F1 metrics with golden dataset
- **Caching**: Auto-caching of indexes and embeddings

## Installation

```bash
uv sync
```

## Quick Setup

```bash
# Create .env file for AI features (optional)
echo "GEMINI_API_KEY=your_key_here" > .env

# Basic search (no API needed)
uv run cli/hybrid_search_cli.py rrf-search "family comedy" --limit 5

# With query enhancement (requires API key)
uv run cli/hybrid_search_cli.py rrf-search "family comedy" --enhance expand --limit 5

# With reranking
uv run cli/hybrid_search_cli.py rrf-search "family comedy" --rerank-method batch --limit 5

# Debug mode - see pipeline stages
uv run cli/hybrid_search_cli.py rrf-search "family comedy" --enhance expand --debug --limit 3
```

## Project Structure

```
cli/
├── lib/
│   ├── keyword_search.py           # BM25 search
│   ├── semantic_search.py          # Embedding-based search
│   ├── hybrid_search.py            # Combined search + AI features
│   ├── multimodal_search.py        # Image-based search with CLIP
│   ├── augmented_generation.py     # RAG pipeline
│   ├── describe_image.py           # Image enhancement
│   ├── constants.py                # Configuration
│   └── utils.py                    # Utilities
├── *_cli.py                         # CLI interfaces
data/
├── movies.json                      # Movie dataset
├── stopwords.txt                    # Stopwords
└── golden_dataset.json              # Test cases
```

## Usage Examples

### Keyword Search
```bash
uv run cli/keyword_search_cli.py build
uv run cli/keyword_search_cli.py bm25search "adventure" --limit 5
```

### Semantic Search
```bash
uv run cli/semantic_search_cli.py search "family animation" --limit 5
```

### Multimodal Search
```bash
uv run cli/multimodal_search_cli.py image_search data/paddington.jpeg --limit 5
uv run cli/describe_image_cli.py --image data/paddington.jpeg --query "bear in London"
```

### Hybrid Search
```bash
# RRF (combines BM25 and semantic)
uv run cli/hybrid_search_cli.py rrf-search "action adventure" --limit 10

# With enhancement and reranking
uv run cli/hybrid_search_cli.py rrf-search "family comedy" \
  --enhance expand \
  --rerank-method batch \
  --limit 5
```

### Query Enhancement (requires GEMINI_API_KEY)
```bash
--enhance spell    # Fix typos
--enhance rewrite  # Improve readability
--enhance expand   # Add related terms
```

### Reranking Methods
```bash
--rerank-method individual   # AI scoring (0-10 per doc, 1 API call per doc)
--rerank-method batch        # AI ranking (1 API call total, recommended)
--rerank-method cross_encoder # Local model, no API key needed
```

### RAG (Retrieval-Augmented Generation)
```bash
uv run cli/augmented_generation_cli.py question "What dinosaur movies?" --limit 5
uv run cli/augmented_generation_cli.py summarize "action movies" --limit 5
uv run cli/augmented_generation_cli.py citations "family films" --limit 5
```

### Evaluation
```bash
uv run cli/evaluation_cli.py --limit 5
```

## Configuration

### Environment Variables
```bash
# .env
GEMINI_API_KEY=your_key_here  # Required for enhancement/reranking/RAG
```

### Constants (`cli/lib/constants.py`)
```python
BM25_K1 = 1.5          # Term saturation
BM25_B = 0.75          # Length normalization
SEARCH_EXPANSION_FACTOR = 500  # Candidates before reranking
```

## API Usage

```python
from cli.lib.hybrid_search import HybridSearch
from cli.lib.utils import load_movies_data

documents = load_movies_data()
search = HybridSearch(documents, debug=False)

# RRF search
results = search.rrf_search("bears", k=60, limit=5)

# With enhancement
from cli.lib.hybrid_search import expand_query
query = expand_query("bear", api_key)
results = search.rrf_search(query, limit=5)

# Individual reranking
from cli.lib.hybrid_search import rate_matches_with_query
reranked = rate_matches_with_query("bears", results, api_key)
```

## Performance

- **First run**: ~30s (builds indexes + downloads models)
- **Next runs**: ~1s (loads from `cache/`)
- **API costs**: 
  - Enhancement: 1 call per method
  - Batch reranking: 1 call per query
  - Individual reranking: 1 call per document

## Search Methods Explained

| Method | Best For | Speed | Accuracy | API |
|--------|----------|-------|----------|-----|
| BM25 | Exact keywords | Fast | Good | No |
| Semantic | Conceptual queries | Medium | Good | No |
| Weighted Hybrid | General purpose | Medium | Good | No |
| RRF | Multiple methods | Medium | Excellent | No |
| Batch Reranking | High quality | Medium | Excellent | Yes |
| Cross-Encoder | API-free ranking | Fast | Good | No |

## Troubleshooting

**"GEMINI_API_KEY is required"**
- Create `.env` file: `GEMINI_API_KEY=your_key`
- Or use `--rerank-method cross_encoder` (no API)

**Slow first run**
- Normal: models download and indexes build (~30s)
- Future runs load from cache

**Clear cache**
```bash
rm -rf cache/
```

**Library logs verbose**
- Already suppressed with `--debug`
- Only app logs shown

## API Reference

See `/cli/` files for CLI interfaces. Key functions in `cli/lib/`:

- `HybridSearch.rrf_search(query, k, limit)` - Reciprocal Rank Fusion
- `spell_correct_query(query, api_key)` - Fix spelling
- `rewrite_query(query, api_key)` - Improve query
- `expand_query(query, api_key)` - Add related terms
- `rate_matches_with_query_batch(query, docs, api_key)` - Rerank efficiently

More details in code docstrings and `README_FULL.md`.

## License

MIT
