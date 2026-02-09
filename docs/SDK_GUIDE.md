# Scout-KV SDK Documentation

> Complete guide to using Scout-KV for infinite context retrieval.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Search Strategies](#search-strategies)
6. [Training Custom Models](#training-custom-models)
7. [Advanced Usage](#advanced-usage)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)

---

## Installation

### From PyPI (Recommended)

```bash
pip install scoutkv
```

### From Source

```bash
git clone https://github.com/your-org/scout-kv.git
cd scout-kv
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory for model loading

---

## Quick Start

### Basic Usage

```python
from scoutkv import ScoutKV

# Initialize with pre-trained model
client = ScoutKV(model_path="scout_bge.pth")

# Index a document
client.index_file("books/sherlock_holmes.txt")

# Query the document
results = client.query("What was the speckled band?")

# Get the relevant context
context = client.get_context(results, max_blocks=5)
print(context)
```

### Understanding Results

```python
results = client.query("Who is Dr. Watson?")

print(f"Strategy used: {results.strategy_used}")
print(f"Latency: {results.latency_ms:.2f} ms")
print(f"Block IDs: {results.block_ids}")
print(f"Scores: {results.scores}")
print(f"Metadata: {results.metadata}")
```

---

## Core Concepts

### What is Scout-KV?

Scout-KV is a learned retrieval system that predicts which parts of a document an LLM will actually pay attention to. Unlike standard RAG (which uses cosine similarity), Scout-KV uses a trained neural network to score relevance.

### Key Components

| Component | Purpose |
|-----------|---------|
| **Indexer** | Chunks documents and generates embeddings |
| **ScoutBGE** | Neural network that scores block relevance |
| **Scanner** | Implements search strategies |
| **ScoutKV** | High-level client API |

### How It Works

```
Document → Chunk → Embed (BGE) → Score (Scout) → Top-K Blocks → LLM
```

1. **Chunking**: Document is split into ~500 character blocks
2. **Embedding**: Each block is encoded using BGE (384 dimensions)
3. **Scoring**: Scout MLP predicts relevance: `[query, block] → score`
4. **Selection**: Top-K blocks are selected as context

---

## API Reference

### ScoutKV Client

The main interface for using Scout-KV.

```python
from scoutkv import ScoutKV

client = ScoutKV(
    model_path="scout_bge.pth",  # Path to trained model
    device="cuda",               # "cuda" or "cpu"
    chunk_size=500,              # Characters per block
    strategy="adaptive"          # Default search strategy
)
```

#### Methods

##### `index_text(text, chunk_size=None) -> Index`

Index a text string.

```python
index = client.index_text("""
    Long document content here...
""")
print(f"Created {index.num_blocks} blocks")
```

##### `index_file(path, chunk_size=None) -> Index`

Index a text file.

```python
index = client.index_file("document.txt")
```

##### `save_index(path)`

Save the current index to disk.

```python
client.save_index("my_index")
# Creates: my_index.embeddings.pt, my_index.meta.npz
```

##### `load_index(path) -> Index`

Load an index from disk.

```python
index = client.load_index("my_index")
```

##### `query(question, k=5, strategy=None) -> SearchResult`

Search the indexed document.

```python
results = client.query(
    "What is the main theme?",
    k=5,                    # Minimum results
    strategy="adaptive"     # Override default strategy
)
```

##### `get_block(block_id) -> Block`

Retrieve a specific block.

```python
block = client.get_block(42)
print(block.text)
print(f"Characters {block.start_char}-{block.end_char}")
```

##### `get_context(result, max_blocks=None) -> str`

Get concatenated text from search results.

```python
context = client.get_context(results, max_blocks=5)
# Returns: "Block 1 text\n\nBlock 2 text\n\n..."
```

---

### SearchResult

Returned by `query()`.

```python
@dataclass
class SearchResult:
    block_ids: List[int]      # Selected block indices
    scores: List[float]       # Relevance scores (0-1)
    strategy_used: str        # Strategy that was used
    latency_ms: float         # Search time in milliseconds
    metadata: dict            # Strategy-specific data
```

#### Metadata Fields

**Adaptive strategy:**
```python
{
    "entropy": 7.05,          # Score distribution entropy
    "max_confidence": 0.45,   # Highest score
    "mass_coverage": 0.80,    # Cumulative probability covered
    "action": "Adaptive Selection"  # or "RAG Fallback"
}
```

**Hierarchical strategy:**
```python
{
    "chapters_scanned": 50,
    "blocks_scanned": 500,
    "top_chapters": [12, 34, 56, 78, 90]
}
```

---

### Indexer

Low-level indexing API.

```python
from scoutkv import Indexer, Index

indexer = Indexer(device="cuda", batch_size=32)

# Create index from text
blocks = indexer.chunk_text(text, chunk_size=500)
embeddings = indexer.encode_blocks(blocks)
index = Index(blocks=blocks, embeddings=embeddings)

# Encode a query
query_emb = indexer.encode_query("What happened?")

# Save/load index
indexer.save_index(index, "my_index")
loaded = indexer.load_index("my_index")
```

---

### ScoutBGE Model

The neural network component.

```python
from scoutkv import ScoutBGE

# Load pre-trained model
model = ScoutBGE.load("scout_bge.pth", device="cuda")

# Create new model
model = ScoutBGE(
    input_dim=768,    # 384 (query) + 384 (block)
    hidden_dim=128,   # Hidden layer size
    dropout=0.3       # Dropout rate
)

# Save model
model.save("my_model.pth")

# Get parameter count
print(f"Parameters: {model.num_parameters:,}")
```

---

### Scanner

Search engine with multiple strategies.

```python
from scoutkv import Scanner, ScoutBGE

model = ScoutBGE.load("scout_bge.pth")
scanner = Scanner(
    model,
    device="cuda",
    confidence_threshold=0.15,  # Fallback trigger
    mass_target=0.80,           # Adaptive K target
    max_blocks=20,              # Maximum selection
    chapter_size=100            # Hierarchical grouping
)

# Direct search
result = scanner.search(query_emb, index, k=5, strategy="adaptive")
```

---

## Search Strategies

### Linear (Default for small docs)

Scores all blocks, returns top-K.

```python
results = client.query("...", strategy="linear")
```

**Best for:** Documents < 1,000 blocks  
**Latency:** O(N) where N = number of blocks

### Hierarchical (For massive docs)

Two-stage search: chapters first, then blocks.

```python
results = client.query("...", strategy="hierarchical")
```

**How it works:**
1. Group blocks into chapters (100 blocks each)
2. Score all chapters, select top-5
3. Score blocks within selected chapters
4. Return top-K blocks

**Best for:** Documents > 10,000 blocks  
**Latency:** O(N/100 + 500) ≈ 15x faster than linear

### Adaptive (Recommended)

Entropy-aware with dynamic K selection.

```python
results = client.query("...", strategy="adaptive")
```

**How it works:**
1. Score all blocks
2. Calculate entropy of score distribution
3. If max_score < threshold → fallback to cosine similarity
4. Otherwise, accumulate blocks until 80% mass coverage
5. Return selected blocks (min K, max 20)

**Best for:** Unknown query quality, production use  
**Features:** Handles garbage queries, adjusts output size

---

## Training Custom Models

### Preparing Training Data

Training data should be a list of dictionaries:

```python
training_data = [
    {
        "query_emb": torch.tensor([...]),  # [384]
        "block_emb": torch.tensor([...]),  # [384]
        "label": 1,  # 1 = relevant, 0 = not relevant
        "question": "Original question text",
        "block_id": 0
    },
    ...
]
```

### Training

```python
from scoutkv import ScoutTrainer

trainer = ScoutTrainer(
    input_dim=768,
    hidden_dim=128,
    dropout=0.3,
    device="cuda"
)

model, best_recall = trainer.train(
    training_data,
    epochs=20,
    batch_size=32,
    lr=1e-4,
    val_split=0.2,
    save_path="my_scout.pth"
)

print(f"Best Recall@5: {best_recall*100:.1f}%")
```

### Generating Oracle Data

To create training data, you need "oracle" labels from an LLM:

```python
# 1. Run queries through LLM with full context
# 2. Extract attention patterns
# 3. Identify which blocks received highest attention
# 4. Use these as positive labels (label=1)
```

See `generate_oracle_data.py` for a complete example.

---

## Advanced Usage

### Using with LLMs

```python
from scoutkv import ScoutKV
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Scout-KV
client = ScoutKV(model_path="scout_bge.pth")
client.index_file("document.txt")

# Get relevant context
results = client.query("What is the conclusion?")
context = client.get_context(results, max_blocks=5)

# Generate answer with LLM
model = AutoModelForCausalLM.from_pretrained("...")
tokenizer = AutoTokenizer.from_pretrained("...")

prompt = f"""Context:
{context}

Question: What is the conclusion?
Answer:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
answer = tokenizer.decode(outputs[0])
```

### Batch Queries

```python
questions = [
    "Who is the protagonist?",
    "What is the setting?",
    "How does it end?"
]

for q in questions:
    results = client.query(q)
    print(f"Q: {q}")
    print(f"A: {client.get_context(results, max_blocks=2)[:200]}...")
    print()
```

### Pre-computing Indexes

For production, pre-compute and save indexes:

```python
# Offline: Index once
client = ScoutKV(model_path="scout_bge.pth")
client.index_file("corpus/book1.txt")
client.save_index("indexes/book1")

# Online: Load and query
client = ScoutKV(model_path="scout_bge.pth")
client.load_index("indexes/book1")
results = client.query("...")  # Fast!
```

---

## Performance Tuning

### Chunk Size

Smaller chunks = more precise, more blocks to search.

```python
# Fine-grained (slower, more precise)
client.index_file("doc.txt", chunk_size=200)

# Coarse-grained (faster, less precise)
client.index_file("doc.txt", chunk_size=1000)
```

### Batch Size

Increase for faster encoding on GPU:

```python
indexer = Indexer(device="cuda", batch_size=64)
```

### Device Selection

```python
# Force CPU (for systems without GPU)
client = ScoutKV(model_path="scout.pth", device="cpu")

# Auto-detect
from scoutkv import get_device
device = str(get_device())  # Returns "cuda" if available
```

### Index Caching

Always save indexes for documents you'll query multiple times:

```python
if Path("my_index.embeddings.pt").exists():
    client.load_index("my_index")
else:
    client.index_file("document.txt")
    client.save_index("my_index")
```

---

## Troubleshooting

### Common Errors

**`ModuleNotFoundError: No module named 'scoutkv'`**

Install the package:
```bash
pip install -e .
```

**`RuntimeError: CUDA out of memory`**

Reduce batch size:
```python
indexer = Indexer(device="cuda", batch_size=16)
```

**`ValueError: No index loaded`**

Call `index_file()` or `load_index()` before `query()`:
```python
client.index_file("document.txt")  # Don't forget this!
results = client.query("...")
```

**Poor retrieval quality**

1. Check if model was trained on similar data
2. Try different chunk sizes
3. Use `strategy="linear"` for comparison
4. Lower `confidence_threshold` in Scanner

### Debugging

Enable verbose logging:

```python
import logging
from scoutkv import setup_logging

setup_logging(level=logging.DEBUG)
```

### Getting Help

- Check `examples/` for working code
- Review benchmark scripts for patterns
- Examine `metadata` in SearchResult for diagnostics

---

## Appendix: Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         ScoutKV                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Indexer  │───▶│  Index   │───▶│ Scanner  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│       │              │                │                     │
│       ▼              ▼                ▼                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │   BGE    │    │ Blocks + │    │ ScoutBGE │              │
│  │ Encoder  │    │ Embeddings    │   MLP    │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

*Scout-KV: Infinite context on consumer hardware.*
