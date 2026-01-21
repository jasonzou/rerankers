# AGENTS.md - Coding Agent Instructions for rerankers

## Project Overview

`rerankers` is a lightweight unified Python API for various document re-ranking models. It supports multiple reranking backends including cross-encoders, T5, ColBERT, RankGPT, FlashRank, and various API providers (Cohere, Jina, Voyage, Pinecone, etc.).

**Key Design Principles:**
- Zero core dependencies - optional extras install only what's needed
- Unified interface via `Reranker()` factory function
- All rankers inherit from `BaseRanker` and implement `rank()` method
- Results returned as `RankedResults` containing `Result` objects with `Document`s

---

## Build / Lint / Test Commands

### Installation
```bash
# Core package (no dependencies)
pip install -e .

# Development dependencies (ruff, isort, pytest)
pip install -e ".[dev]"

# All optional dependencies
pip install -e ".[all]"

# Specific model backends
pip install -e ".[transformers]"  # Cross-encoders, T5, ColBERT
pip install -e ".[api]"           # Cohere, Jina, etc.
pip install -e ".[gpt]"           # RankGPT (litellm)
pip install -e ".[flashrank]"     # FlashRank ONNX models
```

### Running Tests
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_results.py

# Run a specific test function
pytest tests/test_results.py::test_ranked_results_functions

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "result"
```

### Linting
```bash
# Check with ruff
ruff check .

# Auto-fix with ruff
ruff check --fix .

# Sort imports with isort
isort .

# Check import sorting only
isort --check-only .
```

---

## Code Style Guidelines

### Imports

**Order (enforce with isort):**
1. Standard library imports
2. Third-party imports (torch, transformers, etc.)
3. Local imports (rerankers.*)

**Pattern for optional dependencies:**
```python
# Wrap optional imports in try/except at module level
try:
    import torch
    from transformers import AutoModelForSequenceClassification
except ImportError:
    pass
```

**Local imports - use relative when inside package:**
```python
from rerankers.models.ranker import BaseRanker
from rerankers.results import RankedResults, Result
from rerankers.documents import Document
from rerankers.utils import vprint, prep_docs
```

### Formatting

- **Line length:** ~100 characters (flexible, not strictly enforced)
- **Indentation:** 4 spaces
- **Quotes:** Double quotes for strings
- **Trailing commas:** Use in multi-line structures

### Type Hints

**Always use type hints for function signatures:**
```python
def rank(
    self,
    query: str,
    docs: Union[str, List[str], Document, List[Document]],
    doc_ids: Optional[Union[List[str], List[int]]] = None,
    metadata: Optional[List[dict]] = None,
) -> RankedResults:
```

**Common type patterns in this codebase:**
```python
from typing import List, Optional, Union, Literal, Tuple, Iterable

# Document inputs accept multiple formats
docs: Union[str, List[str], Document, List[Document]]

# IDs can be strings or ints
doc_ids: Optional[Union[List[str], List[int]]] = None

# Device/dtype flexibility
device: Optional[Union[str, torch.device]] = None
dtype: Optional[Union[str, torch.dtype]] = None
```

### Naming Conventions

- **Classes:** PascalCase (`TransformerRanker`, `RankedResults`, `Document`)
- **Functions/methods:** snake_case (`prep_docs`, `get_device`, `rank_async`)
- **Constants:** UPPER_SNAKE_CASE (`AVAILABLE_RANKERS`, `DEFAULTS`, `URLS`)
- **Private methods:** Single underscore prefix (`_get_score`, `_parse_response`)

### Class Structure

**Ranker classes must inherit from `BaseRanker`:**
```python
class MyNewRanker(BaseRanker):
    def __init__(
        self,
        model_name_or_path: str,
        verbose: int = 1,
        **kwargs,
    ):
        self.verbose = verbose
        self.ranking_type = "pointwise"  # or "listwise"
        # ... initialization
    
    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> RankedResults:
        docs = prep_docs(docs, doc_ids, metadata)
        # ... ranking logic
        return RankedResults(results=ranked_results, query=query, has_scores=True)
    
    def score(self, query: str, doc: str) -> float:
        # Single document scoring
        pass
```

### Error Handling

**Use ValueError for validation:**
```python
if rank is None and score is None:
    raise ValueError("Either score or rank must be provided.")
```

**Print warnings/info, don't raise for recoverable issues:**
```python
print("Warning: Model type could not be auto-mapped. Defaulting to TransformerRanker.")
```

**Handle optional dependencies gracefully:**
```python
try:
    from rerankers.integrations.langchain import RerankerLangChainCompressor
    return RerankerLangChainCompressor(model=self, k=k)
except ImportError:
    print("You need to install langchain to use this feature!")
```

### Verbose Output

**Use `vprint` utility for conditional output:**
```python
from rerankers.utils import vprint

vprint(f"Loaded model {model_name}", self.verbose)
vprint(f"Using device {self.device}.", self.verbose)
```

### Docstrings

- Use docstrings for public methods and complex functions
- Keep inline with existing minimal style (not overly verbose)

```python
def top_k(self, k: int) -> List[Result]:
    """Returns the top k results based on the score, if available, or rank."""
```

---

## Architecture Notes

### Core Classes
- `Reranker()` - Factory function in `rerankers/reranker.py`
- `BaseRanker` - Abstract base in `rerankers/models/ranker.py`
- `Document` - Data class in `rerankers/documents.py`
- `Result`, `RankedResults` - Result classes in `rerankers/results.py`

### Adding a New Ranker
1. Create `rerankers/models/my_ranker.py` inheriting from `BaseRanker`
2. Add try/except import in `rerankers/models/__init__.py`
3. Add model type mapping in `rerankers/reranker.py` (`_get_model_type`)
4. Add default models to `DEFAULTS` dict if applicable
5. Add dependency mapping to `DEPS_MAPPING`

### Key Utilities
- `prep_docs()` - Normalizes doc input formats to `List[Document]`
- `get_device()` / `get_dtype()` - Auto-detect CUDA/MPS/CPU
- `vprint()` - Verbose conditional printing

---

## Testing Patterns

```python
from unittest.mock import patch
import pytest
from rerankers import Reranker
from rerankers.results import Result, RankedResults
from rerankers.documents import Document

# Mock expensive model loading
@patch("rerankers.models.transformer_ranker.TransformerRanker.rank")
def test_transformer_ranker_rank(mock_rank):
    expected_results = RankedResults(
        results=[Result(document=Document(doc_id=1, text="..."), score=1.0, rank=1)],
        query="test",
        has_scores=True,
    )
    mock_rank.return_value = expected_results
    # ... test logic

# Test validation errors
def test_result_validation_error():
    with pytest.raises(ValueError) as excinfo:
        Result(document=Document(doc_id=2, text="Doc 2"))
    assert "Either score or rank must be provided." in str(excinfo.value)
```

---

## Python Version

- Minimum: Python 3.8
- Supported: 3.8, 3.9, 3.10, 3.11, 3.12
- Note: Some features (RankLLM) require Python 3.10+
