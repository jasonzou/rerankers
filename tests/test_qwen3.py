"""Tests for Qwen3Ranker."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from rerankers.documents import Document
from rerankers.results import RankedResults, Result

# Skip all tests in this module if torch is not installed
torch_available = "torch" in sys.modules or pytest.importorskip("torch", reason="torch not installed")


@pytest.fixture
def mock_qwen3_ranker():
    """Create a Qwen3Ranker with mocked dependencies."""
    with patch("rerankers.models.qwen3_ranker.AutoModelForCausalLM") as mock_model, \
         patch("rerankers.models.qwen3_ranker.AutoTokenizer") as mock_tokenizer:
        
        # Setup tokenizer mock
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.convert_tokens_to_ids.side_effect = lambda x: 1 if x == "yes" else 0
        mock_tokenizer_instance.encode.return_value = [1, 2, 3]
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

        # Setup model mock
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        from rerankers.models.qwen3_ranker import Qwen3Ranker
        ranker = Qwen3Ranker(
            model_name_or_path="Qwen/Qwen3-Reranker-0.6B",
            verbose=0,
            device="cpu",
        )
        yield ranker, mock_tokenizer, mock_model


def test_qwen3_ranker_initialization(mock_qwen3_ranker):
    """Test Qwen3Ranker can be initialized."""
    ranker, mock_tokenizer, mock_model = mock_qwen3_ranker

    assert ranker.token_true_id == 1
    assert ranker.token_false_id == 0
    assert ranker.ranking_type == "pointwise"
    mock_tokenizer.from_pretrained.assert_called_once()
    mock_model.from_pretrained.assert_called_once()


def test_qwen3_ranker_rank(mock_qwen3_ranker):
    """Test Qwen3Ranker.rank() returns correct results structure."""
    ranker, _, _ = mock_qwen3_ranker
    
    with patch.object(ranker, '_process_inputs') as mock_process, \
         patch.object(ranker, '_compute_scores') as mock_compute:
        
        mock_process.return_value = {"input_ids": MagicMock()}
        mock_compute.return_value = [0.9, 0.3]

        query = "What is the capital of China?"
        docs = ["The capital of China is Beijing.", "Paris is in France."]

        results = ranker.rank(query=query, docs=docs)

        assert isinstance(results, RankedResults)
        assert results.has_scores is True
        assert results.query == query
        assert len(results.results) == 2
        # Higher score should be ranked first
        assert results.results[0].score == 0.9
        assert results.results[0].rank == 1
        assert results.results[1].score == 0.3
        assert results.results[1].rank == 2


def test_qwen3_ranker_score(mock_qwen3_ranker):
    """Test Qwen3Ranker.score() returns a float score."""
    ranker, _, _ = mock_qwen3_ranker

    with patch.object(ranker, '_process_inputs') as mock_process, \
         patch.object(ranker, '_compute_scores') as mock_compute:
        
        mock_process.return_value = {"input_ids": MagicMock()}
        mock_compute.return_value = [0.85]

        score = ranker.score(
            query="What is the capital of China?",
            doc="The capital of China is Beijing."
        )

        assert isinstance(score, float)
        assert score == 0.85


def test_qwen3_ranker_format_instruction(mock_qwen3_ranker):
    """Test that _format_instruction produces correct format."""
    ranker, _, _ = mock_qwen3_ranker
    
    query = "test query"
    doc = "test document"
    
    result = ranker._format_instruction(query, doc)
    
    assert "<Instruct>:" in result
    assert "<Query>: test query" in result
    assert "<Document>: test document" in result


def test_qwen3_ranker_custom_instruction(mock_qwen3_ranker):
    """Test that custom instruction is used when provided."""
    ranker, _, _ = mock_qwen3_ranker
    
    custom_instruction = "Find relevant code snippets"
    query = "how to sort a list"
    doc = "Use sorted() function"
    
    result = ranker._format_instruction(query, doc, instruction=custom_instruction)
    
    assert f"<Instruct>: {custom_instruction}" in result


def test_result_with_qwen3_document():
    """Test Result objects work correctly with Documents."""
    doc = Document(doc_id=1, text="The capital of China is Beijing.")
    result = Result(document=doc, score=0.95, rank=1)

    assert result.doc_id == 1
    assert result.text == "The capital of China is Beijing."
    assert result.score == 0.95
    assert result.rank == 1


def test_ranked_results_with_qwen3():
    """Test RankedResults works correctly with Qwen3-style results."""
    results = RankedResults(
        results=[
            Result(
                document=Document(doc_id=0, text="The capital of China is Beijing."),
                score=0.95,
                rank=1,
            ),
            Result(
                document=Document(doc_id=1, text="Paris is in France."),
                score=0.3,
                rank=2,
            ),
        ],
        query="What is the capital of China?",
        has_scores=True,
    )

    assert results.results_count() == 2
    top_1 = results.top_k(1)
    assert len(top_1) == 1
    assert top_1[0].score == 0.95
    assert results.get_score_by_docid(0) == 0.95
    assert results.get_score_by_docid(1) == 0.3
