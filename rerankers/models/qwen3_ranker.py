"""
Qwen3Ranker implementation for Qwen3-Reranker models.

Supports Qwen3-Reranker-0.6B, 4B, and 8B models from https://huggingface.co/Qwen

These models use a causal LM architecture with yes/no token scoring for reranking.
"""

from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rerankers.documents import Document
from rerankers.models.ranker import BaseRanker
from rerankers.results import RankedResults, Result
from rerankers.utils import get_device, get_dtype, prep_docs, vprint

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

SYSTEM_PROMPT = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."


class Qwen3Ranker(BaseRanker):
    """
    Reranker using Qwen3-Reranker models.

    These models judge document relevance by comparing logits of "yes" vs "no" tokens.
    Supports instruction-aware reranking for customized retrieval scenarios.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Reranker-0.6B",
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 16,
        max_length: int = 8192,
        verbose: int = 1,
        instruction: Optional[str] = None,
        use_flash_attention: bool = False,
        **kwargs,
    ):
        """
        Initialize the Qwen3 Reranker.

        Args:
            model_name_or_path: HuggingFace model name or local path.
            dtype: Data type for model weights (e.g., "float16", "bfloat16").
            device: Device to use ("cuda", "cpu", "mps").
            batch_size: Batch size for inference.
            max_length: Maximum sequence length.
            verbose: Verbosity level (0=silent, 1=normal).
            instruction: Custom instruction for reranking task. If None, uses default.
            use_flash_attention: Whether to use flash attention 2 for faster inference.
            **kwargs: Additional arguments (model_kwargs, tokenizer_kwargs).
        """
        self.verbose = verbose
        self.device = get_device(device, verbose=self.verbose)
        self.dtype = get_dtype(dtype, self.device, self.verbose)
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction = instruction if instruction is not None else DEFAULT_INSTRUCTION
        self.ranking_type = "pointwise"

        vprint(
            f"Loading model {model_name_or_path}, this might take a while...",
            self.verbose,
        )
        vprint(f"Using device {self.device}.", self.verbose)
        vprint(f"Using dtype {self.dtype}.", self.verbose)

        # Load tokenizer
        tokenizer_kwargs = kwargs.get("tokenizer_kwargs", {})
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            **tokenizer_kwargs,
        )

        # Load model
        model_kwargs = kwargs.get("model_kwargs", {})
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.dtype,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()

        # Get token IDs for yes/no
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        vprint(f"True token ID (yes): {self.token_true_id}", self.verbose)
        vprint(f"False token ID (no): {self.token_false_id}", self.verbose)

        # Build prefix and suffix tokens
        self.prefix = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        vprint(f"Loaded Qwen3Ranker with model {model_name_or_path}", self.verbose)

    def _format_instruction(self, query: str, doc: str, instruction: Optional[str] = None) -> str:
        """Format the input with instruction, query, and document."""
        if instruction is None:
            instruction = self.instruction
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_inputs(self, pairs: List[str]) -> dict:
        """Tokenize and prepare inputs for the model."""
        # Tokenize without prefix/suffix, with truncation
        max_content_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=max_content_length,
            add_special_tokens=False,
        )

        # Add prefix and suffix tokens
        for i, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + input_ids + self.suffix_tokens

        # Pad to same length
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        return inputs

    @torch.inference_mode()
    def _compute_scores(self, inputs: dict) -> List[float]:
        """Compute relevance scores from model outputs."""
        outputs = self.model(**inputs)
        # Get logits for the last token position
        batch_scores = outputs.logits[:, -1, :]
        
        # Extract yes/no logits
        true_logits = batch_scores[:, self.token_true_id]
        false_logits = batch_scores[:, self.token_false_id]
        
        # Stack and apply log softmax
        stacked_scores = torch.stack([false_logits, true_logits], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked_scores, dim=1)
        
        # Return probability of "yes" (index 1)
        scores = log_probs[:, 1].exp().tolist()
        return scores

    @torch.inference_mode()
    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
        batch_size: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> RankedResults:
        """
        Rank documents by relevance to the query.

        Args:
            query: The search query.
            docs: Documents to rank (strings, list of strings, or Document objects).
            doc_ids: Optional document IDs.
            metadata: Optional metadata for each document.
            batch_size: Override default batch size.
            instruction: Override default instruction for this ranking call.

        Returns:
            RankedResults containing scored and ranked documents.
        """
        docs = prep_docs(docs, doc_ids, metadata)
        
        # Format all query-document pairs
        pairs = [self._format_instruction(query, doc.text, instruction) for doc in docs]

        # Use provided batch_size or default
        if batch_size is None:
            batch_size = self.batch_size

        # Process in batches
        batched_pairs = [
            pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)
        ]
        
        all_scores = []
        for batch in batched_pairs:
            inputs = self._process_inputs(batch)
            batch_scores = self._compute_scores(inputs)
            all_scores.extend(batch_scores)

        # Create ranked results
        ranked_results = [
            Result(document=doc, score=score, rank=idx + 1)
            for idx, (doc, score) in enumerate(
                sorted(zip(docs, all_scores), key=lambda x: x[1], reverse=True)
            )
        ]

        return RankedResults(results=ranked_results, query=query, has_scores=True)

    @torch.inference_mode()
    def score(self, query: str, doc: str, instruction: Optional[str] = None) -> float:
        """
        Score a single document's relevance to a query.

        Args:
            query: The search query.
            doc: The document text.
            instruction: Optional custom instruction.

        Returns:
            Relevance score (probability of "yes").
        """
        pair = self._format_instruction(query, doc, instruction)
        inputs = self._process_inputs([pair])
        scores = self._compute_scores(inputs)
        return scores[0]
