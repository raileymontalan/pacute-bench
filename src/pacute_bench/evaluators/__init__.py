"""
pacute_bench.evaluators — evaluator classes for all supported backends.

Hierarchy:
    BaseEvaluator          (abstract: metrics, answer extraction, result persistence)
    ├── VLLMEvaluator      (MCQ log-prob scoring + async generation via vLLM server)
    └── BatchEvaluator     (abstract: shared submit-poll-collect lifecycle)
        ├── OpenAIEvaluator    (OpenAI Batch API)
        ├── AnthropicEvaluator (Anthropic Message Batches API)
        └── GeminiEvaluator    (Google Gemini Batch API)
"""

from __future__ import annotations

from typing import Optional

from .base import BaseEvaluator, BENCHMARK_FORMATS
from .vllm import VLLMEvaluator
from .batch import BatchEvaluator
from .openai import OpenAIEvaluator
from .anthropic import AnthropicEvaluator
from .gemini import GeminiEvaluator

__all__ = [
    "BaseEvaluator",
    "BatchEvaluator",
    "VLLMEvaluator",
    "OpenAIEvaluator",
    "AnthropicEvaluator",
    "GeminiEvaluator",
    "BENCHMARK_FORMATS",
    "make_evaluator",
]

# ---------------------------------------------------------------------------
# Provider registry — add new providers here only
# ---------------------------------------------------------------------------

_BATCH_PROVIDERS: dict = {
    "openai":    OpenAIEvaluator,
    "anthropic": AnthropicEvaluator,
    "gemini":    GeminiEvaluator,
}


def make_evaluator(provider: Optional[str], **kwargs) -> BaseEvaluator:
    """
    Factory that returns the right evaluator for a given provider string.

    Args:
        provider: ``"openai"``, ``"anthropic"``, ``"gemini"``, or ``None``
            (``None`` → :class:`VLLMEvaluator`).
        **kwargs: Passed through to the evaluator constructor.

    Returns:
        A concrete :class:`BaseEvaluator` instance.

    Raises:
        ValueError: If ``provider`` is an unrecognised string.
    """
    if provider is None:
        return VLLMEvaluator(**kwargs)

    key = provider.lower()
    cls = _BATCH_PROVIDERS.get(key)
    if cls is None:
        supported = ", ".join(f'"{p}"' for p in _BATCH_PROVIDERS)
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {supported}, or None for vLLM."
        )
    return cls(**kwargs)
