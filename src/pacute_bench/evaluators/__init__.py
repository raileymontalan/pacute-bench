"""
pacute_bench.evaluators — evaluator classes for all supported backends.

Hierarchy:
    BaseEvaluator          (abstract: metrics, answer extraction, result persistence)
    ├── VLLMEvaluator      (MCQ log-prob scoring + async generation via vLLM server)
    └── BatchEvaluator     (abstract: shared submit-poll-collect lifecycle)
        ├── OpenAIEvaluator    (OpenAI Batch API; async fallback via OPENAI_BASE_URL or force_async)
        ├── AnthropicEvaluator (Anthropic Message Batches API; async fallback via ANTHROPIC_BASE_URL)
        ├── GeminiEvaluator    (Google Gemini Batch API)
        └── XAIEvaluator       (xAI Grok — async via OpenAI-compatible endpoint)
"""

from __future__ import annotations

from typing import Optional

from .base import BaseEvaluator, BENCHMARK_FORMATS
from .vllm import VLLMEvaluator
from .batch import BatchEvaluator
from .openai import OpenAIEvaluator
from .anthropic import AnthropicEvaluator
from .gemini import GeminiEvaluator
from .xai import XAIEvaluator

__all__ = [
    "BaseEvaluator",
    "BatchEvaluator",
    "VLLMEvaluator",
    "OpenAIEvaluator",
    "AnthropicEvaluator",
    "GeminiEvaluator",
    "XAIEvaluator",
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
    "xai":       XAIEvaluator,
}


def make_evaluator(provider: Optional[str], **kwargs) -> BaseEvaluator:
    """
    Factory that returns the right evaluator for a given provider string.

    Args:
        provider: ``"openai"``, ``"openai-async"``, ``"anthropic"``, ``"gemini"``,
            ``"xai"``, or ``None`` (``None`` → :class:`VLLMEvaluator`).
            ``"openai-async"`` forces real-time async generation (useful for
            models where batch turnaround is impractical, e.g. gpt-5 family).
        **kwargs: Passed through to the evaluator constructor.

    Returns:
        A concrete :class:`BaseEvaluator` instance.

    Raises:
        ValueError: If ``provider`` is an unrecognised string.
    """
    if provider is None:
        return VLLMEvaluator(**kwargs)

    key = provider.lower()

    # "openai-async" is the same evaluator as "openai" but with async forced on,
    # bypassing the batch API regardless of whether OPENAI_BASE_URL is set.
    if key == "openai-async":
        return OpenAIEvaluator(force_async=True, **kwargs)

    cls = _BATCH_PROVIDERS.get(key)
    if cls is None:
        supported = ", ".join(f'"{p}"' for p in list(_BATCH_PROVIDERS) + ["openai-async"])
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {supported}, or None for vLLM."
        )
    return cls(**kwargs)
