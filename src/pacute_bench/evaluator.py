"""
Backward-compatibility shim.

The evaluator classes have been moved to the ``pacute_bench.evaluators`` package.
All public names are re-exported here so existing imports continue to work.

    from pacute_bench.evaluator import VLLMEvaluator, CommercialEvaluator, BENCHMARK_FORMATS

is still valid, but prefer importing from ``pacute_bench.evaluators`` directly.
"""

from .evaluators import (  # noqa: F401
    BaseEvaluator,
    BatchEvaluator,
    VLLMEvaluator,
    OpenAIEvaluator,
    AnthropicEvaluator,
    GeminiEvaluator,
    XAIEvaluator,
    BENCHMARK_FORMATS,
    make_evaluator,
)

# CommercialEvaluator kept as a backward-compat alias.
# New code should use the specific provider class or make_evaluator().
CommercialEvaluator = BatchEvaluator