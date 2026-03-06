"""
pacute-bench: End-to-end evaluation suite for Filipino morphology benchmarks.

Benchmarks:
  PACUTE   – affixation, composition, manipulation, syllabification (MCQ + gen)
  CUTE     – character-level understanding (generative)
  Hierarchical – compositional morphology levels 0–5 (MCQ + gen)
  LangGame – word-property language games (MCQ + gen)
  Multi-digit Addition – arithmetic (MCQ + gen)

Usage:
  python scripts/generate_benchmarks.py    # generate all benchmark JSONL files
  python scripts/run_evaluation.py ...     # run evaluation against a vLLM server
"""

__version__ = "0.1.0"

from .evaluator import VLLMEvaluator, CommercialEvaluator, BENCHMARK_FORMATS  # noqa: F401
