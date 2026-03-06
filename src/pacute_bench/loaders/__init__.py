"""
Benchmark loaders for pacute-bench.
"""

from .registry import load_benchmark, BENCHMARK_REGISTRY
from .pacute import load_pacute
from .cute import load_cute
from .hierarchical import load_hierarchical
from .langgame import load_langgame
from .multi_digit_addition import load_multi_digit_addition

__all__ = [
    "load_benchmark",
    "BENCHMARK_REGISTRY",
    "load_pacute",
    "load_cute",
    "load_hierarchical",
    "load_langgame",
    "load_multi_digit_addition",
]
