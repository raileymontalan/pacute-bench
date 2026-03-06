"""
Benchmark loader registry.

All registered benchmark names map to a callable that returns a generator of
(prefix, ground_truth, false_options, sample_id, category) tuples.
"""
from functools import partial

from .cute import load_cute
from .hierarchical import load_hierarchical
from .langgame import load_langgame
from .multi_digit_addition import load_multi_digit_addition
from .pacute import load_pacute

BENCHMARK_REGISTRY: dict = {
    # ── CUTE ────────────────────────────────────────────────────────────────
    "cute":     partial(load_cute, max_per_task=100),
    "cute-gen": partial(load_cute, max_per_task=100),

    # ── Hierarchical ────────────────────────────────────────────────────────
    "hierarchical":     partial(load_hierarchical, format="mcq"),
    "hierarchical-mcq": partial(load_hierarchical, format="mcq"),
    "hierarchical-gen": partial(load_hierarchical, format="gen"),

    # ── LangGame ────────────────────────────────────────────────────────────
    "langgame":     partial(load_langgame, format="mcq"),
    "langgame-mcq": partial(load_langgame, format="mcq"),
    "langgame-gen": partial(load_langgame, format="gen"),

    # ── Multi-digit Addition ─────────────────────────────────────────────────
    "multi-digit-addition":     partial(load_multi_digit_addition, format="gen",  max_samples=1000),
    "multi-digit-addition-gen": partial(load_multi_digit_addition, format="gen",  max_samples=1000),
    "multi-digit-addition-mcq": partial(load_multi_digit_addition, format="mcq",  max_samples=1000),

    # ── PACUTE — all categories ──────────────────────────────────────────────
    "pacute":     partial(load_pacute),
    "pacute-mcq": partial(load_pacute, format="mcq"),
    "pacute-gen": partial(load_pacute, format="gen"),

    # PACUTE — per-category variants
    "pacute-affixation":        partial(load_pacute, categories=["affixation"]),
    "pacute-affixation-mcq":    partial(load_pacute, categories=["affixation"], format="mcq"),
    "pacute-affixation-gen":    partial(load_pacute, categories=["affixation"], format="gen"),
    "pacute-composition":       partial(load_pacute, categories=["composition"]),
    "pacute-composition-mcq":   partial(load_pacute, categories=["composition"], format="mcq"),
    "pacute-composition-gen":   partial(load_pacute, categories=["composition"], format="gen"),
    "pacute-manipulation":      partial(load_pacute, categories=["manipulation"]),
    "pacute-manipulation-mcq":  partial(load_pacute, categories=["manipulation"], format="mcq"),
    "pacute-manipulation-gen":  partial(load_pacute, categories=["manipulation"], format="gen"),
    "pacute-syllabification":       partial(load_pacute, categories=["syllabification"]),
    "pacute-syllabification-mcq":   partial(load_pacute, categories=["syllabification"], format="mcq"),
    "pacute-syllabification-gen":   partial(load_pacute, categories=["syllabification"], format="gen"),
}


def load_benchmark(name: str):
    """Return an iterable of benchmark items for the given benchmark name."""
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(
            f"Unknown benchmark: {name!r}.\n"
            f"Available benchmarks: {sorted(BENCHMARK_REGISTRY)}"
        )
    return BENCHMARK_REGISTRY[name]()
