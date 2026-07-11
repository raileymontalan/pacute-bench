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
    # Default keys use language="en" (the loader's default), consistent with
    # the rest of PACUTE below. Previously defaulted to Tagalog — all
    # already-collected results/ were run under that old default; they must
    # be re-run against these English-default keys before being reported
    # alongside the rest of PACUTE. Use the explicit -tl variants to
    # reproduce the old runs or to run the cross-lingual ablation.
    "hierarchical":     partial(load_hierarchical, format="mcq"),
    "hierarchical-mcq": partial(load_hierarchical, format="mcq"),
    "hierarchical-gen": partial(load_hierarchical, format="gen"),
    "hierarchical-mcq-en": partial(load_hierarchical, format="mcq", language="en"),
    "hierarchical-mcq-tl": partial(load_hierarchical, format="mcq", language="tl"),
    "hierarchical-gen-en": partial(load_hierarchical, format="gen", language="en"),
    "hierarchical-gen-tl": partial(load_hierarchical, format="gen", language="tl"),

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
    "pacute-composition":       partial(load_pacute, categories=["composition"]),
    "pacute-composition-mcq":   partial(load_pacute, categories=["composition"], format="mcq"),
    "pacute-composition-gen":   partial(load_pacute, categories=["composition"], format="gen"),
    "pacute-manipulation":      partial(load_pacute, categories=["manipulation"]),
    "pacute-manipulation-mcq":  partial(load_pacute, categories=["manipulation"], format="mcq"),
    "pacute-manipulation-gen":  partial(load_pacute, categories=["manipulation"], format="gen"),
    "pacute-morphological-extraction":     partial(load_pacute, categories=["morphological_extraction"]),
    "pacute-morphological-extraction-mcq": partial(load_pacute, categories=["morphological_extraction"], format="mcq"),
    "pacute-morphological-extraction-gen": partial(load_pacute, categories=["morphological_extraction"], format="gen"),
    "pacute-morphological-production":     partial(load_pacute, categories=["morphological_production"]),
    "pacute-morphological-production-mcq": partial(load_pacute, categories=["morphological_production"], format="mcq"),
    "pacute-morphological-production-gen": partial(load_pacute, categories=["morphological_production"], format="gen"),
    "pacute-syllabification":       partial(load_pacute, categories=["syllabification"]),
    "pacute-syllabification-mcq":   partial(load_pacute, categories=["syllabification"], format="mcq"),
    "pacute-syllabification-gen":   partial(load_pacute, categories=["syllabification"], format="gen"),

    # Ablation: same syllabification GEN items, paired with an alternate
    # instruction (see configs/evaluation.yaml) that explicitly tells the
    # model to treat "ng" as a single sound. Tests whether ng-digraph
    # overcounting (reported in our error analysis) is a missing-knowledge
    # failure (persists despite the hint) or an attention/salience failure
    # (resolved by the hint).
    "pacute-syllabification-gen-nghint": partial(load_pacute, categories=["syllabification"], format="gen"),

    # ── Cross-lingual instruction-language ablation ─────────────────────────
    # Every PACUTE item ships with both text_en and text_tl instruction
    # variants (see data/benchmarks/*.jsonl). These -en/-tl pairs let us
    # measure the delta between English- and Tagalog-instructed performance
    # per category, holding the underlying item (and gold answer) fixed.
    # Requires configs/evaluation.yaml instruction text to also be varied by
    # language for a fully controlled comparison (see NEXT_STEPS.md).
    "pacute-composition-mcq-en":               partial(load_pacute, categories=["composition"], format="mcq", language="en"),
    "pacute-composition-mcq-tl":               partial(load_pacute, categories=["composition"], format="mcq", language="tl"),
    "pacute-composition-gen-en":               partial(load_pacute, categories=["composition"], format="gen", language="en"),
    "pacute-composition-gen-tl":               partial(load_pacute, categories=["composition"], format="gen", language="tl"),
    "pacute-manipulation-mcq-en":              partial(load_pacute, categories=["manipulation"], format="mcq", language="en"),
    "pacute-manipulation-mcq-tl":              partial(load_pacute, categories=["manipulation"], format="mcq", language="tl"),
    "pacute-manipulation-gen-en":              partial(load_pacute, categories=["manipulation"], format="gen", language="en"),
    "pacute-manipulation-gen-tl":              partial(load_pacute, categories=["manipulation"], format="gen", language="tl"),
    "pacute-morphological-extraction-mcq-en":  partial(load_pacute, categories=["morphological_extraction"], format="mcq", language="en"),
    "pacute-morphological-extraction-mcq-tl":  partial(load_pacute, categories=["morphological_extraction"], format="mcq", language="tl"),
    "pacute-morphological-extraction-gen-en":  partial(load_pacute, categories=["morphological_extraction"], format="gen", language="en"),
    "pacute-morphological-extraction-gen-tl":  partial(load_pacute, categories=["morphological_extraction"], format="gen", language="tl"),
    "pacute-morphological-production-mcq-en":  partial(load_pacute, categories=["morphological_production"], format="mcq", language="en"),
    "pacute-morphological-production-mcq-tl":  partial(load_pacute, categories=["morphological_production"], format="mcq", language="tl"),
    "pacute-morphological-production-gen-en":  partial(load_pacute, categories=["morphological_production"], format="gen", language="en"),
    "pacute-morphological-production-gen-tl":  partial(load_pacute, categories=["morphological_production"], format="gen", language="tl"),
    "pacute-syllabification-mcq-en":           partial(load_pacute, categories=["syllabification"], format="mcq", language="en"),
    "pacute-syllabification-mcq-tl":           partial(load_pacute, categories=["syllabification"], format="mcq", language="tl"),
    "pacute-syllabification-gen-en":           partial(load_pacute, categories=["syllabification"], format="gen", language="en"),
    "pacute-syllabification-gen-tl":           partial(load_pacute, categories=["syllabification"], format="gen", language="tl"),

    # ── Morphological (aliases → load_pacute) ────────────────────────────────
    "morphological":     partial(load_pacute, categories=["morphological_extraction", "morphological_production"]),
    "morphological-mcq": partial(load_pacute, categories=["morphological_extraction", "morphological_production"], format="mcq"),
    "morphological-gen": partial(load_pacute, categories=["morphological_extraction", "morphological_production"], format="gen"),

    "morphological-extraction":     partial(load_pacute, categories=["morphological_extraction"]),
    "morphological-extraction-mcq": partial(load_pacute, categories=["morphological_extraction"], format="mcq"),
    "morphological-extraction-gen": partial(load_pacute, categories=["morphological_extraction"], format="gen"),

    "morphological-production":     partial(load_pacute, categories=["morphological_production"]),
    "morphological-production-mcq": partial(load_pacute, categories=["morphological_production"], format="mcq"),
    "morphological-production-gen": partial(load_pacute, categories=["morphological_production"], format="gen"),
}


def load_benchmark(name: str):
    """Return an iterable of benchmark items for the given benchmark name."""
    if name not in BENCHMARK_REGISTRY:
        raise KeyError(
            f"Unknown benchmark: {name!r}.\n"
            f"Available benchmarks: {sorted(BENCHMARK_REGISTRY)}"
        )
    return BENCHMARK_REGISTRY[name]()
