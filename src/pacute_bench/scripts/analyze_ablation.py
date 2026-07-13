"""
Ablation comparison script.

Compares EN vs TL benchmark variants (cross-lingual ablation) and
standard vs ng-hint syllabification (ng-hint ablation) for a set of models.

Usage:
    # Cross-lingual ablation
    python -m pacute_bench.scripts.analyze_ablation --crosslingual \
        --models gpt2 gemma-3-4b-it qwen3.6-27b-it sea-lion-qwen-v4.5-27b-it gemma-4-31b-it

    # ng-hint ablation
    python -m pacute_bench.scripts.analyze_ablation --nghint \
        --models gemma-3-27b-it sea-lion-qwen-v4.5-27b-it qwen3.6-27b-it gemma-4-31b-it

    # Both
    python -m pacute_bench.scripts.analyze_ablation --crosslingual --nghint --models <...>
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("results")

# EN benchmark key → TL variant key
CROSSLINGUAL_PAIRS = [
    ("hierarchical-mcq",                       "hierarchical-mcq-tl"),
    ("hierarchical-gen",                       "hierarchical-gen-tl"),
    ("pacute-composition-mcq",                 "pacute-composition-mcq-tl"),
    ("pacute-composition-gen",                 "pacute-composition-gen-tl"),
    ("pacute-manipulation-mcq",                "pacute-manipulation-mcq-tl"),
    ("pacute-manipulation-gen",                "pacute-manipulation-gen-tl"),
    ("pacute-morphological-extraction-mcq",    "pacute-morphological-extraction-mcq-tl"),
    ("pacute-morphological-extraction-gen",    "pacute-morphological-extraction-gen-tl"),
    ("pacute-morphological-production-mcq",    "pacute-morphological-production-mcq-tl"),
    ("pacute-morphological-production-gen",    "pacute-morphological-production-gen-tl"),
    ("pacute-syllabification-mcq",             "pacute-syllabification-mcq-tl"),
    ("pacute-syllabification-gen",             "pacute-syllabification-gen-tl"),
]

NGHINT_PAIR = ("pacute-syllabification-gen", "pacute-syllabification-gen-nghint")

# Primary metric per benchmark format
def _primary_metric(bm_key: str, bm_data: dict):
    """Return (metric_name, value) using the most informative single metric."""
    if bm_data is None:
        return None, None
    if bm_key.endswith("-mcq") or bm_key.endswith("-mcq-tl"):
        return "norm_acc", bm_data.get("normalized_accuracy")
    return "cont_match", bm_data.get("contains_match")


def load_results(model: str) -> dict:
    path = RESULTS_DIR / model / "evaluation_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f).get("benchmarks", {})


def _fmt(val, pct=True):
    if val is None:
        return "  N/A  "
    v = val * 100 if pct else val
    return f"{v:+6.1f}%" if pct else f"{v:.4f}"


def print_crosslingual(models: list[str]):
    print("\n" + "=" * 90)
    print("CROSS-LINGUAL ABLATION: English (EN) vs Tagalog (TL) item text")
    print("Metric: normalized_accuracy for MCQ, contains_match for GEN")
    print("=" * 90)

    # Short display names
    bm_labels = {
        "hierarchical-mcq":                     "Hier-MCQ",
        "hierarchical-gen":                     "Hier-GEN",
        "pacute-composition-mcq":               "Comp-MCQ",
        "pacute-composition-gen":               "Comp-GEN",
        "pacute-manipulation-mcq":              "Manip-MCQ",
        "pacute-manipulation-gen":              "Manip-GEN",
        "pacute-morphological-extraction-mcq":  "MExt-MCQ",
        "pacute-morphological-extraction-gen":  "MExt-GEN",
        "pacute-morphological-production-mcq":  "MProd-MCQ",
        "pacute-morphological-production-gen":  "MProd-GEN",
        "pacute-syllabification-mcq":           "Syll-MCQ",
        "pacute-syllabification-gen":           "Syll-GEN",
    }

    # Header
    col_w = 10
    header = f"{'Benchmark':<14}"
    for m in models:
        short = m[:18]
        header += f"  {'EN':>{col_w//2}}{'TL':>{col_w//2}}{'Δ':>{col_w//2}}"
    print(header)
    print("-" * (14 + len(models) * (col_w * 3 // 2 + 2)))

    missing_tl = set()
    for en_key, tl_key in CROSSLINGUAL_PAIRS:
        label = bm_labels.get(en_key, en_key)
        row = f"{label:<14}"
        for model in models:
            bm = load_results(model)
            _, en_val = _primary_metric(en_key, bm.get(en_key))
            _, tl_val = _primary_metric(tl_key, bm.get(tl_key))
            if tl_val is None:
                missing_tl.add((model, tl_key))
            en_s = f"{en_val*100:+5.1f}%" if en_val is not None else "  N/A "
            tl_s = f"{tl_val*100:+5.1f}%" if tl_val is not None else "  N/A "
            delta = (tl_val - en_val) if (tl_val is not None and en_val is not None) else None
            d_s = f"{delta*100:+5.1f}%" if delta is not None else "   N/A"
            row += f"  {en_s}{tl_s}{d_s}"
        print(row)

    if missing_tl:
        print("\nMISSING TL results (need to submit jobs):")
        for model, key in sorted(missing_tl):
            print(f"  {model}: {key}")


def print_nghint(models: list[str]):
    print("\n" + "=" * 70)
    print("NG-HINT ABLATION: standard Syll-GEN vs Syll-GEN with ng-digraph hint")
    print("Metric: contains_match")
    print("=" * 70)

    en_key, hint_key = NGHINT_PAIR
    print(f"\n{'Model':<45} {'Standard':>10} {'ng-hint':>10} {'Δ':>8}")
    print("-" * 75)
    missing = []
    for model in models:
        bm = load_results(model)
        _, std_val  = _primary_metric(en_key,   bm.get(en_key))
        _, hint_val = _primary_metric(hint_key, bm.get(hint_key))
        if hint_val is None:
            missing.append(model)
        std_s  = f"{std_val*100:5.1f}%"  if std_val  is not None else "   N/A"
        hint_s = f"{hint_val*100:5.1f}%" if hint_val is not None else "   N/A"
        delta  = (hint_val - std_val) if (hint_val is not None and std_val is not None) else None
        d_s    = f"{delta*100:+5.1f}%" if delta is not None else "    N/A"
        print(f"{model:<45} {std_s:>10} {hint_s:>10} {d_s:>8}")

    if missing:
        print(f"\nMISSING ng-hint results: {missing}")


def main():
    parser = argparse.ArgumentParser(description="Ablation comparison")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--crosslingual", action="store_true")
    parser.add_argument("--nghint",       action="store_true")
    parser.add_argument("--results-dir",  default="results", help="Path to results dir")
    args = parser.parse_args()

    global RESULTS_DIR
    RESULTS_DIR = Path(args.results_dir)

    if not args.crosslingual and not args.nghint:
        parser.error("Specify at least one of --crosslingual or --nghint")

    if args.crosslingual:
        print_crosslingual(args.models)
    if args.nghint:
        print_nghint(args.models)


if __name__ == "__main__":
    main()
