#!/usr/bin/env python3
"""
Generate stratified 10% samples from all active PACUTE-bench benchmarks.

Run from the repo root:
    python scripts/sample_human_baseline.py

Outputs:
    data/human_baseline/sample_mcq.jsonl
    data/human_baseline/sample_gen.jsonl
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

SEED = 42
SAMPLE_RATE = 0.10
DATA_DIR = Path("data/benchmarks")
EVAL_CONFIG = Path("configs/evaluation.yaml")
OUT_DIR = Path("data/human_baseline")

# --- Benchmark definitions ---------------------------------------------------
# (benchmark_name, jsonl_stem, format)
# affixation excluded (deprecated)
MCQ_BENCHMARKS = [
    "composition",
    "manipulation",
    "syllabification",
    "morphological_extraction",
    "morphological_production",
    "hierarchical",
    "langgame",
    "multi_digit_addition",
]
GEN_BENCHMARKS = MCQ_BENCHMARKS + ["cute"]

# Maps JSONL stem → system_prompt key in evaluation.yaml
GEN_PROMPT_KEYS = {
    "composition":              "pacute-composition-gen",
    "manipulation":             "pacute-manipulation-gen",
    "syllabification":          "pacute-syllabification-gen",
    "morphological_extraction": "pacute-morphological-extraction-gen",
    "morphological_production": "pacute-morphological-production-gen",
    "hierarchical":             "hierarchical-gen",
    "langgame":                 "langgame-gen",
    "multi_digit_addition":     "multi-digit-addition-gen",
    "cute":                     "cute-gen",
}


def load_gen_instructions():
    with open(EVAL_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {k: v["instruction"] for k, v in cfg["generative_instructions"].items()}


def _stratum_key(item, benchmark):
    """Return (category, subcategory) stratum for an item."""
    if benchmark == "cute":
        tt = item.get("task_type", "unknown")
        return (tt, "")
    cat = item.get("category", "")
    sub = item.get("subcategory", str(item.get("level", "")))
    return (cat, sub)


def _stratified_sample(items, rate, seed):
    """
    Sample ~rate% of items, stratified by _stratum key.

    Uses the Hamilton (largest-remainder) method so per-stratum allocations
    sum to exactly floor(total * rate).  Each stratum gets at least 1 item.
    """
    rng = random.Random(seed)
    strata = defaultdict(list)
    for item in items:
        strata[item["_stratum"]].append(item)

    total = len(items)
    target = max(len(strata), math.floor(total * rate))  # at least 1 per stratum

    # Initial allocation: floor quota, min 1
    sorted_keys = sorted(strata.keys())
    alloc = {}
    remainders = {}
    for key in sorted_keys:
        quota = len(strata[key]) * rate
        alloc[key] = max(1, math.floor(quota))
        remainders[key] = quota - math.floor(quota)  # fractional part (pre-max)

    # Distribute remaining slots to strata with largest fractional remainders
    deficit = target - sum(alloc.values())
    if deficit > 0:
        for key in sorted(sorted_keys, key=lambda k: remainders[k], reverse=True):
            if deficit == 0:
                break
            if alloc[key] < len(strata[key]):
                alloc[key] += 1
                deficit -= 1
    elif deficit < 0:
        # Over-allocated (can happen when min-1 inflates small strata).
        # Remove from strata with smallest remainders, but keep >= 1.
        for key in sorted(sorted_keys, key=lambda k: remainders[k]):
            if deficit == 0:
                break
            if alloc[key] > 1:
                alloc[key] -= 1
                deficit += 1

    sampled = []
    for key in sorted_keys:
        n = min(alloc[key], len(strata[key]))
        sampled.extend(rng.sample(strata[key], n))
    return sampled


def _item_rng(item_id):
    """Deterministic per-item RNG for option shuffling."""
    seed = abs(hash(item_id)) % (2**31)
    return random.Random(seed)


def _assign_options(options_list, correct_value, item_id):
    """Shuffle options and return (options_dict, correct_letter)."""
    rng = _item_rng(item_id)
    shuffled = options_list[:]
    rng.shuffle(shuffled)
    letters = ["A", "B", "C", "D"]
    options_dict = {letters[i]: v for i, v in enumerate(shuffled)}
    correct_letter = None
    norm_correct = str(correct_value).strip()
    for letter, val in options_dict.items():
        if str(val).strip() == norm_correct:
            correct_letter = letter
            break
    if correct_letter is None:
        # fallback: find by stripped comparison
        for letter, val in options_dict.items():
            if str(val).strip().lower() == norm_correct.lower():
                correct_letter = letter
                break
    return options_dict, correct_letter


# --- Per-format extractors ---------------------------------------------------

def _extract_pacute_mcq(item):
    """Return (question_en, question_tl, options_dict, correct_letter) for PACUTE MCQ."""
    p = item["prompts"][0]
    q_en = p["text_en"]
    q_tl = p.get("text_tl")

    # PACUTE MCQ already has shuffled choices; label tells us correct letter
    has_choices = "choice1" in p
    if has_choices:
        n_choices = sum(1 for k in p if k.startswith("choice") and k[6:].isdigit())
        letters = ["A", "B", "C", "D"][:n_choices]
        options_dict = {letters[i]: p[f"choice{i+1}"] for i in range(n_choices)}
        correct_letter = item["label"]
    else:
        # fallback: mcq_options dict
        mcq = p["mcq_options"]
        all_vals = [mcq["correct"]] + [v for k, v in sorted(mcq.items()) if k.startswith("incorrect")]
        options_dict, correct_letter = _assign_options(all_vals, mcq["correct"], item["id"])

    return q_en, q_tl, options_dict, correct_letter


def _extract_pacute_gen(item):
    """Return (question_en, question_tl, answer) for PACUTE GEN."""
    p = item["prompts"][0]
    return p["text_en"], p.get("text_tl"), item["label"]


def _extract_hierarchical_mcq(item):
    q_en = item.get("prompt_en", "")
    q_tl = item.get("prompt_tl")
    options_list = item["options"]
    correct_value = item["answer"]
    options_dict, correct_letter = _assign_options(options_list, correct_value, item["id"])
    return q_en, q_tl, options_dict, correct_letter


def _extract_hierarchical_gen(item):
    return item.get("prompt_en", ""), item.get("prompt_tl"), item["answer"]


def _extract_flat_mcq(item):
    """For langgame and multi_digit_addition MCQ."""
    q_en = item["question"]
    options_list = item["options"]
    correct_value = item["answer"]
    options_dict, correct_letter = _assign_options(options_list, correct_value, item["id"])
    return q_en, None, options_dict, correct_letter


def _extract_flat_gen(item):
    return item["question"], None, item["answer"]


def _extract_cute_gen(item):
    return item["question"], None, item["answer"]


def normalize_mcq_item(raw, benchmark, gen_instructions):
    raw["_stratum"] = _stratum_key(raw, benchmark)
    cat, sub = raw["_stratum"]

    if benchmark in ("composition", "manipulation", "syllabification",
                     "morphological_extraction", "morphological_production"):
        q_en, q_tl, options, correct = _extract_pacute_mcq(raw)
    elif benchmark == "hierarchical":
        q_en, q_tl, options, correct = _extract_hierarchical_mcq(raw)
    else:
        q_en, q_tl, options, correct = _extract_flat_mcq(raw)

    return {
        "item_id":       raw.get("id", f"{benchmark}_mcq_unknown"),
        "benchmark":     benchmark,
        "format":        "mcq",
        "category":      cat or None,
        "subcategory":   sub or None,
        "question_en":   q_en,
        "question_tl":   q_tl,
        "options":       options,
        "correct_option": correct,
        "n_options":     len(options),
    }


def normalize_gen_item(raw, benchmark, gen_instructions):
    raw["_stratum"] = _stratum_key(raw, benchmark)
    cat, sub = raw["_stratum"]

    if benchmark in ("composition", "manipulation", "syllabification",
                     "morphological_extraction", "morphological_production"):
        q_en, q_tl, answer = _extract_pacute_gen(raw)
    elif benchmark == "hierarchical":
        q_en, q_tl, answer = _extract_hierarchical_gen(raw)
    elif benchmark == "cute":
        q_en, q_tl, answer = _extract_cute_gen(raw)
    else:
        q_en, q_tl, answer = _extract_flat_gen(raw)

    prompt_key = GEN_PROMPT_KEYS.get(benchmark, "")
    system_prompt = gen_instructions.get(prompt_key, "")

    return {
        "item_id":          raw.get("id", f"{benchmark}_gen_unknown"),
        "benchmark":        benchmark,
        "format":           "gen",
        "category":         cat or None,
        "subcategory":      sub or None,
        "question_en":      q_en,
        "question_tl":      q_tl,
        "system_prompt":    system_prompt,
        "reference_answer": str(answer).strip(),
    }


def sample_benchmark(benchmark, fmt, gen_instructions):
    stem = f"{benchmark}_{fmt}"
    path = DATA_DIR / f"{stem}.jsonl"
    if not path.exists():
        print(f"  SKIP: {path} not found", file=sys.stderr)
        return []

    with open(path, encoding="utf-8") as f:
        raw_items = [json.loads(l) for l in f if l.strip()]

    # Attach stratum key before sampling
    for item in raw_items:
        item["_stratum"] = _stratum_key(item, benchmark)

    sampled = _stratified_sample(raw_items, SAMPLE_RATE, seed=f"{benchmark}_{fmt}_{SEED}")

    if fmt == "mcq":
        out = [normalize_mcq_item(item, benchmark, gen_instructions) for item in sampled]
    else:
        out = [normalize_gen_item(item, benchmark, gen_instructions) for item in sampled]

    print(f"  {stem}: {len(raw_items)} → sampled {len(out)}")
    return out


def main():
    gen_instructions = load_gen_instructions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Sampling MCQ benchmarks...")
    mcq_items = []
    for bm in MCQ_BENCHMARKS:
        mcq_items.extend(sample_benchmark(bm, "mcq", gen_instructions))

    print(f"\nTotal MCQ items: {len(mcq_items)}")
    out_mcq = OUT_DIR / "sample_mcq.jsonl"
    with open(out_mcq, "w", encoding="utf-8") as f:
        for item in mcq_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Written → {out_mcq}")

    print("\nSampling GEN benchmarks...")
    gen_items = []
    for bm in GEN_BENCHMARKS:
        gen_items.extend(sample_benchmark(bm, "gen", gen_instructions))

    print(f"\nTotal GEN items: {len(gen_items)}")
    out_gen = OUT_DIR / "sample_gen.jsonl"
    with open(out_gen, "w", encoding="utf-8") as f:
        for item in gen_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Written → {out_gen}")


if __name__ == "__main__":
    main()
