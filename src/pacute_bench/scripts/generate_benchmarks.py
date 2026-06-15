#!/usr/bin/env python3
"""
Generate all evaluation benchmarks for pacute-bench.

Benchmarks generated:
  1. pacute       – all 5 PACUTE tasks: composition, manipulation, syllabification,
                    morphological_extraction, morphological_production (MCQ + GEN)
  2. hierarchical – 6-level diagnostic tasks (MCQ + GEN)
  3. cute         – character-understanding tasks from HuggingFace (GEN)
  4. langgame     – language reasoning tasks (MCQ + GEN)
  5. math         – multi-digit addition (GEN + MCQ)

After generation, unique IDs are added to every sample and MCQ↔GEN variants
are derived where needed.

Usage:
    python scripts/generate_benchmarks.py
    python scripts/generate_benchmarks.py --corpora-dir /path/to/corpora \\
                                          --output-dir data/benchmarks
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# PACUTE benchmarks
# ---------------------------------------------------------------------------

def generate_pacute(output_dir: Path, corpora_dir: Path, random_seed: int = 1859) -> bool:
    """Generate all 5 PACUTE benchmarks: composition, manipulation, syllabification,
    morphological_extraction, and morphological_production."""
    import pandas as pd
    from pacute_bench.generators import (
        create_composition_dataset,
        create_corpus_composition_dataset,
        create_manipulation_dataset,
        create_syllabification_dataset,
    )

    print("=" * 80)
    print("Generating PACUTE Benchmarks")
    print("=" * 80)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pacute_data      = corpora_dir / "pacute_data"
    syllables_path   = pacute_data / "syllables.jsonl"
    corpus_comp_path = pacute_data / "corpus_composition.jsonl"

    if not syllables_path.exists():
        print(f"  ERROR: {syllables_path} not found — skip pacute"); return False
    syllables = pd.read_json(syllables_path, lines=True)
    print(f"  Loaded {len(syllables)} syllabified words")

    corpus_comp = None
    if corpus_comp_path.exists():
        corpus_comp = pd.read_json(corpus_comp_path, lines=True)
        print(f"  Loaded {len(corpus_comp)} corpus composition rows")
    else:
        print(f"  Warning: {corpus_comp_path} not found — skipping corpus-driven tasks")

    # ---- composition -------------------------------------------------------
    print("\n[1/3] Composition …")
    for mode in ("mcq", "gen"):
        # Syllable-based tasks (spelling, character, length variants)
        ds = create_composition_dataset(syllables, mode=mode, num_samples=100, random_seed=random_seed)
        # Corpus-driven tasks (diacritics, uppercasing, character_counting, character_recognition)
        if corpus_comp is not None:
            ds_corpus = create_corpus_composition_dataset(corpus_comp, mode=mode, random_seed=random_seed)
            ds = pd.concat([ds, ds_corpus], ignore_index=True)
        out = output_dir / f"composition_{mode}.jsonl"
        ds.to_json(out, lines=True, orient="records", force_ascii=False)
        print(f"  ✓ composition_{mode}.jsonl  ({len(ds)} samples)")

    # ---- manipulation ------------------------------------------------------
    print("\n[2/3] Manipulation …")
    for mode in ("mcq", "gen"):
        ds = create_manipulation_dataset(syllables, mode=mode, num_samples=100, random_seed=random_seed)
        out = output_dir / f"manipulation_{mode}.jsonl"
        ds.to_json(out, lines=True, orient="records", force_ascii=False)
        print(f"  ✓ manipulation_{mode}.jsonl  ({len(ds)} samples)")

    # ---- syllabification (stress tasks only) --------------------------------
    print("\n[3/3] Syllabification (stress tasks only) …")
    csv_dir = str(pacute_data)
    _STRESS_SUBCATEGORIES = {"stress_identification", "stress_disambiguation"}
    for mode in ("mcq", "gen"):
        ds = create_syllabification_dataset(syllables, mode=mode, num_samples=100,
                                            random_seed=random_seed, csv_dir=csv_dir)
        ds = ds[ds["subcategory"].isin(_STRESS_SUBCATEGORIES)].reset_index(drop=True)
        out = output_dir / f"syllabification_{mode}.jsonl"
        ds.to_json(out, lines=True, orient="records", force_ascii=False)
        print(f"  ✓ syllabification_{mode}.jsonl  ({len(ds)} samples)")

    # ---- morphological extraction ------------------------------------------
    print("\n[4/5] Morphological Extraction …")
    if not _generate_morphological_extraction(output_dir, corpora_dir, random_seed):
        return False

    # ---- morphological production ------------------------------------------
    print("\n[5/5] Morphological Production …")
    if not _generate_morphological_production(output_dir, corpora_dir, random_seed):
        return False

    print()
    return True


# ---------------------------------------------------------------------------
# Morphological Extraction benchmarks
# ---------------------------------------------------------------------------

def _generate_morphological_extraction(output_dir: Path, corpora_dir: Path, random_seed: int = 1859) -> bool:
    """Generate morphological extraction benchmarks."""
    import pandas as pd
    from pacute_bench.generators import create_morphological_extraction_dataset

    print("=" * 80)
    print("Generating Morphological Extraction Benchmarks")
    print("=" * 80)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpora_dir / "pacute_data" / "corpus_morphological_extraction.jsonl"
    if not corpus_path.exists():
        print(f"  ERROR: {corpus_path} not found — run convert_dataset first"); return False

    corpus_df = pd.read_json(corpus_path, lines=True)
    print(f"  Loaded {len(corpus_df)} corpus rows")

    for mode in ("mcq", "gen"):
        ds = create_morphological_extraction_dataset(corpus_df, mode=mode, random_seed=random_seed, num_samples=100)
        out = output_dir / f"morphological_extraction_{mode}.jsonl"
        ds.to_json(out, lines=True, orient="records", force_ascii=False)
        from collections import Counter
        sub_counts = Counter(ds["subcategory"].tolist())
        print(f"  ✓ morphological_extraction_{mode}.jsonl  ({len(ds)} samples: {dict(sub_counts)})")

    print()
    return True


# ---------------------------------------------------------------------------
# Morphological Production benchmarks
# ---------------------------------------------------------------------------

def _generate_morphological_production(output_dir: Path, corpora_dir: Path, random_seed: int = 1859) -> bool:
    """Generate morphological production benchmarks."""
    import pandas as pd
    from pacute_bench.generators import create_morphological_production_dataset

    print("=" * 80)
    print("Generating Morphological Production Benchmarks")
    print("=" * 80)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = corpora_dir / "pacute_data" / "corpus_morphological_extraction.jsonl"
    if not corpus_path.exists():
        print(f"  ERROR: {corpus_path} not found — run convert_dataset first"); return False

    corpus_df = pd.read_json(corpus_path, lines=True)
    print(f"  Loaded {len(corpus_df)} corpus rows")

    for mode in ("mcq", "gen"):
        ds = create_morphological_production_dataset(corpus_df, mode=mode, random_seed=random_seed, num_samples=100)
        out = output_dir / f"morphological_production_{mode}.jsonl"
        ds.to_json(out, lines=True, orient="records", force_ascii=False)
        print(f"  ✓ morphological_production_{mode}.jsonl  ({len(ds)} samples)")

    print()
    return True


# ---------------------------------------------------------------------------
# Hierarchical benchmarks
# ---------------------------------------------------------------------------

def generate_hierarchical(output_dir: Path, corpora_dir: Path) -> bool:
    """Generate hierarchical diagnostic benchmarks (6 levels)."""
    import pandas as pd
    from pacute_bench.generators import HierarchicalTaskGenerator

    print("=" * 80)
    print("Generating Hierarchical Benchmarks")
    print("=" * 80)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    syllables_path   = corpora_dir / "pacute_data" / "syllables.jsonl"
    annotations_path = corpora_dir / "affix_annotations.jsonl"

    if not syllables_path.exists():
        print(f"  ERROR: {syllables_path} not found"); return False

    syllables_df  = pd.read_json(syllables_path, lines=True)
    affixes_df    = None
    if annotations_path.exists():
        affixes_df = pd.read_json(annotations_path, lines=True)
        print(f"  Loaded {len(affixes_df)} affix annotations")
    else:
        print(f"  Warning: {annotations_path} not found — levels 2-5 will be sparse")

    print(f"  Loaded {len(syllables_df)} syllabified words")

    gen = HierarchicalTaskGenerator(syllables_df, affixes_df)

    for fmt in ("mcq", "gen"):
        tasks_by_level = gen.generate_all_levels(n_per_subcategory=20, format=fmt)
        all_tasks = [t for lvl in tasks_by_level.values() for t in lvl]
        out = output_dir / f"hierarchical_{fmt}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for task in all_tasks:
                record = task.to_mcq_dict() if fmt == "mcq" else task.to_gen_dict()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts = {lvl: len(tasks) for lvl, tasks in tasks_by_level.items()}
        total  = sum(counts.values())
        print(f"  ✓ hierarchical_{fmt}.jsonl  ({total} tasks across levels {list(counts)})")

    print()
    return True


# ---------------------------------------------------------------------------
# LangGame benchmark
# ---------------------------------------------------------------------------

def _build_langgame_question(question_type_idx, all_words, total_options=4):
    """Generate one LangGame MCQ question (adapted from original script)."""
    # question type descriptions
    placeholder_options = "<OPTIONS>"
    synonyms_options = ["options", "choices", "option words", "option strings"]
    placeholder_option = "<OPTION>"
    synonyms_option = ["word", "", "string", "option", "choice", "option word", "option string"]
    placeholder_the = "<THE>"
    synonyms_the = ["the", "the possible", "the available"]
    question1_part1_strings = [
        "Which <OPTION>",
        "What <OPTION>",
        "Which of <THE> <OPTIONS>",
    ]
    question1_option_part_strings = [
        "<THE> <OPTIONS>:",
        "<THE> <OPTIONS> are:",
        "These are <THE> <OPTIONS>:",
    ]
    q_descriptions = [
        "most of letter",
        "contains",
        "starts with",
        "ends with",
        "longest",
        "shortest",
    ]

    def q0_mostofletter(all_words, total_options):
        target_letter = random.choice("abcdefghijklmnopqrstuvwxyz")
        words = random.sample(all_words, total_options)
        counts = [w.count(target_letter) for w in words]
        max_count = max(counts)
        max_indices = [i for i, c in enumerate(counts) if c == max_count]
        if len(max_indices) != 1:
            return False, None, None
        correct_idx = max_indices[0]
        answer = words[correct_idx]
        options = [answer] + [words[i] for i in range(total_options) if i != correct_idx]
        return True, target_letter, options

    def q1_contains(all_words, total_options):
        target_letter = random.choice("abcdefghijklmnopqrstuvwxyz")
        candidates = [w for w in all_words if target_letter in w]
        non_candidates = [w for w in all_words if target_letter not in w]
        if len(candidates) < 1 or len(non_candidates) < total_options - 1:
            return False, None, None
        answer = random.choice(candidates)
        distractors = random.sample(non_candidates, total_options - 1)
        options = [answer] + distractors
        return True, target_letter, options

    def q2_startswith(all_words, total_options):
        target_letter = random.choice("abcdefghijklmnopqrstuvwxyz")
        candidates = [w for w in all_words if w.startswith(target_letter)]
        non_candidates = [w for w in all_words if not w.startswith(target_letter)]
        if len(candidates) < 1 or len(non_candidates) < total_options - 1:
            return False, None, None
        answer = random.choice(candidates)
        distractors = random.sample(non_candidates, total_options - 1)
        options = [answer] + distractors
        return True, target_letter, options

    def q3_endswith(all_words, total_options):
        target_letter = random.choice("abcdefghijklmnopqrstuvwxyz")
        candidates = [w for w in all_words if w.endswith(target_letter)]
        non_candidates = [w for w in all_words if not w.endswith(target_letter)]
        if len(candidates) < 1 or len(non_candidates) < total_options - 1:
            return False, None, None
        answer = random.choice(candidates)
        distractors = random.sample(non_candidates, total_options - 1)
        options = [answer] + distractors
        return True, target_letter, options

    def q4_longest(all_words, total_options):
        words = random.sample(all_words, total_options)
        lengths = [len(w) for w in words]
        max_len = max(lengths)
        max_indices = [i for i, l in enumerate(lengths) if l == max_len]
        if len(max_indices) != 1:
            return False, None, None
        correct_idx = max_indices[0]
        answer = words[correct_idx]
        options = [answer] + [words[i] for i in range(total_options) if i != correct_idx]
        return True, None, options

    def q5_shortest(all_words, total_options):
        words = random.sample(all_words, total_options)
        lengths = [len(w) for w in words]
        min_len = min(lengths)
        min_indices = [i for i, l in enumerate(lengths) if l == min_len]
        if len(min_indices) != 1:
            return False, None, None
        correct_idx = min_indices[0]
        answer = words[correct_idx]
        options = [answer] + [words[i] for i in range(total_options) if i != correct_idx]
        return True, None, options

    q_strings_list = [
        ["has the most letter '<AUX>'s?"],
        ["contains '<AUX>'?"],
        ["starts with '<AUX>'?"],
        ["ends with '<AUX>'?"],
        ["is the longest?"],
        ["is the shortest?"],
    ]
    q_functions = [q0_mostofletter, q1_contains, q2_startswith, q3_endswith, q4_longest, q5_shortest]

    q_idx = (question_type_idx % 6) + 1
    if q_idx < 1 or q_idx > 6:
        return None, None, None

    success, aux, options = False, None, None
    for _ in range(100):
        success, aux, options = q_functions[q_idx - 1](all_words, total_options)
        if success:
            break
    if not success:
        return None, None, None

    synonym_options1 = random.choice(synonyms_options)
    synonym_option1  = random.choice(synonyms_option)
    synonym_the1     = random.choice(synonyms_the)
    gen_part = random.choice(question1_part1_strings)
    spec_part = random.choice(q_strings_list[q_idx - 1])
    opts_part = random.choice(question1_option_part_strings)

    gen_part  = gen_part.replace("<OPTIONS>", synonym_options1).replace("<OPTION>", synonym_option1).replace("<THE>", synonym_the1)
    gen_part  = gen_part[0].upper() + gen_part[1:]
    if "<AUX>" in spec_part and aux is not None:
        spec_part = spec_part.replace("<AUX>", aux)
    opts_part = opts_part.replace("<OPTIONS>", synonym_options1).replace("<OPTION>", synonym_option1).replace("<THE>", synonym_the1)
    opts_part = opts_part[0].upper() + opts_part[1:]

    i = np.random.randint(len(options))
    shuffled = options[i:] + options[:i]
    opts_part = opts_part + f" [ {', '.join(shuffled)}]."

    if np.random.choice([True, False]):
        question = f"{gen_part} {spec_part} {opts_part} Answer:"
    else:
        question = f"{opts_part} {gen_part} {spec_part} Answer:"

    options_ = [f" {opt}" for opt in options]
    answer   = options_[0]
    return question, answer, options_


def generate_langgame(output_dir: Path, corpora_dir: Path) -> bool:
    """Generate LangGame benchmark (MCQ format, requires top_1k_words corpus)."""
    print("=" * 80)
    print("Generating LangGame Dataset")
    print("=" * 80)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_1k_path = corpora_dir / "top_1k_words"
    if not top_1k_path.exists():
        print(f"  ERROR: {top_1k_path} not found"); return False

    with open(top_1k_path, encoding="utf-8") as f:
        top_1k_words = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(top_1k_words)} words")

    random.seed(42)
    np.random.seed(42)

    val_size = 1000
    total_options = 4
    question_s, answer_s, options_s = [], [], []

    start = time.time()
    max_tries = val_size * 4
    for _ in range(max_tries):
        if len(question_s) >= val_size:
            break
        qt = np.random.choice(range(6))
        q, a, opts = _build_langgame_question(qt, top_1k_words, total_options)
        if q is not None:
            question_s.append(q)
            answer_s.append(a)
            options_s.append(opts)

    print(f"  Generated {len(question_s)} samples in {time.time()-start:.1f}s")

    out_mcq = output_dir / "langgame_mcq.jsonl"
    with open(out_mcq, "w", encoding="utf-8") as f:
        for q, a, opts in zip(question_s, answer_s, options_s):
            f.write(json.dumps({"question": q, "answer": a, "options": opts}, ensure_ascii=False) + "\n")
    print(f"  ✓ langgame_mcq.jsonl  ({len(question_s)} samples)")
    print()
    return True


# ---------------------------------------------------------------------------
# Multi-digit Addition benchmark
# ---------------------------------------------------------------------------

def generate_math(output_dir: Path) -> bool:
    """Generate multi-digit addition benchmark (GEN format)."""
    print("=" * 80)
    print("Generating Multi-digit Addition Dataset")
    print("=" * 80)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    num_digits = 3
    val_size   = 1000

    all_pairs = np.arange(10 ** (2 * num_digits - 1), 10 ** (2 * num_digits))
    np.random.shuffle(all_pairs)

    items = []
    for pair in all_pairs[:val_size]:
        n1 = int(pair) // (10 ** num_digits)
        n2 = int(pair) % (10 ** num_digits)
        items.append({"question": f"{n1}+{n2}=", "answer": str(n1 + n2)})

    out = output_dir / "multi_digit_addition_gen.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ multi_digit_addition_gen.jsonl  ({len(items)} samples)")
    print()
    return True


# ---------------------------------------------------------------------------
# CUTE benchmark
# ---------------------------------------------------------------------------

def generate_cute(output_dir: Path) -> bool:
    """Download and save CUTE benchmark from HuggingFace."""
    print("=" * 80)
    print("Generating CUTE Dataset")
    print("=" * 80)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  Warning: 'datasets' library required — pip install datasets"); return False

    random.seed(42)
    all_task_types = [
        "spell", "spell_inverse", "contains_char", "contains_word",
        "orth", "sem", "ins_char", "ins_word", "del_char", "del_word",
        "sub_char", "sub_word", "swap_char", "swap_word",
    ]

    print("  Downloading CUTE from HuggingFace (leukas/cute)…")
    dataset = load_dataset("leukas/cute")

    samples = []
    for task_type in all_task_types:
        if task_type not in dataset:
            continue
        items = list(dataset[task_type])
        random.shuffle(items)
        for item in items[:100]:
            samples.append({"question": item["prompt"], "answer": item["answer"], "task_type": task_type})
        print(f"  ✓ {task_type}: {min(len(items), 100)} samples")

    out = output_dir / "cute_gen.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  ✓ cute_gen.jsonl  ({len(samples)} samples)")
    print()
    return True


# ---------------------------------------------------------------------------
# Post-processing: add IDs and generate MCQ↔GEN variants
# ---------------------------------------------------------------------------

def add_ids(output_dir: Path) -> None:
    """Add unique IDs to every sample in all JSONL benchmark files."""
    print("=" * 80)
    print("Adding Unique IDs")
    print("=" * 80)
    for fp in sorted(output_dir.glob("*.jsonl")):
        stem  = fp.stem                      # e.g. "langgame_mcq"
        parts = stem.rsplit("_", 1)
        bname = parts[0] if len(parts) == 2 else stem
        fmt   = parts[1] if len(parts) == 2 else "unknown"

        samples = [json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines() if l.strip()]
        if all("id" in s for s in samples):
            print(f"  ✓ {fp.name}  (IDs already present)")
            continue
        for idx, s in enumerate(samples):
            s["id"] = f"{bname}_{fmt}_{idx:05d}"
        with open(fp, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  ✓ {fp.name}  ({len(samples)} IDs added)")
    print()


def _gen_incorrect_answer(correct: str) -> str:
    """Generate a plausible-but-wrong numeric distractor for math MCQ."""
    val = int(correct)
    strategies = [
        lambda x: str(x + random.randint(1, 20)),
        lambda x: str(max(1, x - random.randint(1, 20))),
        lambda x: "".join(random.sample(list(str(x)), len(str(x)))) if len(str(x)) > 1 else str(x + 1),
        lambda x: str(x + 10) if x % 10 < 5 else str(x - 10),
    ]
    for _ in range(20):
        try:
            attempt = random.choice(strategies)(val)
            if attempt != correct and attempt.lstrip("-").isdigit():
                return attempt
        except Exception:
            pass
    return str(val + random.randint(1, 50))


def generate_variants(output_dir: Path) -> None:
    """
    Derive missing MCQ↔GEN variants:
      • langgame_gen    ← langgame_mcq    (strip options)
      • multi_digit_addition_mcq ← multi_digit_addition_gen (add distractors)
    """
    print("=" * 80)
    print("Generating Benchmark Variants")
    print("=" * 80)

    # langgame_gen
    src = output_dir / "langgame_mcq.jsonl"
    dst = output_dir / "langgame_gen.jsonl"
    if src.exists() and not dst.exists():
        samples = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        with open(dst, "w", encoding="utf-8") as f:
            for idx, s in enumerate(samples):
                f.write(json.dumps({
                    "question": s["question"],
                    "answer":   s["answer"],
                    "id":       f"langgame_gen_{idx:05d}",
                }, ensure_ascii=False) + "\n")
        print(f"  ✓ langgame_gen.jsonl  ({len(samples)} samples)")
    elif dst.exists():
        print(f"  ✓ langgame_gen.jsonl  (already exists)")

    # multi_digit_addition_mcq
    src = output_dir / "multi_digit_addition_gen.jsonl"
    dst = output_dir / "multi_digit_addition_mcq.jsonl"
    if src.exists() and not dst.exists():
        random.seed(42)
        samples = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        with open(dst, "w", encoding="utf-8") as f:
            for idx, s in enumerate(samples):
                correct = s["answer"]
                distractors: list[str] = []
                seen = {correct}
                attempts = 0
                while len(distractors) < 3 and attempts < 50:
                    d = _gen_incorrect_answer(correct)
                    if d not in seen:
                        distractors.append(d)
                        seen.add(d)
                    attempts += 1
                while len(distractors) < 3:
                    distractors.append(str(int(correct) + len(distractors) + 1))
                options = [f" {correct}"] + [f" {d}" for d in distractors]
                f.write(json.dumps({
                    "question": s["question"],
                    "answer":   f" {correct}",
                    "options":  options,
                    "id":       f"multi_digit_addition_mcq_{idx:05d}",
                }, ensure_ascii=False) + "\n")
        print(f"  ✓ multi_digit_addition_mcq.jsonl  ({len(samples)} samples)")
    elif dst.exists():
        print(f"  ✓ multi_digit_addition_mcq.jsonl  (already exists)")

    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all evaluation benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["pacute", "hierarchical", "langgame", "math", "cute", "all"],
        default=["all"],
        help="Which benchmarks to generate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/benchmarks",
        help="Output directory for benchmark JSONL files (default: data/benchmarks)",
    )
    parser.add_argument(
        "--corpora-dir",
        default="data/corpora",
        help="Root directory for corpus data files (default: data/corpora)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1859,
        help="Random seed for reproducible generation (default: 1859)",
    )
    args = parser.parse_args()

    output_dir  = Path(args.output_dir)
    corpora_dir = Path(args.corpora_dir)

    _ALL_BENCHMARKS = ["pacute", "hierarchical", "langgame", "math", "cute"]
    wanted = set(_ALL_BENCHMARKS if "all" in args.benchmarks else args.benchmarks)

    print("=" * 80)
    print("pacute-bench  –  Benchmark Generation")
    print("=" * 80)
    print(f"Output dir  : {output_dir}")
    print(f"Corpora dir : {corpora_dir}")
    print(f"Benchmarks  : {', '.join(sorted(wanted))}")
    print()

    ok, fail = 0, 0
    runners = {
        "pacute":       lambda: generate_pacute(output_dir, corpora_dir, args.random_seed),
        "hierarchical": lambda: generate_hierarchical(output_dir, corpora_dir),
        "langgame":     lambda: generate_langgame(output_dir, corpora_dir),
        "math":         lambda: generate_math(output_dir),
        "cute":         lambda: generate_cute(output_dir),
    }
    for name in _ALL_BENCHMARKS:
        if name not in wanted:
            continue
        try:
            result = runners[name]()
            if result:
                ok += 1
            else:
                fail += 1
        except Exception as exc:
            import traceback
            print(f"ERROR generating {name}: {exc}")
            traceback.print_exc()
            fail += 1

    if ok > 0:
        add_ids(output_dir)
        generate_variants(output_dir)

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  {ok} benchmark(s) generated successfully")
    if fail:
        print(f"  {fail} benchmark(s) failed")
    print("=" * 80)


if __name__ == "__main__":
    main()
