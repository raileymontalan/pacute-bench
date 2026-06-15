"""
Morphological Extraction Dataset Generation Module

Covers four tasks driven by corpus_morphological_extraction.jsonl:
  - inflected_affix_extraction   : given inflected word → identify affix
  - inflected_root_extraction    : given inflected word → identify root
  - reduplicated_root_extraction : given reduplicated word → identify root
  - reduplicant_extraction       : given reduplicated word → identify reduplicant
                                   (GEN includes etymological/none rows;
                                    MCQ excludes them)

All GEN items use the pre-built text_en/text_tl prompts from the corpus.
MCQ items use the same prompts with generated distractor options.
"""

import random
from typing import Any, Dict, List, Optional

import Levenshtein
import pandas as pd

from pacute_bench.utils.constants import MCQ_LABEL_MAP, NUM_MCQ_OPTIONS
from pacute_bench.utils.helpers import prepare_gen_outputs, prepare_mcq_outputs
from pacute_bench.utils.strings import word_perturbation_distractors


# ---------------------------------------------------------------------------
# Distractor helpers
# ---------------------------------------------------------------------------

def _levenshtein_distractors(target: str, pool: List[str], k: int = 3) -> List[str]:
    """Return k affixes most similar to target by Levenshtein ratio (excluding target)."""
    candidates = [s for s in pool if s != target]
    ranked = sorted(candidates, key=lambda s: Levenshtein.ratio(target, s), reverse=True)
    return ranked[:k]


def _pad_distractors(distractors: List[str], correct: str, fallback_pool: List[str], k: int = 3) -> List[str]:
    """Ensure exactly k distractors, padding from fallback_pool if needed."""
    seen = {correct} | set(distractors)
    result = list(distractors[:k])
    for item in fallback_pool:
        if len(result) >= k:
            break
        if item not in seen:
            result.append(item)
            seen.add(item)
    i = 1
    while len(result) < k:
        placeholder = f"{correct}_alt{i}"
        if placeholder not in seen:
            result.append(placeholder)
            seen.add(placeholder)
        i += 1
    return result


# ---------------------------------------------------------------------------
# GEN item creators (passthrough from corpus prompts)
# ---------------------------------------------------------------------------

def _create_gen_item(row: pd.Series, subcategory: str, label_field: str) -> Dict[str, Any]:
    label = str(row[label_field]) if row.get(label_field) is not None else "none"
    return prepare_gen_outputs(str(row["text_en"]), str(row["text_tl"]), label)


# ---------------------------------------------------------------------------
# MCQ item creators
# ---------------------------------------------------------------------------

def _build_mcq_row(row: pd.Series, correct: str, distractors: List[str]) -> Dict[str, Any]:
    mcq_options = {
        "correct":    correct,
        "incorrect1": distractors[0],
        "incorrect2": distractors[1],
        "incorrect3": distractors[2],
    }
    return prepare_mcq_outputs(str(row["text_en"]), str(row["text_tl"]), mcq_options)


def _create_mcq_affix(row: pd.Series, affix_pool: List[str]) -> Optional[Dict[str, Any]]:
    correct = str(row["affix"])
    distractors = _levenshtein_distractors(correct, affix_pool)
    if len(distractors) < 3:
        distractors = _pad_distractors(distractors, correct, affix_pool)
    return _build_mcq_row(row, correct, distractors)


def _create_mcq_root(row: pd.Series, rng: random.Random) -> Optional[Dict[str, Any]]:
    correct = str(row["root"])
    distractors = word_perturbation_distractors(correct, rng=rng)
    return _build_mcq_row(row, correct, distractors)


def _create_mcq_reduplicant(row: pd.Series, rng: random.Random) -> Optional[Dict[str, Any]]:
    correct = str(row["reduplicant"])
    distractors = word_perturbation_distractors(correct, rng=rng)
    return _build_mcq_row(row, correct, distractors)


# ---------------------------------------------------------------------------
# Shuffle MCQ options and assign A/B/C/D labels
# ---------------------------------------------------------------------------

def _shuffle_mcq_options(dataset: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    rng = random.Random(random_seed)
    for i in range(len(dataset)):
        label_index = i % NUM_MCQ_OPTIONS
        correct = dataset.iloc[i]["prompts"][0]["mcq_options"]["correct"]
        options = [
            dataset.iloc[i]["prompts"][0]["mcq_options"]["incorrect1"],
            dataset.iloc[i]["prompts"][0]["mcq_options"]["incorrect2"],
            dataset.iloc[i]["prompts"][0]["mcq_options"]["incorrect3"],
        ]
        rng.shuffle(options)
        options.insert(label_index, correct)
        choices = {f"choice{j+1}": options[j] for j in range(4)}
        dataset.at[i, "prompts"][0].update(choices)
        dataset.at[i, "label"] = MCQ_LABEL_MAP[label_index]
    return dataset


# ---------------------------------------------------------------------------
# Main dataset creation function
# ---------------------------------------------------------------------------

def create_morphological_extraction_dataset(
    corpus_df: pd.DataFrame,
    mode: str = "gen",
    random_seed: int = 42,
    num_samples: int = 100,
) -> pd.DataFrame:
    """
    Create morphological extraction benchmark items from corpus.

    Args:
        corpus_df: DataFrame loaded from corpus_morphological_extraction.jsonl
        mode: 'gen' or 'mcq'
        random_seed: Random seed for reproducibility
        num_samples: Max samples per subcategory (default 100)

    Returns:
        DataFrame with columns: category, subcategory, prompts, label
    """
    rng = random.Random(random_seed)
    dataset = pd.DataFrame(columns=["category", "subcategory", "prompts", "label"])

    def _strat(df: pd.DataFrame, col: str, strata: list) -> pd.DataFrame:
        """Sample stratified by col; strata = [(value_or_list, n), ...]."""
        frames = []
        for bucket, n in strata:
            mask = df[col].isin(bucket) if isinstance(bucket, list) else (df[col] == bucket)
            s = df[mask].reset_index(drop=True)
            take = min(n, len(s))
            if take > 0:
                frames.append(s.sample(take, random_state=random_seed))
        return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0]

    # Pre-build distractor pools from full corpus before any sampling
    affix_pool = sorted(set(
        str(r["affix"])
        for _, r in corpus_df[
            (corpus_df["subcategory"] == "inflected_affix_extraction") &
            (corpus_df["subcategory2"].isin(["prefix", "infix", "suffix"]))
        ].iterrows()
        if r.get("affix") is not None
    ))

    # ---- inflected_affix_extraction: prefix/infix/suffix only, 34/33/33 = 100
    iae_full = corpus_df[
        (corpus_df["subcategory"] == "inflected_affix_extraction") &
        (corpus_df["subcategory2"].isin(["prefix", "infix", "suffix"]))
    ].reset_index(drop=True)
    sub_df = _strat(iae_full, "subcategory2", [
        ("prefix", 34), ("infix", 33), ("suffix", 33),
    ])
    for _, row in sub_df.iterrows():
        if mode == "gen":
            result = _create_gen_item(row, "inflected_affix_extraction", "affix")
            dataset = pd.concat([dataset, pd.DataFrame({
                "category":    ["morphological_extraction"],
                "subcategory": ["inflected_affix_extraction"],
                "prompts":     [result["prompts"]],
                "label":       [result["label"]],
            })], ignore_index=True)
        else:
            result = _create_mcq_affix(row, affix_pool)
            if result is not None:
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category":    ["morphological_extraction"],
                    "subcategory": ["inflected_affix_extraction"],
                    "prompts":     [result["prompts"]],
                })], ignore_index=True)

    # ---- inflected_root_extraction: 100 rows
    sub_df = corpus_df[corpus_df["subcategory"] == "inflected_root_extraction"].reset_index(drop=True)
    sub_df = sub_df.sample(min(100, len(sub_df)), random_state=random_seed)
    for _, row in sub_df.iterrows():
        if mode == "gen":
            result = _create_gen_item(row, "inflected_root_extraction", "root")
            dataset = pd.concat([dataset, pd.DataFrame({
                "category":    ["morphological_extraction"],
                "subcategory": ["inflected_root_extraction"],
                "prompts":     [result["prompts"]],
                "label":       [result["label"]],
            })], ignore_index=True)
        else:
            result = _create_mcq_root(row, rng)
            if result is not None:
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category":    ["morphological_extraction"],
                    "subcategory": ["inflected_root_extraction"],
                    "prompts":     [result["prompts"]],
                })], ignore_index=True)

    # ---- reduplicated_root_extraction: 34 partial, 33 full, 33 none ----------
    sub_df = _strat(
        corpus_df[corpus_df["subcategory"] == "reduplicated_root_extraction"].reset_index(drop=True),
        "redup_type",
        [("partial", 34), ("full", 33), ("none", 33)],
    )
    for _, row in sub_df.iterrows():
        if mode == "gen":
            result = _create_gen_item(row, "reduplicated_root_extraction", "root")
            dataset = pd.concat([dataset, pd.DataFrame({
                "category":    ["morphological_extraction"],
                "subcategory": ["reduplicated_root_extraction"],
                "prompts":     [result["prompts"]],
                "label":       [result["label"]],
            })], ignore_index=True)
        else:
            result = _create_mcq_root(row, rng)
            if result is not None:
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category":    ["morphological_extraction"],
                    "subcategory": ["reduplicated_root_extraction"],
                    "prompts":     [result["prompts"]],
                })], ignore_index=True)

    # ---- reduplicant_extraction ----------------------------------------------
    all_redup = corpus_df[corpus_df["subcategory"] == "reduplicant_extraction"].reset_index(drop=True)
    sub_df = _strat(all_redup, "redup_type", [("partial", 34), ("full", 33), ("none", 33)])

    for _, row in sub_df.iterrows():
        redup_type = row.get("redup_type")

        if mode == "gen":
            result = _create_gen_item(row, "reduplicant_extraction", "reduplicant")
            dataset = pd.concat([dataset, pd.DataFrame({
                "category":    ["morphological_extraction"],
                "subcategory": ["reduplicant_extraction"],
                "prompts":     [result["prompts"]],
                "label":       [result["label"]],
            })], ignore_index=True)
        else:
            result = _create_mcq_reduplicant(row, rng)
            if result is not None:
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category":    ["morphological_extraction"],
                    "subcategory": ["reduplicant_extraction"],
                    "prompts":     [result["prompts"]],
                })], ignore_index=True)

    if mode == "mcq":
        dataset = _shuffle_mcq_options(dataset, random_seed)

    return dataset
