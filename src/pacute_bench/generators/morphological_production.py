"""
Morphological Production Dataset Generation Module

Covers one task:
  - inflected_form_production: given root word + affix → produce inflected form

Data source: inflected_affix_extraction and inflected_root_extraction rows from
corpus_morphological_extraction.jsonl (word, root, affix, affix_type present; label = word).

Prompts are generated (no pre-built prompts in corpus for this task).
MCQ distractors are character-level perturbations of the correct inflected form.
"""

import random
from typing import Any, Dict, Optional

import pandas as pd

from pacute_bench.utils.constants import MCQ_LABEL_MAP, NUM_MCQ_OPTIONS
from pacute_bench.utils.helpers import prepare_gen_outputs, prepare_mcq_outputs
from pacute_bench.utils.strings import word_perturbation_distractors


# Mapping from internal affix_type values to display labels in prompts
_AFFIX_TYPE_EN: Dict[str, str] = {
    "prefix":        "prefix",
    "suffix":        "suffix",
    "infix":         "infix",
    "circumfix":     "circumfix",
    "infix & suffix": "infix and suffix",
    "prefix & infix": "prefix and infix",
}


def _affix_type_label(affix_type: str) -> str:
    return _AFFIX_TYPE_EN.get(str(affix_type), str(affix_type))


def _make_prompts(root: str, affix: str, affix_type: str):
    type_label = _affix_type_label(affix_type)
    text_en = f'Inflect the word "{root}" using the {type_label} "{affix}".'
    text_tl = f'Lapian ng panlaping "{affix}" ang salitang "{root}".'
    return text_en, text_tl


# ---------------------------------------------------------------------------
# GEN / MCQ creators
# ---------------------------------------------------------------------------

def _create_gen_production(row: pd.Series) -> Dict[str, Any]:
    text_en, text_tl = _make_prompts(
        str(row["root"]), str(row["affix"]), str(row.get("affix_type", ""))
    )
    return prepare_gen_outputs(text_en, text_tl, str(row["word"]))


def _create_mcq_production(row: pd.Series, rng: random.Random) -> Optional[Dict[str, Any]]:
    correct = str(row["word"])
    distractors = word_perturbation_distractors(correct, rng=rng)
    text_en, text_tl = _make_prompts(
        str(row["root"]), str(row["affix"]), str(row.get("affix_type", ""))
    )
    mcq_options = {
        "correct":    correct,
        "incorrect1": distractors[0],
        "incorrect2": distractors[1],
        "incorrect3": distractors[2],
    }
    return prepare_mcq_outputs(text_en, text_tl, mcq_options)


# ---------------------------------------------------------------------------
# Shuffle helpers
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
# Main
# ---------------------------------------------------------------------------

def create_morphological_production_dataset(
    corpus_df: pd.DataFrame,
    mode: str = "gen",
    random_seed: int = 42,
    num_samples: int = 100,
) -> pd.DataFrame:
    """
    Create morphological production benchmark items from corpus.

    Uses both inflected_affix_extraction and inflected_root_extraction rows,
    deduplicated on (word, root, affix), then 150 sampled from the unique pool.

    Args:
        corpus_df: DataFrame loaded from corpus_morphological_extraction.jsonl
        mode: 'gen' or 'mcq'
        random_seed: Random seed for reproducibility
        num_samples: Unused; sampling is fixed at min(150, unique pool size)

    Returns:
        DataFrame with columns: category, subcategory, prompts, label
    """
    rng = random.Random(random_seed)
    dataset = pd.DataFrame(columns=["category", "subcategory", "prompts", "label"])

    src_affix = corpus_df[corpus_df["subcategory"] == "inflected_affix_extraction"].reset_index(drop=True)
    src_root  = corpus_df[corpus_df["subcategory"] == "inflected_root_extraction"].reset_index(drop=True)

    # Dedup full pool first (the two sources share many identical (word,root,affix) triples),
    # then sample up to 150 from the unique set.
    unique_pool = (
        pd.concat([src_affix, src_root], ignore_index=True)
        .drop_duplicates(subset=["word", "root", "affix"])
        .reset_index(drop=True)
    )
    src_df = unique_pool.sample(min(150, len(unique_pool)), random_state=random_seed).reset_index(drop=True)

    for _, row in src_df.iterrows():
        if mode == "gen":
            result = _create_gen_production(row)
            dataset = pd.concat([dataset, pd.DataFrame({
                "category":    ["morphological_production"],
                "subcategory": ["inflected_form_production"],
                "prompts":     [result["prompts"]],
                "label":       [result["label"]],
            })], ignore_index=True)
        else:
            result = _create_mcq_production(row, rng)
            if result is not None:
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category":    ["morphological_production"],
                    "subcategory": ["inflected_form_production"],
                    "prompts":     [result["prompts"]],
                })], ignore_index=True)

    if mode == "mcq":
        dataset = _shuffle_mcq_options(dataset, random_seed)

    return dataset
