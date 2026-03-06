"""
Affixation Dataset Generation Module

This module provides functionality for creating linguistic datasets focused on
Filipino affixation patterns, including prefixes, suffixes, infixes, and circumfixes.
Supports both multiple-choice questions (MCQ) and generative (GEN) formats.
"""

import random
from typing import Dict, List, Any, Optional
import pandas as pd
import Levenshtein

from pacute_bench.utils.constants import (
    AFFIX_TYPES,
    MCQ_LABEL_MAP,
    NUM_MCQ_OPTIONS,
    NUM_INCORRECT_OPTIONS
)
from pacute_bench.utils.helpers import prepare_mcq_outputs, prepare_gen_outputs


def _get_affix_inflection_prompts(affix_type: str) -> tuple[str, str]:
    """
    Get the question prompts for affix inflection based on affix type.
    
    Args:
        affix_type: Type of affix ('prefix', 'suffix', 'infix', or 'circumfix')
    
    Returns:
        Tuple of (English prompt, Filipino prompt)
    
    Raises:
        ValueError: If affix_type is not valid
    """
    prompts = {
        "prefix": (
            'Which option has the prefix "{prefix}-"?',
            'Alin sa mga sumusunod ang may laping "{prefix}-"?'
        ),
        "suffix": (
            'Which option has the suffix "-{suffix}"?',
            'Alin sa mga sumusunod ang may laping "-{suffix}"?'
        ),
        "infix": (
            'Which option has the infix "-{infix}-"?',
            'Alin sa mga sumusunod ang may laping "-{infix}-"?'
        ),
        "circumfix": (
            'Which option has the circumfix "{prefix}-" and "-{suffix}"?',
            'Alin sa mga sumusunod ang may laping "{prefix}-" at "-{suffix}"?'
        )
    }
    
    if affix_type not in prompts:
        raise ValueError(f"Invalid affix type: {affix_type}. Choose from {', '.join(AFFIX_TYPES)}.")
    
    return prompts[affix_type]


def _get_gen_inflection_prompts(affix_type: str) -> tuple[str, str]:
    """
    Get the generative prompts for affix inflection based on affix type.
    
    Args:
        affix_type: Type of affix ('prefix', 'suffix', 'infix', or 'circumfix')
    
    Returns:
        Tuple of (English prompt, Filipino prompt)
    
    Raises:
        ValueError: If affix_type is not valid
    """
    prompts = {
        "prefix": (
            'Inflect the word "{root}" to use the prefix "{prefix}-".',
            'Lapian ng "{prefix}-" ang salitang "{root}".'
        ),
        "suffix": (
            'Inflect the word "{root}" to use the suffix "-{suffix}".',
            'Lapian ng "-{suffix}" ang salitang "{root}".'
        ),
        "infix": (
            'Inflect the word "{root}" to use the infix "-{infix}-".',
            'Lapian ng "-{infix}-" ang salitang "{root}".'
        ),
        "circumfix": (
            'Inflect the word "{root}" to use the circumfix "{prefix}-" and "-{suffix}".',
            'Lapian ng "{prefix}-" at "-{suffix}" ang salitang "{root}".'
        )
    }
    
    if affix_type not in prompts:
        raise ValueError(f"Invalid affix type: {affix_type}. Choose from {', '.join(AFFIX_TYPES)}.")
    
    return prompts[affix_type]


def create_mcq_affix_inflection(row: pd.Series, affix_type: str) -> Dict[str, Any]:
    """
    Create a multiple-choice question for affix inflection.
    
    Args:
        row: DataFrame row containing inflection data with correct/incorrect options
        affix_type: Type of affix to test ('prefix', 'suffix', 'infix', or 'circumfix')
    
    Returns:
        Dictionary containing formatted MCQ prompts and options
    """
    text_en, text_tl = _get_affix_inflection_prompts(affix_type)

    mcq_options = {
        "correct": row["correct"],
        "incorrect1": row["incorrect1"],
        "incorrect2": row["incorrect2"],
        "incorrect3": row["incorrect3"],
    }

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs={})


def create_gen_affix_inflection(row: pd.Series, affix_type: str) -> Dict[str, Any]:
    """
    Create a generative question for affix inflection.
    
    Args:
        row: DataFrame row containing inflection data with root word and correct form
        affix_type: Type of affix to apply ('prefix', 'suffix', 'infix', or 'circumfix')
    
    Returns:
        Dictionary containing formatted generative prompts and label
    """
    text_en, text_tl = _get_gen_inflection_prompts(affix_type)
    
    label = row["correct"]
    return prepare_gen_outputs(text_en, text_tl, label, row=row)


def find_similar_incorrect_affixes(affix: str, unique_affixes: List[str], top_k: int = 3) -> List[str]:
    """
    Find the most similar affixes using Levenshtein distance for creating distractors.
    
    This function computes the Levenshtein ratio between the target affix and all
    other unique affixes, then returns the top-k most similar ones to use as
    plausible incorrect options in multiple-choice questions.
    
    Args:
        affix: The target affix to find similar affixes for
        unique_affixes: List of all available affixes to compare against
        top_k: Number of similar affixes to return (default: 3)
    
    Returns:
        List of the top-k most similar affixes (excluding the target affix itself)
    """
    levenshtein_ratios = {
        ua: Levenshtein.ratio(affix, ua) 
        for ua in unique_affixes 
        if ua != affix
    }
    sorted_affixes = sorted(
        levenshtein_ratios, 
        key=levenshtein_ratios.get, 
        reverse=True
    )
    return sorted_affixes[:top_k]


def create_mcq_affix_identification(row: pd.Series, affix_type: str) -> Dict[str, Any]:
    """
    Create a multiple-choice question for identifying the affix used in a word.
    
    This is the reverse task of affix inflection - given an inflected word,
    identify which affix was used to create it.
    
    Args:
        row: DataFrame row containing the inflected word and incorrect affix options
        affix_type: Type of affix to identify ('prefix', 'suffix', or 'infix')
    
    Returns:
        Dictionary containing formatted MCQ prompts and options
    """
    text_en = 'Which option is the affix used to inflect the word "{correct}"?'
    text_tl = 'Alin sa sumusunod ang lapi na ginamit sa salitang "{correct}"?'

    mcq_options = {
        "correct": row[affix_type],
        "incorrect1": row["incorrect_affixes"][0],
        "incorrect2": row["incorrect_affixes"][1],
        "incorrect3": row["incorrect_affixes"][2],
    }

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs={})


def create_gen_affix_identification(row: pd.Series, affix_type: str) -> Dict[str, Any]:
    """
    Create a generative question for identifying the affix used in a word.
    
    This is the reverse task of affix inflection - given an inflected word,
    the model must generate which affix was used to create it.
    
    Args:
        row: DataFrame row containing the inflected word and the correct affix
        affix_type: Type of affix to identify ('prefix', 'suffix', or 'infix')
    
    Returns:
        Dictionary containing formatted generative prompts and label
    """
    text_en = 'What is the affix used to inflect the word "{correct}"?'
    text_tl = 'Ano ang lapi na ginamit sa salitang "{correct}"?'

    label = row[affix_type]
    return prepare_gen_outputs(text_en, text_tl, label, row=row)


def _extract_unique_affixes(inflections_df: pd.DataFrame) -> List[str]:
    """
    Extract all unique affixes from the inflections dataframe.
    
    Args:
        inflections_df: DataFrame containing inflection data with prefix, suffix, and infix columns
    
    Returns:
        List of all unique affixes across all affix types
    """
    unique_affixes = (
        inflections_df["prefix"].unique().tolist() + 
        inflections_df["suffix"].unique().tolist() + 
        inflections_df["infix"].unique().tolist()
    )
    return unique_affixes


def _shuffle_mcq_options(dataset: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Shuffle MCQ options and assign labels to the dataset.
    
    This function randomizes the position of the correct answer among the four
    choices and assigns corresponding labels (A, B, C, D) to each question.
    
    Args:
        dataset: DataFrame containing MCQ questions with mcq_options
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Modified dataset with shuffled choices and assigned labels
    """
    random.seed(random_seed)
    
    for i in range(len(dataset)):
        # Distribute correct answers evenly across positions
        label_index = i % NUM_MCQ_OPTIONS
        
        correct = dataset.iloc[i]['prompts'][0]["mcq_options"]['correct']
        options = [
            dataset.iloc[i]['prompts'][0]["mcq_options"]['incorrect1'],
            dataset.iloc[i]['prompts'][0]["mcq_options"]['incorrect2'],
            dataset.iloc[i]['prompts'][0]["mcq_options"]['incorrect3'],
        ]
        
        # Shuffle incorrect options and insert correct answer at label_index
        random.shuffle(options)
        options.insert(label_index, correct)
        
        choices = {
            "choice1": options[0],
            "choice2": options[1],
            "choice3": options[2],
            "choice4": options[3],
        }
        
        label = MCQ_LABEL_MAP[label_index]
        dataset.at[i, 'prompts'][0].update(choices)
        dataset.at[i, 'label'] = label
    
    return dataset


def _create_mcq_inflection_questions(inflections_df: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Create MCQ questions for affix inflection tasks.
    
    Args:
        inflections_df: DataFrame containing inflection data
        dataset: Existing dataset to append questions to
    
    Returns:
        Dataset with added inflection questions
    """
    for _, row in inflections_df.iterrows():
        affix_type = row["affix_type"]
        outputs = create_mcq_affix_inflection(row, affix_type)
        dataset = pd.concat([dataset, pd.DataFrame([{
            "category": "affix_inflection",
            "subcategory": affix_type,
            "prompts": outputs["prompts"],
        }])], ignore_index=True)
    
    return dataset


def _create_mcq_identification_questions(inflections_df: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Create MCQ questions for affix identification tasks (reverse of inflection).
    
    Args:
        inflections_df: DataFrame containing inflection data
        dataset: Existing dataset to append questions to
    
    Returns:
        Dataset with added identification questions
    """
    unique_affixes = _extract_unique_affixes(inflections_df)
    
    for _, row in inflections_df.iterrows():
        affix_type = row["affix_type"]
        
        # Skip circumfix for identification tasks
        if affix_type == "circumfix":
            continue
        
        row_copy = row.copy()
        row_copy["incorrect_affixes"] = find_similar_incorrect_affixes(
            row[affix_type], 
            unique_affixes
        )
        outputs = create_mcq_affix_identification(row_copy, affix_type)
        dataset = pd.concat([dataset, pd.DataFrame([{
            "category": "affix_identification",
            "subcategory": affix_type,
            "prompts": outputs["prompts"],
        }])], ignore_index=True)
    
    return dataset


def _create_gen_inflection_questions(inflections_df: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Create generative questions for affix inflection tasks.
    
    Args:
        inflections_df: DataFrame containing inflection data
        dataset: Existing dataset to append questions to
    
    Returns:
        Dataset with added inflection questions
    """
    for _, row in inflections_df.iterrows():
        affix_type = row["affix_type"]
        outputs = create_gen_affix_inflection(row, affix_type)
        dataset = pd.concat([dataset, pd.DataFrame([{
            "category": "affix_inflection",
            "subcategory": affix_type,
            "prompts": outputs["prompts"],
            "label": outputs["label"]
        }])], ignore_index=True)
    
    return dataset


def _create_gen_identification_questions(inflections_df: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Create generative questions for affix identification tasks.
    
    Args:
        inflections_df: DataFrame containing inflection data
        dataset: Existing dataset to append questions to
    
    Returns:
        Dataset with added identification questions
    """
    for _, row in inflections_df.iterrows():
        affix_type = row["affix_type"]
        
        # Skip circumfix for identification tasks
        if affix_type == "circumfix":
            continue
        
        outputs = create_gen_affix_identification(row, affix_type)
        dataset = pd.concat([dataset, pd.DataFrame([{
            "category": "affix_identification",
            "subcategory": affix_type,
            "prompts": outputs["prompts"],
            "label": outputs["label"]
        }])], ignore_index=True)
    
    return dataset


def create_affixation_dataset(
    inflections_df: pd.DataFrame, 
    mode: str = 'mcq', 
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Create a complete affixation dataset with multiple question types.
    
    This is the main function that generates a comprehensive dataset for testing
    Filipino affixation knowledge. It creates two types of tasks:
    1. Affix Inflection: Given a root word and affix, produce/identify the inflected form
    2. Affix Identification: Given an inflected word, identify which affix was used
    
    Args:
        inflections_df: DataFrame containing inflection data with columns:
            - root: The base/root word
            - affix_type: Type of affix (prefix, suffix, infix, circumfix)
            - prefix/suffix/infix: The specific affix used
            - correct: The correctly inflected form
            - incorrect1-3: Incorrect options for MCQ mode
        mode: Question format - 'mcq' for multiple-choice or 'gen' for generative (default: 'mcq')
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        DataFrame with columns:
            - category: Task type (affix_inflection or affix_identification)
            - subcategory: Affix type (prefix, suffix, infix, circumfix)
            - prompts: List containing question prompts in English and Filipino
            - label: Correct answer (A/B/C/D for MCQ, actual answer for GEN)
    
    Raises:
        ValueError: If mode is not 'mcq' or 'gen'
    
    Examples:
        >>> inflections = pd.read_excel("inflections.xlsx")
        >>> mcq_dataset = create_affixation_dataset(inflections, mode='mcq')
        >>> gen_dataset = create_affixation_dataset(inflections, mode='gen')
    """
    random.seed(random_seed)

    dataset = pd.DataFrame(columns=["category", "subcategory", "prompts", "label"])

    if mode == 'mcq':
        # Generate inflection and identification questions
        dataset = _create_mcq_inflection_questions(inflections_df, dataset)
        dataset = _create_mcq_identification_questions(inflections_df, dataset)
        
        # Shuffle options and assign labels
        dataset = _shuffle_mcq_options(dataset, random_seed)

    elif mode == 'gen':
        # Generate inflection and identification questions
        dataset = _create_gen_inflection_questions(inflections_df, dataset)
        dataset = _create_gen_identification_questions(inflections_df, dataset)
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'mcq' or 'gen'.")

    return dataset
