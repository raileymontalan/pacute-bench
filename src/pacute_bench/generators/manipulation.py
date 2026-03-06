"""
String Manipulation Dataset Generation Module

This module provides functionality for creating linguistic datasets focused on
Filipino string manipulation tasks, including character operations (insertion,
deletion, substitution, permutation, duplication) and case transformations.
Supports both multiple-choice questions (MCQ) and generative (GEN) formats.
"""

import random
from typing import Dict, List, Any, Optional, Tuple, Callable
import pandas as pd
from pacute_bench.utils.strings import (
    string_to_chars, chars_to_string, get_random_char,
    delete_char, insert_char, substitute_char, permute_char, duplicate_char,
    normalize_diacritic, diacritize, randomly_diacritize, same_string
)
from pacute_bench.utils.constants import (
    MCQ_LABEL_MAP,
    NUM_MCQ_OPTIONS,
    NUM_INCORRECT_OPTIONS,
    MIN_WORD_LENGTH_MANIPULATION
)
from pacute_bench.utils.helpers import prepare_mcq_outputs, prepare_gen_outputs

# Manipulation function mappings
MANIPULATIONS: Dict[str, Callable] = {
    "none": same_string,
    "deletion": delete_char,
    "insertion": insert_char,
    "substitution": substitute_char,
    "permutation": permute_char,
    "duplication": duplicate_char,
}


# ============================================================================
# Manipulation Helper Functions
# ============================================================================

def get_invalid_manipulations(target_manipulation: str = "insertion") -> List[Tuple[str, Callable]]:
    """
    Get all manipulation functions except the target one.
    
    Used to create plausible incorrect options by applying different
    manipulation operations.
    
    Args:
        target_manipulation: The manipulation to exclude (default: "insertion")
    
    Returns:
        List of tuples containing (manipulation_name, manipulation_function)
    """
    return [(name, func) for name, func in MANIPULATIONS.items() if name != target_manipulation]


def apply_manipulation_incorrectly(
    string: str,
    target_manipulation: str = "deletion",
    kwargs: Optional[Dict[str, Any]] = None
) -> str:
    """
    Apply the target manipulation incorrectly to create plausible distractors.
    
    This function applies the correct manipulation type but with wrong parameters
    (e.g., deleting the wrong character) to create believable incorrect options.
    
    Args:
        string: Input string to manipulate
        target_manipulation: Type of manipulation to apply incorrectly
        kwargs: Parameters for the manipulation (contains the correct parameters)
    
    Returns:
        String with the manipulation applied incorrectly
    """
    if kwargs is None:
        kwargs = {}

    if target_manipulation == "deletion" and "char_to_delete" in kwargs:
        incorrect_char = kwargs["char_to_delete"]
        remaining_chars = string.replace(incorrect_char, '')
        if remaining_chars:
            incorrect_char = get_random_char(remaining_chars)
        return delete_char(string, char_to_delete=incorrect_char)
    elif target_manipulation == "insertion" and "preceding_char" in kwargs and "char_to_insert" in kwargs:
        preceding_char = kwargs["preceding_char"]
        char_to_insert = kwargs["char_to_insert"]
        remaining_chars = 'abcdefghijklmnopqrstuvwxyz'.replace(char_to_insert, '')
        incorrect_char_to_insert = get_random_char(remaining_chars)
        return insert_char(string, preceding_char=preceding_char, char_to_insert=incorrect_char_to_insert)
    elif target_manipulation == "substitution" and "char_to_replace" in kwargs and "char_to_substitute" in kwargs:
        char_to_replace = kwargs["char_to_replace"]
        char_to_substitute = kwargs["char_to_substitute"]
        remaining_chars = 'abcdefghijklmnopqrstuvwxyz'.replace(char_to_replace, '')
        remaining_chars = remaining_chars.replace(char_to_substitute, '')
        incorrect_char_to_substitute = get_random_char(remaining_chars)
        return substitute_char(string, char_to_replace=char_to_replace, char_to_substitute=incorrect_char_to_substitute)
    elif target_manipulation == "permutation" and "char1" in kwargs and "char2" in kwargs:
        char1 = kwargs["char1"]
        char2 = kwargs["char2"]
        remaining_chars = string.replace(char1, '')
        remaining_chars = remaining_chars.replace(char2, '')
        # Fallback to alphabet if no characters left in string
        if not remaining_chars:
            remaining_chars = 'abcdefghijklmnopqrstuvwxyz'.replace(char1, '').replace(char2, '')
        incorrect_char2 = get_random_char(remaining_chars)
        return permute_char(string, char1=char1, char2=incorrect_char2)
    elif target_manipulation == "duplication" and "char_to_duplicate" in kwargs:
        char_to_duplicate = kwargs["char_to_duplicate"]
        remaining_chars = string.replace(char_to_duplicate, '')
        if remaining_chars:
            incorrect_char_to_duplicate = get_random_char(remaining_chars)
        else:
            incorrect_char_to_duplicate = char_to_duplicate
        return duplicate_char(string, char_to_duplicate=incorrect_char_to_duplicate)


def manipulate_string(
    string: str,
    target_manipulation: str = "deletion",
    kwargs: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Create incorrect manipulation options for MCQ questions.
    
    Generates three incorrect options:
    - Two from applying different manipulation types
    - One from applying the correct type incorrectly
    
    Args:
        string: Input string to manipulate
        target_manipulation: The correct manipulation type
        kwargs: Parameters for the correct manipulation
    
    Returns:
        List of three incorrectly manipulated strings
    """
    if kwargs is None:
        kwargs = {}

    manipulation_functions = get_invalid_manipulations(target_manipulation=target_manipulation)
    chosen_functions = random.sample([func for name, func in manipulation_functions], 2)
    results = [func(string) for func in chosen_functions]

    incorrect_application = apply_manipulation_incorrectly(
        string, 
        target_manipulation=target_manipulation, 
        kwargs=kwargs
    )
    results.append(incorrect_application)
    return results


def diacritize_string(string: str, correct_string: str) -> List[str]:
    """
    Create incorrect diacritic options for MCQ questions.
    
    Generates three plausible incorrect options for diacritic normalization:
    - Original string (no normalization)
    - String with added diacritics
    - String with random diacritics
    
    Args:
        string: Input string
        correct_string: The correctly normalized string
    
    Returns:
        List of three incorrect diacritization options
    """
    results = [
        same_string(string),
        diacritize(string),
        randomly_diacritize(string),
    ]

    # If correct answer accidentally included, replace it
    if correct_string in results:
        results.remove(correct_string)
        from pacute_bench.utils.strings import shuffle_chars
        results.append(chars_to_string(shuffle_chars(string_to_chars(correct_string))))
    
    return results


# ============================================================================
# MCQ Question Creation Functions
# ============================================================================

def create_mcq_deletion(row: pd.Series) -> Dict[str, Any]:
    """
    Create MCQ for character deletion operation.
    
    Args:
        row: DataFrame row containing word data
    
    Returns:
        Dictionary containing formatted MCQ prompts and options
    """
    text_en = 'Which option correctly removes every character "{char_to_delete}" in the word "{normalized_word}"?'
    text_tl = 'Alin sa sumusunod ang nagtatanggal ng lahat ng titik na "{char_to_delete}" mula sa salitang "{normalized_word}"?'

    string = row['normalized_word']
    char_to_delete = get_random_char(string)
    kwargs = {"char_to_delete": char_to_delete}

    mcq_correct = delete_char(string, **kwargs)
    mcq_incorrect = manipulate_string(string, target_manipulation="deletion", kwargs=kwargs)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs=kwargs)


def create_mcq_insertion(row):
    text_en = 'Which option correctly puts the character "{char_to_insert}" after every character "{preceding_char}" in the word "{normalized_word}"?'
    text_tl = 'Alin sa sumusunod ang naglalagay ng titik na "{char_to_insert}" pagkatapos ng bawat titik na "{preceding_char}" sa salitang "{normalized_word}"?'

    string = row['normalized_word']
    preceding_char = get_random_char(string)
    char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz')
    kwargs = {"preceding_char": preceding_char, "char_to_insert": char_to_insert}

    mcq_correct = insert_char(string, **kwargs)
    mcq_incorrect = manipulate_string(string, target_manipulation="insertion", kwargs=kwargs)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs=kwargs)
    return outputs


def create_mcq_substitution(row):
    text_en = 'Which option correctly replaces every character "{char_to_replace}" with the character "{char_to_substitute}" in the word "{normalized_word}"?'
    text_tl = 'Alin sa sumusunod ang pumapalit sa bawat titik na "{char_to_replace}" gamit ang titik na "{char_to_substitute}" sa salitang "{normalized_word}"?'

    string = row['normalized_word']
    char_to_replace = get_random_char(string)
    remaining_chars = 'abcdefghijklmnopqrstuvwxyz'.replace(char_to_replace, '')
    char_to_substitute = get_random_char(remaining_chars)
    kwargs = {"char_to_replace": char_to_replace, "char_to_substitute": char_to_substitute}

    mcq_correct = substitute_char(string, **kwargs)
    mcq_incorrect = manipulate_string(string, target_manipulation="substitution", kwargs=kwargs)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs=kwargs)
    return outputs


def create_mcq_permutation(row):
    text_en = 'Which option correctly swaps every character "{char1}" and character "{char2}" in the word "{normalized_word}"?'
    text_tl = 'Alin sa sumusunod ang pinagpapalit ang bawat titik na "{char1}" at titik na "{char2}" sa salitang "{normalized_word}"?'

    string = row['normalized_word']
    char1 = get_random_char(string)
    remaining_string = string.replace(char1, '')
    char2 = get_random_char(remaining_string)
    kwargs = {"char1": char1, "char2": char2}

    mcq_correct = permute_char(string, **kwargs)
    mcq_incorrect = manipulate_string(string, target_manipulation="permutation", kwargs=kwargs)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs=kwargs)
    return outputs


def create_mcq_duplication(row):
    text_en = 'Which option correctly duplicates every character "{char_to_duplicate}" once in the word "{normalized_word}"?'
    text_tl = 'Alin sa sumusunod ang umuulit sa bawat titik na "{char_to_duplicate}" nang isang beses sa salitang "{normalized_word}"?'

    string = row['normalized_word']
    char_to_duplicate = get_random_char(string)
    kwargs = {"char_to_duplicate": char_to_duplicate}

    mcq_correct = duplicate_char(string, **kwargs)
    mcq_incorrect = manipulate_string(string, target_manipulation="duplication", kwargs=kwargs)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row, kwargs=kwargs)
    return outputs


def create_mcq_uppercasing(row):
    text_en = 'Which option correctly changes the word "{normalized_word}" to all uppercase?'
    text_tl = 'Alin sa sumusunod ang ginagawang malaki ang lahat ng titik sa salitang "{normalized_word}"?'

    string = row['normalized_word']
    mcq_correct = string.upper()
    mcq_incorrect = [string.lower(), string[:len(string)//2].upper() + string[len(string)//2:].lower(), ''.join(c.upper() if random.random() < 0.5 else c.lower() for c in string)]
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


def create_mcq_lowercasing(row):
    text_en = 'Which option correctly changes the word "{normalized_word}" to all lowercase?'
    text_tl = 'Alin sa sumusunod ang ginagawang maliit ang lahat ng titik sa salitang "{normalized_word}"?'

    string = row['normalized_word']
    mcq_correct = string.lower()
    mcq_incorrect = [string[:len(string)//2].upper() + string[len(string)//2:].lower(), ''.join(c.lower() if random.random() < 0.5 else c.upper() for c in string), string.upper()]
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


def create_mcq_diacritic_normalization(row):
    text_en = 'Which option correctly normalizes diacritics from the word "{word}"?'
    text_tl = 'Alin sa sumusunod ang nagtatanggal ng mga tuldik sa salitang "{word}"?'

    string = row['word']
    mcq_correct = normalize_diacritic(string)
    mcq_incorrect = diacritize_string(string, mcq_correct)
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    outputs = prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)
    return outputs


# ============================================================================
# Generative Question Creation Functions
# ============================================================================

def create_gen_deletion(row: pd.Series) -> Dict[str, Any]:
    """
    Create generative question for character deletion operation.
    
    Args:
        row: DataFrame row containing word data
    
    Returns:
        Dictionary containing formatted generative prompts and label
    """
    text_en = 'Remove every character "{char_to_delete}" in the word "{normalized_word}".'
    text_tl = 'Tanggalin ang bawat titik na "{char_to_delete}" sa salitang "{normalized_word}".'

    string = row['normalized_word']
    char_to_delete = get_random_char(string)
    kwargs = {"char_to_delete": char_to_delete}
    label = delete_char(string, **kwargs)

    return prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)


def create_gen_insertion(row):
    text_en = 'Put a character "{char_to_insert}" after every character "{preceding_char}" in the word "{normalized_word}".'
    text_tl = 'Maglagay ng titik na "{char_to_insert}" pagkatapos ng bawat titik na "{preceding_char}" sa salitang "{normalized_word}".'

    string = row['normalized_word']
    preceding_char = get_random_char(string)
    char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz')
    kwargs = {"preceding_char": preceding_char, "char_to_insert": char_to_insert}
    label = insert_char(string, **kwargs)

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def create_gen_substitution(row):
    text_en = 'Replace every character "{char_to_replace}" with the character "{char_to_substitute}" in the word "{normalized_word}".'
    text_tl = 'Palitan ang bawat titik na "{char_to_replace}" gamit ng titik na "{char_to_substitute}" sa salitang "{normalized_word}".'

    string = row['normalized_word']
    char_to_replace = get_random_char(string)
    remaining_chars = 'abcdefghijklmnopqrstuvwxyz'.replace(char_to_replace, '')
    char_to_substitute = get_random_char(remaining_chars)
    kwargs = {"char_to_replace": char_to_replace, "char_to_substitute": char_to_substitute}
    label = substitute_char(string, **kwargs)

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def create_gen_permutation(row):
    text_en = 'Swap every character "{char1}" and character "{char2}" in the word "{normalized_word}".'
    text_tl = 'Pagpalitin ang bawat titik na "{char1}" at titik na "{char2}" sa salitang "{normalized_word}".'

    string = row['normalized_word']
    char1 = get_random_char(string)
    remaining_string = string.replace(char1, '')
    char2 = get_random_char(remaining_string)
    kwargs = {"char1": char1, "char2": char2}
    label = permute_char(string, **kwargs)

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def create_gen_duplication(row):
    text_en = 'Duplicate every character "{char_to_duplicate}" once in the word "{normalized_word}".'
    text_tl = 'Ulitin ang bawat titik na "{char_to_duplicate}" nang isang beses sa salitang "{normalized_word}".'

    string = row['normalized_word']
    char_to_duplicate = get_random_char(string)
    kwargs = {"char_to_duplicate": char_to_duplicate}
    label = duplicate_char(string, **kwargs)

    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)
    return outputs


def create_gen_uppercasing(row):
    text_en = 'Change the word "{normalized_word}" into uppercase.'
    text_tl = 'Gawing malaki ang lahat ng titik sa salitang "{normalized_word}".'

    label = row["normalized_word"].upper()
    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs={})
    return outputs


def create_gen_lowercasing(row):
    text_en = 'Change the word "{normalized_word}" into lowercase.'
    text_tl = 'Gawing maliit ang lahat ng titik sa salitang "{normalized_word}".'

    label = row["normalized_word"].lower()
    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs={})
    return outputs


def create_gen_diacritic_normalization(row):
    text_en = 'Normalize the diacritics from the word "{word}".'
    text_tl = 'Tanggalin ang lahat ng mga tuldik sa salitang "{word}".'

    label = normalize_diacritic(row["word"])
    outputs = prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs={})
    return outputs


# ============================================================================
# Helper Functions for Dataset Creation
# ============================================================================

def _shuffle_mcq_options(dataset: pd.DataFrame, random_seed: int = 100) -> pd.DataFrame:
    """
    Shuffle MCQ options and assign labels to the dataset.
    
    Args:
        dataset: DataFrame containing MCQ questions with mcq_options
        random_seed: Random seed for reproducibility
    
    Returns:
        Modified dataset with shuffled choices and assigned labels
    """
    random.seed(random_seed)
    
    for i in range(len(dataset)):
        label_index = i % NUM_MCQ_OPTIONS
        correct = dataset.iloc[i]['prompts'][0]["mcq_options"]['correct']
        options = [
            dataset.iloc[i]['prompts'][0]["mcq_options"]['incorrect1'],
            dataset.iloc[i]['prompts'][0]["mcq_options"]['incorrect2'],
            dataset.iloc[i]['prompts'][0]["mcq_options"]['incorrect3'],
        ]
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


def _filter_by_word_length(syllables_df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """
    Filter syllables DataFrame to only include words of sufficient length.
    
    Args:
        syllables_df: Input DataFrame containing syllable/word data
        min_length: Minimum word length (inclusive)
    
    Returns:
        Filtered DataFrame containing only words with length >= min_length
    """
    return syllables_df[syllables_df['normalized_word'].str.len() >= min_length].reset_index(drop=True)


def _apply_frequency_weighting(syllables_df: pd.DataFrame, freq_weight: float, random_seed: int) -> pd.DataFrame:
    """
    Apply frequency-based sampling to the syllables dataframe.
    
    Args:
        syllables_df: Input DataFrame containing syllable/word data
        freq_weight: Weight for frequency-based sampling (0.0 to 1.0)
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with frequency rankings and sampling applied
    """
    from pacute_bench.utils.sampling import load_frequency_data, add_frequency_ranks, sample_by_frequency
    freq_df = load_frequency_data()
    syllables_df = add_frequency_ranks(syllables_df, freq_df)
    syllables_df = sample_by_frequency(
        syllables_df,
        n_samples=len(syllables_df),
        freq_weight=freq_weight,
        random_state=random_seed
    )
    return syllables_df


# ============================================================================
# Main Dataset Creation Function
# ============================================================================

def create_manipulation_dataset(
    syllables_df: pd.DataFrame,
    num_samples: int,
    mode: str = 'mcq',
    random_seed: int = 100,
    freq_weight: float = 0.0
) -> pd.DataFrame:
    """
    Create a complete string manipulation dataset with various transformation tasks.
    
    This function generates a comprehensive dataset for testing Filipino string
    manipulation skills. It creates multiple types of tasks including:
    - Character operations: insertion, deletion, substitution, permutation, duplication
    - Case transformations: uppercasing, lowercasing
    - Diacritic operations: normalization (removal of diacritics)
    
    Args:
        syllables_df: DataFrame containing syllable/word data with columns:
            - word: The original word (may contain diacritics/uppercase)
            - normalized_word: Normalized version of the word
        num_samples: Number of samples to generate per task type
        mode: Question format - 'mcq' for multiple-choice or 'gen' for generative (default: 'mcq')
        random_seed: Random seed for reproducibility (default: 100)
        freq_weight: Weight for frequency-based sampling, 0.0 to 1.0 (default: 0.0)
            Higher values prioritize more common words
    
    Returns:
        DataFrame with columns:
            - category: Task category (always "manipulation")
            - subcategory: Specific task type (insertion, deletion, etc.)
            - prompts: List containing question prompts in English and Filipino
            - label: Correct answer (A/B/C/D for MCQ, actual answer for GEN)
    
    Raises:
        ValueError: If mode is not 'mcq' or 'gen'
    
    Examples:
        >>> syllables = pd.read_json("syllables.jsonl", lines=True)
        >>> mcq_dataset = create_manipulation_dataset(syllables, num_samples=100, mode='mcq')
        >>> gen_dataset = create_manipulation_dataset(syllables, num_samples=100, mode='gen')
    """
    random.seed(random_seed)

    dataset = pd.DataFrame(columns=["category", "subcategory", "prompts", "label"])
    
    # Filter for words of sufficient length (manipulation needs longer words)
    syllables_df = _filter_by_word_length(syllables_df, MIN_WORD_LENGTH_MANIPULATION)

    if mode == 'mcq':
        tasks = {
            "insertion": create_mcq_insertion,
            "deletion": create_mcq_deletion,
            "substitution": create_mcq_substitution,
            "permutation": create_mcq_permutation,
            "duplication": create_mcq_duplication,
            "uppercasing": create_mcq_uppercasing,
            "lowercasing": create_mcq_lowercasing,
            "diacritic_normalization": create_mcq_diacritic_normalization,
        }

        for subcategory_name, subcategory_function in tasks.items():
            # Apply frequency weighting if requested, otherwise use uniform sampling
            if freq_weight > 0:
                sampled_df = _apply_frequency_weighting(syllables_df, freq_weight, random_seed)
                sample_rows = sampled_df.head(num_samples)
            else:
                sample_rows = syllables_df.sample(num_samples, random_state=random_seed)
            
            for _, row in sample_rows.iterrows():
                mcq_row = subcategory_function(row)
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category": ["manipulation"],
                    "subcategory": [subcategory_name],
                    "prompts": [mcq_row["prompts"]],
                })], ignore_index=True)

        # Shuffle options and assign labels
        dataset = _shuffle_mcq_options(dataset, random_seed)

    elif mode == 'gen':
        tasks = {
            "insertion": create_gen_insertion,
            "deletion": create_gen_deletion,
            "substitution": create_gen_substitution,
            "permutation": create_gen_permutation,
            "duplication": create_gen_duplication,
            "uppercasing": create_gen_uppercasing,
            "lowercasing": create_gen_lowercasing,
            "diacritic_normalization": create_gen_diacritic_normalization,
        }

        for subcategory_name, subcategory_function in tasks.items():
            # Apply frequency weighting if requested, otherwise use uniform sampling
            if freq_weight > 0:
                sampled_df = _apply_frequency_weighting(syllables_df, freq_weight, random_seed)
                sample_pool = sampled_df.head(num_samples * 100)
            else:
                sample_pool = syllables_df.sample(num_samples * 100, random_state=random_seed)
            
            for _, row in sample_pool.iterrows():
                if len(dataset[dataset['subcategory'] == subcategory_name]) >= num_samples:
                    break

                from .composition import check_if_diacritic
                if subcategory_name == "diacritic_normalization" and not any(check_if_diacritic(char) for char in row['word']):
                    continue
                if subcategory_name == "lowercasing" and not any(char.isupper() for char in row['normalized_word']):
                    continue
                if subcategory_name == "uppercasing" and not any(char.islower() for char in row['normalized_word']):
                    continue

                gen_row = subcategory_function(row)
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category": ["manipulation"],
                    "subcategory": [subcategory_name],
                    "prompts": [gen_row["prompts"]],
                    "label": [gen_row["label"]],
                })], ignore_index=True)
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'mcq' or 'gen'.")

    return dataset
