"""
String Composition Dataset Generation Module

This module provides functionality for creating linguistic datasets focused on
Filipino string composition tasks, including spelling, character counting, 
length analysis, and diacritic handling. Supports both multiple-choice 
questions (MCQ) and generative (GEN) formats.
"""

import random
from typing import Dict, List, Any, Optional
import pandas as pd
from pacute_bench.utils.strings import (
    string_to_chars, chars_to_string, spell_string, perturb_string, get_random_char
)
from pacute_bench.utils.constants import (
    MCQ_LABEL_MAP,
    NUM_MCQ_OPTIONS,
    NUM_INCORRECT_OPTIONS,
    MIN_WORD_LENGTH_COMPOSITION,
    DIACRITICS,
    UPPERCASE_LETTERS,
    UPPERCASE_DIACRITICS
)
from pacute_bench.utils.helpers import prepare_mcq_outputs, prepare_gen_outputs


# ============================================================================
# Character and Diacritic Checking Functions
# ============================================================================

def check_if_diacritic(char: str) -> bool:
    """
    Check if a character is a diacritic.
    
    Args:
        char: Character to check
    
    Returns:
        True if the character is a diacritic, False otherwise
    """
    return char in DIACRITICS


def check_if_uppercase(char: str) -> bool:
    """
    Check if a character is uppercase.
    
    Args:
        char: Character to check
    
    Returns:
        True if the character is uppercase, False otherwise
    """
    return char in UPPERCASE_LETTERS or char in UPPERCASE_DIACRITICS


# ============================================================================
# MCQ Question Creation Functions
# ============================================================================

def create_mcq_spelling(row: pd.Series) -> Dict[str, Any]:
    """
    Create a multiple-choice question for spelling out a word.
    
    Args:
        row: DataFrame row containing word data
    
    Returns:
        Dictionary containing formatted MCQ prompts and options
    """
    text_en = 'Which option spells out the word "{normalized_word}" by placing spaces between each character?'
    text_tl = 'Alin sa sumusunod ang nagbabaybay sa salitang "{normalized_word}" sa pamamagitan ng paglagay ng espasyo sa pagitan ng bawat titik?'

    mcq_correct = spell_string(row['normalized_word'])
    mcq_incorrect = perturb_string(row['normalized_word'])
    mcq_options = {
        "correct": mcq_correct,
        "incorrect1": mcq_incorrect[0],
        "incorrect2": mcq_incorrect[1],
        "incorrect3": mcq_incorrect[2],
    }

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, row=row)


# ============================================================================
# Generative Question Creation Functions
# ============================================================================

def create_gen_spelling(row: pd.Series) -> Dict[str, Any]:
    """
    Create a generative question for spelling out a word.

    Args:
        row: DataFrame row containing word data

    Returns:
        Dictionary containing formatted generative prompts and label
    """
    text_en = 'To spell a word, place a space between each character. Spell out the word "{normalized_word}".'
    text_tl = 'Para baybayin ang isang salita, maglagay ng espasyo sa pagitan ng bawat titik. Baybayin ang salitang "{normalized_word}".'

    spelling = string_to_chars(row['normalized_word'])
    label = chars_to_string(spelling, add_space=True)
    return prepare_gen_outputs(text_en, text_tl, str(label), row=row)


def create_gen_character(row: pd.Series) -> Dict[str, Any]:
    """
    Create a generative question for counting occurrences of a random character.
    
    Args:
        row: DataFrame row containing word data
    
    Returns:
        Dictionary containing formatted generative prompts and label
    """
    text_en = 'How many "{character}"s are in the word "{normalized_word}"?'
    text_tl = 'Ilang "{character}" ang mayroon sa salitang "{normalized_word}"?'

    character_list = string_to_chars(row['normalized_word'])
    character_counts = {char: character_list.count(char) for char in set(character_list)}
    random_character = random.choice(list(character_counts.keys()))
    label = character_counts[random_character]
    kwargs = {"character": random_character}

    return prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs=kwargs)


def create_gen_length(row: pd.Series) -> Dict[str, Any]:
    """
    Create a generative question for counting total characters in a word.
    
    Args:
        row: DataFrame row containing word data
    
    Returns:
        Dictionary containing formatted generative prompts and label
    """
    text_en = 'How many characters are in the word "{normalized_word}"?'
    text_tl = 'Ilan ang titik sa salitang "{normalized_word}"?'

    label = len(row['normalized_word'])
    return prepare_gen_outputs(text_en, text_tl, str(label), row=row)


def create_gen_diacritic(row: pd.Series) -> Dict[str, Any]:
    """
    Create a generative question for counting diacritics in a word.
    
    Args:
        row: DataFrame row containing word data
    
    Returns:
        Dictionary containing formatted generative prompts and label
    """
    text_en = 'How many diacritics are in the word "{word}"?'
    text_tl = 'Ilang titik ang mayroong tuldik sa salitang "{word}"?'

    character_list = string_to_chars(row['word'])
    diacritic_counts = {
        char: character_list.count(char) 
        for char in set(character_list) 
        if check_if_diacritic(char)
    }
    label = sum(diacritic_counts.values()) if diacritic_counts else 0

    return prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs={})


def create_gen_uppercase(row: pd.Series) -> Dict[str, Any]:
    """
    Create a generative question for counting uppercase characters in a word.
    
    Args:
        row: DataFrame row containing word data
    
    Returns:
        Dictionary containing formatted generative prompts and label
    """
    text_en = 'How many uppercase characters are in the word "{normalized_word}"?'
    text_tl = 'Ilang malaking titik ang mayroon sa salitang "{normalized_word}"?'

    character_list = string_to_chars(row['normalized_word'])
    uppercase_counts = {
        char: character_list.count(char) 
        for char in set(character_list) 
        if char.isupper()
    }
    label = sum(uppercase_counts.values()) if uppercase_counts else 0

    return prepare_gen_outputs(text_en, text_tl, str(label), row=row, kwargs={})


# ============================================================================
# Helper Functions for MCQ Option Preparation
# ============================================================================

def extract_character_counts(rows: List[Dict[str, Any]], target: str, char: str) -> Dict[str, int]:
    """
    Extract character counts for a specific character across multiple words.
    
    Args:
        rows: List of row dictionaries containing word data
        target: Key name for the target word field
        char: Character to count in each word
    
    Returns:
        Dictionary mapping words to their count of the specified character
    """
    if char is not None:
        character_counts = {}
        for row in rows:
            character_counts[row[target]] = row[target].count(char)
        return character_counts


def prepare_options(words: Dict[str, Any], correct_word: str) -> Optional[Dict[str, str]]:
    """
    Prepare MCQ options from a dictionary of words.
    
    Args:
        words: Dictionary of words (keys) and their properties (values)
        correct_word: The word that is the correct answer
    
    Returns:
        Dictionary with correct and incorrect MCQ options, or None if insufficient unique words
    """
    incorrect_words = [word for word in words.keys() if word != correct_word]
    
    # Need at least 3 incorrect options
    if len(incorrect_words) < NUM_INCORRECT_OPTIONS:
        return None
    
    mcq_options = {
        "correct": correct_word,
        "incorrect1": incorrect_words[0],
        "incorrect2": incorrect_words[1],
        "incorrect3": incorrect_words[2],
    }
    return mcq_options


def prepare_options_capitalize(words: Dict[str, Any], correct_word: str) -> Optional[Dict[str, str]]:
    """
    Prepare MCQ options with capitalization variations.
    
    Args:
        words: Dictionary of words (keys) and their properties (values)
        correct_word: The word that is the correct answer
    
    Returns:
        Dictionary with correct (capitalized) and incorrect MCQ options, or None if insufficient unique words
    """
    incorrect_words = [word for word in words.keys() if word != correct_word]

    # Need at least 3 incorrect options
    if len(incorrect_words) < NUM_INCORRECT_OPTIONS:
        return None

    mcq_options = {
        "correct": correct_word.capitalize(),
        "incorrect1": correct_word.lower(),
        "incorrect2": incorrect_words[1].capitalize(),
        "incorrect3": incorrect_words[2].capitalize(),
    }

    return mcq_options


# ============================================================================
# Validation Functions for MCQ Generation
# ============================================================================

def check_if_any_character_counts_are_unique(rows: List[Dict[str, Any]], target: str) -> Optional[str]:
    """
    Find a character whose counts are unique across all four words.
    
    This ensures that each word has a different count of the character,
    making it possible to create an unambiguous MCQ question.
    
    Args:
        rows: List of row dictionaries containing word data
        target: Key name for the target word field
    
    Returns:
        A character with unique counts across all words, or None if not found
    """
    words = {}
    for row in rows:
        words[row[target]] = {
            char: row[target].count(char) for char in set(list(row[target]))
        }

    possible_chars = list(set().union(*[set(counts.keys()) for counts in words.values()]))
    random.shuffle(possible_chars)

    for char in possible_chars:
        char_counts = []
        for char_count in words.values():
            char_counts.append(char_count.get(char, 0))

        # All four words must have different counts
        if len(set(char_counts)) == NUM_MCQ_OPTIONS:
            return char

    return None


def check_if_any_diacritic_counts_are_unique(rows: List[Dict[str, Any]], target: str) -> Optional[str]:
    """
    Find a diacritic character that appears in exactly one of the four words.
    
    Args:
        rows: List of row dictionaries containing word data
        target: Key name for the target word field
    
    Returns:
        A diacritic character appearing in exactly one word, or None if not found
    """
    words = {}
    for row in rows:
        words[row[target]] = {}
        for char in row[target]:
            if check_if_diacritic(char):
                count = row[target].count(char)
                words[row[target]][char] = count

    possible_chars = list(set().union(*[set(counts.keys()) for counts in words.values()]))
    random.shuffle(possible_chars)

    for char in possible_chars:
        char_counts = []
        for char_count in words.values():
            char_counts.append(char_count.get(char, 0))

        # Diacritic should appear in exactly one word
        if sum(char_counts) == 1:
            return char

    return None


def check_if_any_uppercase_counts_are_unique(rows: List[Dict[str, Any]], target: str) -> Optional[str]:
    """
    Find an uppercase character that appears in exactly one of the four words.
    
    Args:
        rows: List of row dictionaries containing word data
        target: Key name for the target word field
    
    Returns:
        An uppercase character appearing in exactly one word, or None if not found
    """
    words = {}
    for row in rows:
        words[row[target]] = {}
        for char in row[target]:
            if check_if_uppercase(char):
                count = row[target].count(char)
                words[row[target]][char] = count

    possible_chars = list(set().union(*[set(counts.keys()) for counts in words.values()]))
    random.shuffle(possible_chars)

    for char in possible_chars:
        char_counts = []
        for char_count in words.values():
            char_counts.append(char_count.get(char, 0))

        # Uppercase char should appear in exactly one word
        if sum(char_counts) == 1:
            return char

    return None


def check_if_row_lengths_are_unique(rows: List[Dict[str, Any]], target: str) -> bool:
    """
    Check if all four words have unique lengths.
    
    Args:
        rows: List of exactly 4 row dictionaries containing word data
        target: Key name for the target word field
    
    Returns:
        True if all four words have different lengths, False otherwise
    """
    string1, string2, string3, string4 = (
        rows[0][target], rows[1][target], rows[2][target], rows[3][target]
    )
    strings = {
        string1: len(string1), 
        string2: len(string2), 
        string3: len(string3), 
        string4: len(string4)
    }

    lengths = list(strings.values())
    return len(lengths) == len(set(lengths))


# ============================================================================
# Character-Based MCQ Creation Functions
# ============================================================================

def create_mcq_char_exactly_one(rows: List[Dict[str, Any]], target: str, char: str) -> Dict[str, Any]:
    """
    Create MCQ asking which word contains exactly one occurrence of a character.
    
    Args:
        rows: List of row dictionaries containing word data
        target: Key name for the target word field
        char: Character to count
    
    Returns:
        Dictionary containing formatted MCQ prompts and options
    """
    character_counts = extract_character_counts(rows, target=target, char=char)

    target_count = 1
    correct_word = [word for word, count in character_counts.items() if count == target_count][0]
    mcq_options = prepare_options(character_counts, correct_word)
    
    if mcq_options is None:
        return None
    
    kwargs = {"target_count": target_count, "char": char}

    text_en = 'Which option contains exactly {target_count} "{char}" character/s?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng eksaktong {target_count} titik na "{char}"?'

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)


def create_mcq_uppercase_exactly_one(rows: List[Dict[str, Any]], target: str, char: str) -> Optional[Dict[str, Any]]:
    """
    Create MCQ asking which word contains exactly one uppercase occurrence of a character.
    
    Args:
        rows: List of row dictionaries containing word data
        target: Key name for the target word field
        char: Uppercase character to count
    
    Returns:
        Dictionary containing formatted MCQ prompts and options, or None if insufficient unique words
    """
    character_counts = extract_character_counts(rows, target=target, char=char)

    target_count = 1
    correct_word = [word for word, count in character_counts.items() if count == target_count][0]
    mcq_options = prepare_options_capitalize(character_counts, correct_word)
    
    if mcq_options is None:
        return None
    
    kwargs = {"target_count": target_count, "char": char}

    text_en = 'Which option contains exactly {target_count} "{char}" character/s?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng eksaktong {target_count} titik na "{char}"?'

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)


def create_mcq_char_exactly(rows: List[Dict[str, Any]], target: str, char: str) -> Optional[Dict[str, Any]]:
    """
    Create MCQ asking which word contains exactly N occurrences of a character.
    
    Args:
        rows: List of row dictionaries containing word data
        target: Key name for the target word field
        char: Character to count
    
    Returns:
        Dictionary containing formatted MCQ prompts and options, or None if insufficient unique words
    """
    character_counts = extract_character_counts(rows, target=target, char=char)

    correct_word = random.choice(list(character_counts.keys()))
    target_count = character_counts[correct_word]
    mcq_options = prepare_options(character_counts, correct_word)
    
    if mcq_options is None:
        return None
    
    kwargs = {"target_count": target_count, "char": char}

    text_en = 'Which option contains exactly {target_count} "{char}" character/s?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng eksaktong {target_count} titik na "{char}"?'

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)


def create_mcq_char_most(rows: List[Dict[str, Any]], target: str, char: str) -> Optional[Dict[str, Any]]:
    """
    Create MCQ asking which word contains the most occurrences of a character.
    
    Args:
        rows: List of row dictionaries containing word data
        target: Key name for the target word field
        char: Character to count
    
    Returns:
        Dictionary containing formatted MCQ prompts and options, or None if insufficient unique words
    """
    character_counts = extract_character_counts(rows, target=target, char=char)

    target_count = max(character_counts.values())
    correct_word = [word for word, count in character_counts.items() if count == target_count][0]
    mcq_options = prepare_options(character_counts, correct_word)
    
    if mcq_options is None:
        return None
    
    kwargs = {"char": char}

    text_en = 'Which option contains the most number of "{char}" characters?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng pinakamaraming titik na "{char}"?'

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)


def create_mcq_char_least(rows: List[Dict[str, Any]], target: str, char: str) -> Optional[Dict[str, Any]]:
    """
    Create MCQ asking which word contains the least occurrences of a character.
    
    Args:
        rows: List of row dictionaries containing word data
        target: Key name for the target word field
        char: Character to count
    
    Returns:
        Dictionary containing formatted MCQ prompts and options, or None if insufficient unique words
    """
    character_counts = extract_character_counts(rows, target=target, char=char)

    target_count = min(character_counts.values())
    correct_word = [word for word, count in character_counts.items() if count == target_count][0]
    mcq_options = prepare_options(character_counts, correct_word)
    
    if mcq_options is None:
        return None
    
    kwargs = {"char": char}

    text_en = 'Which option contains the least number of "{char}" characters?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng pinakakaunting titik na "{char}"?'

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)


# ============================================================================
# Length-Based MCQ Creation Functions
# ============================================================================

def extract_length(rows: List[Dict[str, Any]], target: str) -> Dict[str, int]:
    """
    Extract word lengths from multiple rows.
    
    Args:
        rows: List of exactly 4 row dictionaries containing word data
        target: Key name for the target word field
    
    Returns:
        Dictionary mapping words to their lengths
    """
    word1, word2, word3, word4 = (
        rows[0][target], rows[1][target], rows[2][target], rows[3][target]
    )
    words = {
        word1: len(word1), 
        word2: len(word2), 
        word3: len(word3), 
        word4: len(word4)
    }
    return words


def create_mcq_length_exactly(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Create MCQ asking which word contains exactly N characters.
    
    Args:
        rows: List of row dictionaries containing word data
    
    Returns:
        Dictionary containing formatted MCQ prompts and options, or None if insufficient unique words
    """
    words = extract_length(rows, target="normalized_word")

    correct_word = random.choice(list(words.keys()))
    target_length = words[correct_word]
    mcq_options = prepare_options(words, correct_word)
    
    if mcq_options is None:
        return None
    
    kwargs = {"target_length": target_length}

    text_en = 'Which option contains exactly {target_length} characters?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng eksaktong {target_length} titik?'

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)


def create_mcq_length_most(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Create MCQ asking which word contains the most characters.
    
    Args:
        rows: List of row dictionaries containing word data
    
    Returns:
        Dictionary containing formatted MCQ prompts and options, or None if insufficient unique words
    """
    words = extract_length(rows, target="normalized_word")

    target_length = max(words.values())
    correct_word = [word for word, length in words.items() if length == target_length][0]
    mcq_options = prepare_options(words, correct_word)
    
    if mcq_options is None:
        return None
    
    kwargs = {}

    text_en = 'Which option contains the most number of characters?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng pinakamaraming titik?'

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)


def create_mcq_length_least(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Create MCQ asking which word contains the least characters.
    
    Args:
        rows: List of row dictionaries containing word data
    
    Returns:
        Dictionary containing formatted MCQ prompts and options, or None if insufficient unique words
    """
    words = extract_length(rows, target="normalized_word")

    target_length = min(words.values())
    correct_word = [word for word, length in words.items() if length == target_length][0]
    mcq_options = prepare_options(words, correct_word)
    
    if mcq_options is None:
        return None
    
    kwargs = {}

    text_en = 'Which option contains the least number of characters?'
    text_tl = 'Alin sa sumusunod ang naglalaman ng pinakakaunting titik?'

    return prepare_mcq_outputs(text_en, text_tl, mcq_options, kwargs=kwargs)


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

def create_composition_dataset(
    syllables_df: pd.DataFrame,
    num_samples: int,
    mode: str = 'mcq',
    random_seed: int = 100,
    freq_weight: float = 0.0
) -> pd.DataFrame:
    """
    Create a complete composition dataset with various string manipulation tasks.
    
    This function generates a comprehensive dataset for testing Filipino string
    composition and analysis skills. It creates multiple types of tasks including:
    - Spelling: Spell out words letter by letter
    - Character counting: Count specific characters in words
    - Length analysis: Determine word lengths
    - Diacritic handling: Identify and count diacritics
    - Case analysis: Work with uppercase/lowercase characters
    
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
            - category: Task category (always "composition")
            - subcategory: Specific task type (spelling, character, length, etc.)
            - prompts: List containing question prompts in English and Filipino
            - label: Correct answer (A/B/C/D for MCQ, actual answer for GEN)
    
    Raises:
        ValueError: If mode is not 'mcq' or 'gen'
    
    Examples:
        >>> syllables = pd.read_json("syllables.jsonl", lines=True)
        >>> mcq_dataset = create_composition_dataset(syllables, num_samples=100, mode='mcq')
        >>> gen_dataset = create_composition_dataset(syllables, num_samples=100, mode='gen')
    """
    random.seed(random_seed)

    dataset = pd.DataFrame(columns=["category", "subcategory", "prompts", "label"])
    
    # Filter for words of sufficient length
    syllables_df = _filter_by_word_length(syllables_df, MIN_WORD_LENGTH_COMPOSITION)

    if mode == 'mcq':
        tasks = {
            "spelling": (create_mcq_spelling, 'single'),
            "character_counting_exactly": (create_mcq_char_exactly, 'multi'),
            "character_counting_least": (create_mcq_char_least, 'multi'),
            "character_counting_most": (create_mcq_char_most, 'multi'),
            "length_counting_least": (create_mcq_length_least, 'multi'),
            "length_counting_most": (create_mcq_length_most, 'multi'),
        }

        for subcategory_name, (subcategory_function, task_type) in tasks.items():
            if task_type == 'single':
                # Apply frequency weighting if requested, otherwise use uniform sampling
                if freq_weight > 0:
                    sampled_df = _apply_frequency_weighting(syllables_df, freq_weight, random_seed)
                    sample_rows = sampled_df.head(num_samples)
                else:
                    sample_rows = syllables_df.sample(num_samples, random_state=random_seed)
                
                for _, row in sample_rows.iterrows():
                    mcq_row = subcategory_function(row)
                    dataset = pd.concat([dataset, pd.DataFrame({
                        "category": ["composition"],
                        "subcategory": [subcategory_name],
                        "prompts": [mcq_row["prompts"]],
                    })], ignore_index=True)
            elif task_type == 'multi':
                # For multi tasks, we need to maintain the full dataset but reordered by frequency
                if freq_weight > 0:
                    # Get frequency-weighted full dataset (all rows, just reordered)
                    from pacute_bench.utils.sampling import load_frequency_data, add_frequency_ranks
                    freq_df = load_frequency_data()
                    temp_df = add_frequency_ranks(syllables_df, freq_df)
                    # Sort by rank (lower rank = more common = higher priority)
                    shuffled_dataset = temp_df.sort_values('rank').drop(columns=['rank']).reset_index(drop=True)
                else:
                    shuffled_dataset = syllables_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
                
                processed_count = 0
                attempts = 0
                max_attempts = len(shuffled_dataset) // 4 * 3  # Allow up to 3x the dataset size in attempts
                
                # Keep cycling through the dataset until we get enough samples
                while processed_count < num_samples and attempts < max_attempts:
                    # Get a group of 4 samples
                    start_idx = (attempts * 4) % len(shuffled_dataset)
                    end_idx = min(start_idx + 4, len(shuffled_dataset))
                    
                    # If we're at the end, wrap around
                    if end_idx - start_idx < 4:
                        rows = shuffled_dataset.iloc[start_idx:end_idx]
                        remaining = 4 - (end_idx - start_idx)
                        rows = pd.concat([rows, shuffled_dataset.iloc[:remaining]])
                    else:
                        rows = shuffled_dataset.iloc[start_idx:end_idx]
                    
                    samples = rows.to_dict(orient="records")
                    mcq_row = None

                    if "length" in subcategory_name:
                        valid_length = check_if_row_lengths_are_unique(samples, target="normalized_word")
                        if valid_length:
                            mcq_row = subcategory_function(samples)
                    elif "diacritic_exactly" in subcategory_name:
                        valid_diacritic = check_if_any_diacritic_counts_are_unique(samples, target="word")
                        if valid_diacritic:
                            mcq_row = subcategory_function(samples, target="word", char=valid_diacritic)
                    elif "uppercase_exactly" in subcategory_name:
                        valid_uppercase = check_if_any_uppercase_counts_are_unique(samples, target="word")
                        if valid_uppercase:
                            mcq_row = subcategory_function(samples, target="word", char=valid_uppercase)
                    elif "char" in subcategory_name:
                        valid_char = check_if_any_character_counts_are_unique(samples, target="normalized_word")
                        if valid_char is not None:
                            mcq_row = subcategory_function(samples, target="normalized_word", char=valid_char)

                    if mcq_row is not None:
                        dataset = pd.concat([dataset, pd.DataFrame({
                            "category": ["composition"],
                            "subcategory": [subcategory_name],
                            "prompts": [mcq_row["prompts"]],
                        })], ignore_index=True)
                        processed_count += 1
                    
                    attempts += 1

        # Shuffle options and assign labels
        dataset = _shuffle_mcq_options(dataset, random_seed)

    elif mode == 'gen':
        tasks = {
            "spelling": create_gen_spelling,
            "character_counting": create_gen_character,
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

                if subcategory_name == "diacritic" and not any(check_if_diacritic(char) for char in row['word']):
                    continue
                if subcategory_name == "uppercase" and not any(char.isupper() for char in row['normalized_word']):
                    continue

                gen_row = subcategory_function(row)
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category": ["composition"],
                    "subcategory": [subcategory_name],
                    "prompts": [gen_row["prompts"]],
                    "label": [gen_row["label"]],
                })], ignore_index=True)
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'mcq' or 'gen'.")

    return dataset


# ============================================================================
# Corpus-driven composition tasks (from pacute_dataset.xlsx)
# ============================================================================

# Filipino single-character alphabet for character_recognition distractors.
_FILIPINO_CHARS = list("abcdefghijklmnñopqrstuvwxyz")


def _numeric_distractors(correct: int, pool: List[int], n: int = 3) -> List[str]:
    """Return n unique distractor integers from pool, excluding correct."""
    candidates = [v for v in pool if v != correct]
    if len(candidates) < n:
        # Extend with nearby values not in pool
        extra = correct + 1
        while len(candidates) < n:
            if extra != correct:
                candidates.append(extra)
            extra += 1
    return [str(v) for v in candidates[:n]]


def create_gen_corpus_task(row: pd.Series) -> Dict[str, Any]:
    """
    Create a GEN item directly from a corpus row that already has text_en/text_tl/label.
    """
    return prepare_gen_outputs(row["text_en"], row["text_tl"], str(row["label"]))


def create_mcq_corpus_diacritics(row: pd.Series) -> Dict[str, Any]:
    """MCQ for diacritics: options are numeric diacritic counts."""
    correct = int(row["diacritics_count"])
    pool = [0, 1, 2, 3]
    distractors = _numeric_distractors(correct, pool)
    mcq_options = {
        "correct": str(correct),
        "incorrect1": distractors[0],
        "incorrect2": distractors[1],
        "incorrect3": distractors[2],
    }
    return prepare_mcq_outputs(row["text_en"], row["text_tl"], mcq_options)


def create_mcq_corpus_uppercasing(row: pd.Series) -> Dict[str, Any]:
    """MCQ for uppercasing: options are numeric uppercase counts."""
    correct = int(row["uppercase_count"])
    pool = list(range(max(0, correct - 2), correct + 4))
    distractors = _numeric_distractors(correct, pool)
    mcq_options = {
        "correct": str(correct),
        "incorrect1": distractors[0],
        "incorrect2": distractors[1],
        "incorrect3": distractors[2],
    }
    return prepare_mcq_outputs(row["text_en"], row["text_tl"], mcq_options)


def create_mcq_corpus_character_counting(row: pd.Series) -> Dict[str, Any]:
    """MCQ for character counting: options are nearby letter counts."""
    correct = int(row["label"])
    pool = list(range(max(1, correct - 2), correct + 4))
    distractors = _numeric_distractors(correct, pool)
    mcq_options = {
        "correct": str(correct),
        "incorrect1": distractors[0],
        "incorrect2": distractors[1],
        "incorrect3": distractors[2],
    }
    return prepare_mcq_outputs(row["text_en"], row["text_tl"], mcq_options)


def create_mcq_corpus_character_recognition(row: pd.Series, random_state: random.Random) -> Dict[str, Any]:
    """MCQ for character recognition: options are distinct Filipino letters."""
    correct = str(row["label"]).lower()
    candidates = [c for c in _FILIPINO_CHARS if c != correct]
    random_state.shuffle(candidates)
    distractors = candidates[:3]
    mcq_options = {
        "correct": correct,
        "incorrect1": distractors[0],
        "incorrect2": distractors[1],
        "incorrect3": distractors[2],
    }
    return prepare_mcq_outputs(row["text_en"], row["text_tl"], mcq_options)


def _sample_strata(
    df: pd.DataFrame,
    col: str,
    strata: List[tuple],
    random_seed: int,
) -> pd.DataFrame:
    """
    Stratified sample. strata = list of (label_or_labels, n) where
    label_or_labels is a single value or list of values for that bucket.
    Uses min(n, available) per stratum.
    """
    frames = []
    for bucket, n in strata:
        if isinstance(bucket, list):
            mask = df[col].isin(bucket)
        else:
            mask = df[col] == bucket
        stratum = df[mask].reset_index(drop=True)
        take = min(n, len(stratum))
        if take > 0:
            frames.append(stratum.sample(take, random_state=random_seed))
    return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0]


def create_corpus_composition_dataset(
    corpus_df: pd.DataFrame,
    mode: str = "gen",
    random_seed: int = 100,
) -> pd.DataFrame:
    """
    Create composition benchmark items from the pre-built corpus
    (pacute_dataset.xlsx → corpus_composition.jsonl).

    Sampling:
      - diacritics      : 50 count=0, 50 count=1, 50 count>=2  (150 total)
      - uppercasing     : 50 count=0, 50 count=1, 50 count>=2  (150 total)
      - character_counting    : all rows
      - character_recognition : all rows

    Args:
        corpus_df: DataFrame loaded from corpus_composition.jsonl
        mode: 'gen' or 'mcq'
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: category, subcategory, prompts, label
    """
    rng = random.Random(random_seed)
    dataset = pd.DataFrame(columns=["category", "subcategory", "prompts", "label"])

    # Build per-subcategory sub-dataframes with stratified sampling applied
    sub_dfs: dict = {}

    dc = corpus_df[corpus_df["subcategory"] == "diacritics"].copy()
    dc["_bucket"] = dc["diacritics_count"].apply(lambda x: x if x in (0, 1) else "2+")
    sub_dfs["diacritics"] = _sample_strata(
        dc, "_bucket",
        [(0, 33), (1, 33), ("2+", 34)],
        random_seed,
    )

    uc = corpus_df[corpus_df["subcategory"] == "uppercasing"].copy()
    uc["_bucket"] = uc["uppercase_count"].apply(lambda x: x if x in (0, 1) else "2+")
    sub_dfs["uppercasing"] = _sample_strata(
        uc, "_bucket",
        [(0, 33), (1, 33), ("2+", 34)],
        random_seed,
    )

    cc = corpus_df[corpus_df["subcategory"] == "character_counting"].reset_index(drop=True)
    sub_dfs["character_counting"] = cc.sample(min(100, len(cc)), random_state=random_seed)
    cr = corpus_df[corpus_df["subcategory"] == "character_recognition"].reset_index(drop=True)
    sub_dfs["character_recognition"] = cr.sample(min(100, len(cr)), random_state=random_seed)

    _SUBCAT_RENAME = {
        "diacritics":          "diacritic_counting",
        "uppercasing":         "uppercase_counting",
        "character_counting":  "length_counting",
        "character_recognition": "character_finding",
    }

    for subcat in ["diacritics", "uppercasing", "character_counting", "character_recognition"]:
        sub_df = sub_dfs[subcat]
        if sub_df.empty:
            continue

        for _, row in sub_df.iterrows():
            if mode == "gen":
                result = create_gen_corpus_task(row)
                dataset = pd.concat([dataset, pd.DataFrame({
                    "category":    ["composition"],
                    "subcategory": [_SUBCAT_RENAME.get(subcat, subcat)],
                    "prompts":     [result["prompts"]],
                    "label":       [result["label"]],
                })], ignore_index=True)

            elif mode == "mcq":
                if subcat == "diacritics":
                    result = create_mcq_corpus_diacritics(row)
                elif subcat == "uppercasing":
                    result = create_mcq_corpus_uppercasing(row)
                elif subcat == "character_counting":
                    result = create_mcq_corpus_character_counting(row)
                elif subcat == "character_recognition":
                    result = create_mcq_corpus_character_recognition(row, rng)
                else:
                    continue

                dataset = pd.concat([dataset, pd.DataFrame({
                    "category":    ["composition"],
                    "subcategory": [_SUBCAT_RENAME.get(subcat, subcat)],
                    "prompts":     [result["prompts"]],
                })], ignore_index=True)

    if mode == "mcq":
        dataset = _shuffle_mcq_options(dataset, random_seed)

    return dataset
