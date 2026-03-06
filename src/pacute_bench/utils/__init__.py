"""
Utility functions and constants for pacute-bench evaluation.
"""

from .constants import (
    MCQ_LABEL_MAP, NUM_MCQ_OPTIONS, NUM_INCORRECT_OPTIONS,
    MIN_WORD_LENGTH_COMPOSITION, MIN_WORD_LENGTH_MANIPULATION,
    MIN_WORD_LENGTH_SYLLABIFICATION, MIN_WORD_LENGTH_GENERAL_SYLLABLE_COUNTING,
    AFFIX_TYPES,
    VOWELS, DIACRITICS, ACCENTED_VOWELS, UPPERCASE_LETTERS, UPPERCASE_DIACRITICS,
    MABILIS, MALUMI, MARAGSA,
    DIACRITIC_MAP, REVERSE_DIACRITIC_MAP,
    LETTER_PAIRS, STRESS_PRONUNCIATION_MAP,
    DEFAULT_FREQUENCY_FILE, DEFAULT_RANK_FILLNA, DEFAULT_FREQ_WEIGHT, DEFAULT_RANDOM_STATE,
)
from .helpers import prepare_mcq_outputs, prepare_gen_outputs
from .sampling import load_frequency_data, add_frequency_ranks, sample_by_frequency
from .strings import (
    string_to_chars, chars_to_string,
    get_random_char, same_string,
    delete_char, insert_char, substitute_char, permute_char, duplicate_char,
    normalize_diacritic, diacritize, randomly_diacritize,
    spell_string,
    shuffle_chars, randomly_merge_chars, randomly_insert_char, randomly_delete_char,
    perturb_string,
)
from .syllabification import syllabify

__all__ = [
    "MCQ_LABEL_MAP", "NUM_MCQ_OPTIONS", "NUM_INCORRECT_OPTIONS",
    "MIN_WORD_LENGTH_COMPOSITION", "MIN_WORD_LENGTH_MANIPULATION",
    "MIN_WORD_LENGTH_SYLLABIFICATION", "MIN_WORD_LENGTH_GENERAL_SYLLABLE_COUNTING",
    "AFFIX_TYPES",
    "VOWELS", "DIACRITICS", "ACCENTED_VOWELS", "UPPERCASE_LETTERS", "UPPERCASE_DIACRITICS",
    "MABILIS", "MALUMI", "MARAGSA",
    "DIACRITIC_MAP", "REVERSE_DIACRITIC_MAP",
    "LETTER_PAIRS", "STRESS_PRONUNCIATION_MAP",
    "DEFAULT_FREQUENCY_FILE", "DEFAULT_RANK_FILLNA", "DEFAULT_FREQ_WEIGHT", "DEFAULT_RANDOM_STATE",
    "prepare_mcq_outputs", "prepare_gen_outputs",
    "load_frequency_data", "add_frequency_ranks", "sample_by_frequency",
    "string_to_chars", "chars_to_string",
    "get_random_char", "same_string",
    "delete_char", "insert_char", "substitute_char", "permute_char", "duplicate_char",
    "normalize_diacritic", "diacritize", "randomly_diacritize",
    "spell_string",
    "shuffle_chars", "randomly_merge_chars", "randomly_insert_char", "randomly_delete_char",
    "perturb_string",
    "syllabify",
]
