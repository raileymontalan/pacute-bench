"""
Syllabification Operations Module

This module provides core functionality for Filipino syllabification,
including syllable splitting, stress classification, and word filtering
based on linguistic properties.
"""

import string
import unicodedata
from typing import List, Tuple, Optional
from .constants import (
    VOWELS,
    LETTER_PAIRS,
    ACCENTED_VOWELS,
    MABILIS,
    MALUMI,
    MARAGSA
)

# Use constants from the constants module
vowels = VOWELS
letter_pairs = LETTER_PAIRS
accented_vowels = ACCENTED_VOWELS
mabilis = MABILIS
malumi = MALUMI
maragsa = MARAGSA


def has_vowel(word: str) -> bool:
    """
    Check if a word contains at least one vowel.
    
    Args:
        word: Input word to check
        
    Returns:
        True if word contains at least one vowel, False otherwise
    """
    return any(let in vowels for let in word)


def slice_value_in_list(list_slice: List[str], value_slice: int, index_slice: int) -> List[str]:
    """
    Slice a string element in a list at a given index.
    
    Splits the string at list_slice[value_slice] at position index_slice,
    inserting the second part as a new element after the first.
    
    Args:
        list_slice: List of strings
        value_slice: Index of element to slice
        index_slice: Position within the string to slice at
        
    Returns:
        New list with the sliced element split into two
    """
    result = list_slice[:]
    result.insert(value_slice + 1, result[value_slice][index_slice:])
    result[value_slice] = result[value_slice][:index_slice]
    return result


def merge_value_in_list(list_merge: List[str], from_merge: int, to_merge: int) -> List[str]:
    """
    Merge consecutive elements in a list into a single element.
    
    Args:
        list_merge: List of strings
        from_merge: Starting index of merge range
        to_merge: Ending index of merge range (inclusive)
        
    Returns:
        New list with specified elements merged
    """
    result = list_merge[:]
    result[from_merge : to_merge + 1] = ["".join(result[from_merge : to_merge + 1])]
    return result


def syllabify(word_to_syllabify: str) -> List[str]:
    """
    Syllabify a Filipino word according to Filipino phonological rules.
    
    This function implements Filipino syllabification rules, including:
    - Treating 'ng' as a single consonant unit (digraph)
    - Handling vowel nuclei
    - Managing consonant clusters
    - Preserving diacritics for stress marking
    
    Args:
        word_to_syllabify: Filipino word to syllabify
        
    Returns:
        List of syllables
        
    Example:
        >>> syllabify("kumain")
        ['ku', 'ma', 'in']
        >>> syllabify("magandá")
        ['ma', 'gan', 'dá']
    """
    word = word_to_syllabify

    # Add spaces around vowels and hyphens to separate components
    for letter in word:
        if letter in vowels:
            word = word.replace(letter, f" {letter} ")
        elif letter == "-":
            word = word.replace(letter, f" - ")
    
    # Replace 'ng' with placeholder to treat as single unit
    word = word.replace("ng", "ŋ").replace("NG", "Ŋ")
    word = word.replace("'", "")
    word = word.split()

    offset = 0

    # Split consonant clusters between syllables
    for index, group in enumerate(word[:]):
        index += offset
        if index == 0 or index == len(word[:]) - 1 or word[index-1] == '-':
            continue
        elif len(group) == 2 and word:
            word = slice_value_in_list(word[:], index, 1)
            offset += 1
        elif len(group) == 3:
            # Special handling for nasal + consonant cluster (e.g., "mpl")
            if (
                any((group[0].lower() == "n", group[0].lower() == "m"))
                and group[1:3].lower() in letter_pairs
            ):
                word = slice_value_in_list(word[:], index, 1)
                offset += 1
            else:
                word = slice_value_in_list(word[:], index, 2)
                offset += 1
        elif len(group) > 3:
            word = slice_value_in_list(word[:], index, 2)
            offset += 1

    # Join consonants before vowels with the vowel
    join_word = word[:]
    offset = 0
    for index, group in enumerate(join_word):
        if (
            group[-1] in vowels
            and join_word[index - 1] not in vowels
            and join_word[index - 1] != "-"
            and index != 0
        ):
            word = merge_value_in_list(word, index - 1 - offset, index - offset)
            offset += 1

    # Join vowels with following consonants (coda)
    join_word = word[:]
    offset = 0
    for index, group in enumerate(join_word):
        if index != len(join_word) - 1:
            if (
                group[-1] in vowels
                and not has_vowel(join_word[index + 1])
                and join_word[index + 1] != "-"
            ):
                word = merge_value_in_list(word, index - offset, index + 1 - offset)
                offset += 1
    
    # Restore 'ng' from placeholder
    for i in range(len(word)):
        word[i] = word[i].replace("ŋ", "ng").replace("Ŋ", "NG")

    # Remove hyphens from syllable list
    while "-" in word:
        word.remove("-")

    return word


def normalize_text(text: str) -> str:
    """
    Normalize text by removing punctuation (except hyphens) and combining diacritics.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text with punctuation removed and diacritics normalized
    """
    if isinstance(text, str):
        text = text.strip()
        punctuation = string.punctuation.replace('-', '')
        text = ''.join(c for c in text if c not in punctuation)
        # Remove combining diacritical marks (NFD normalization)
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text


def is_filipino(etymology: str) -> bool:
    """
    Check if a word has Filipino etymology based on etymology tags.
    
    Args:
        etymology: Etymology string containing language tags
        
    Returns:
        True if etymology indicates Filipino origin
    """
    return any(tag in etymology for tag in ["Tag", "ST", "none"])


def is_single_word(word: str) -> bool:
    """
    Check if input is a single word (no spaces).
    
    Args:
        word: Input string to check
        
    Returns:
        True if input contains no spaces
    """
    return len(word.split()) == 1


def has_one_accented_syllable(word: str) -> bool:
    """
    Check if a word has exactly one accented syllable.
    
    Args:
        word: Filipino word to check
        
    Returns:
        True if word has exactly one syllable with accent marks
    """
    syllables = syllabify(word)
    count = sum(1 for syllable in syllables if any(char in accented_vowels for char in syllable))
    return count == 1


def not_circumfixed_with_dash(word: str) -> bool:
    """
    Check if word doesn't start or end with a hyphen (not a circumfix).
    
    Args:
        word: Word to check
        
    Returns:
        True if word doesn't start or end with hyphen
    """
    return not (word.startswith('-') or word.endswith('-'))


def find_accented_syllable(syllables: List[str]) -> Tuple[str, int]:
    """
    Find the first syllable containing an accented vowel.
    
    Args:
        syllables: List of syllables
        
    Returns:
        Tuple of (accented_syllable, index) or ("", -1) if none found
    """
    for i, syllable in enumerate(syllables):
        if any(char in accented_vowels for char in syllable):
            return syllable, i
    return "", -1


def find_last_syllable(syllables: List[str]) -> Tuple[str, int]:
    """
    Find the last syllable in a list.
    
    Args:
        syllables: List of syllables
        
    Returns:
        Tuple of (last_syllable, last_index)
    """
    return syllables[-1], len(syllables) - 1


def classify_last_syllable_pronunciation(last_syllable: str) -> str:
    """
    Classify the pronunciation/stress type of the last syllable.
    
    Filipino has four stress patterns based on accent marks:
    - mabilis: acute accent (á) - stressed on last syllable
    - malumi: grave accent (à) - stressed on penultimate syllable
    - maragsa: circumflex (â) - glottal stop
    - malumay: no accent - default stress
    
    Args:
        last_syllable: The last syllable to classify
        
    Returns:
        Stress classification: "mabilis", "malumi", "maragsa", or "malumay"
    """
    if any(char in mabilis for char in last_syllable):
        return "mabilis"
    elif any(char in malumi for char in last_syllable):
        return "malumi"
    elif any(char in maragsa for char in last_syllable):
        return "maragsa"
    else:
        return "malumay"
