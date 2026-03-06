"""
String Operations Module

This module provides low-level string manipulation functions for linguistic
operations including character manipulations, diacritic handling, and string
perturbations commonly used in Filipino language processing.
"""

import random
from typing import List, Optional
from .constants import DIACRITIC_MAP, REVERSE_DIACRITIC_MAP

# Use constants from the constants module
diacritic_map = DIACRITIC_MAP
reverse_diacritic_map = REVERSE_DIACRITIC_MAP


def string_to_chars(string: str) -> List[str]:
    """
    Convert a string into a list of individual characters.
    
    Args:
        string: Input string
        
    Returns:
        List of individual characters
        
    Example:
        >>> string_to_chars("hello")
        ['h', 'e', 'l', 'l', 'o']
    """
    return list(string)


def chars_to_string(char_list: List[str], add_space: bool = False) -> str:
    """
    Convert a list of characters back into a string.
    
    Args:
        char_list: List of characters
        add_space: If True, join with spaces; otherwise concatenate directly
        
    Returns:
        Reconstructed string
        
    Example:
        >>> chars_to_string(['h', 'e', 'l', 'l', 'o'])
        'hello'
        >>> chars_to_string(['h', 'e', 'l', 'l', 'o'], add_space=True)
        'h e l l o'
    """
    if add_space:
        return ' '.join(char_list)
    return ''.join(char_list)


def get_random_char(string: str) -> str:
    """
    Select a random character from a string.
    
    Args:
        string: Input string
        
    Returns:
        A randomly selected character
        
    Raises:
        IndexError: If string is empty
        
    Example:
        >>> char = get_random_char("hello")
        >>> char in "hello"
        True
    """
    return random.choice(list(string))


def same_string(string: str) -> str:
    """
    Return the same string unchanged (identity operation).
    
    Args:
        string: Input string
        
    Returns:
        The same string
    """
    return string


def delete_char(string: str, char_to_delete: Optional[str] = None) -> str:
    """
    Delete all occurrences of a character from a string.
    
    Args:
        string: Input string
        char_to_delete: Character to delete (if None, selects random character)
        
    Returns:
        String with all occurrences of the character removed
        
    Example:
        >>> delete_char("hello", "l")
        'heo'
    """
    if char_to_delete is None:
        char_to_delete = get_random_char(string)
    char_list = string_to_chars(string)
    return chars_to_string([char for char in char_list if char != char_to_delete])


def insert_char(string: str, preceding_char: Optional[str] = None, 
                char_to_insert: Optional[str] = None) -> str:
    """
    Insert a character after every occurrence of a specified character.
    
    Args:
        string: Input string
        preceding_char: Character after which to insert (if None, selects random)
        char_to_insert: Character to insert (if None, selects random lowercase letter)
        
    Returns:
        String with inserted character
        
    Example:
        >>> insert_char("hello", "l", "x")
        'helxlxo'
    """
    if preceding_char is None:
        preceding_char = get_random_char(string)
    if char_to_insert is None:
        char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz')

    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        result.append(char)
        if char == preceding_char:
            result.append(char_to_insert)
    return chars_to_string(result)


def substitute_char(string: str, char_to_replace: Optional[str] = None, 
                    char_to_substitute: Optional[str] = None) -> str:
    """
    Replace all occurrences of one character with another.
    
    Args:
        string: Input string
        char_to_replace: Character to replace (if None, selects random)
        char_to_substitute: Replacement character (if None, selects random different char)
        
    Returns:
        String with substituted character
        
    Example:
        >>> substitute_char("hello", "l", "r")
        'herro'
    """
    if char_to_replace is None:
        char_to_replace = get_random_char(string)
    if char_to_substitute is None:
        remaining_chars = 'abcdefghijklmnopqrstuvwxyz'.replace(char_to_replace, '')
        char_to_substitute = get_random_char(remaining_chars)

    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        if char == char_to_replace:
            result.append(char_to_substitute)
        else:
            result.append(char)
    return chars_to_string(result)


def permute_char(string: str, char1: Optional[str] = None, 
                 char2: Optional[str] = None) -> str:
    """
    Swap all occurrences of two characters in a string.
    
    Args:
        string: Input string
        char1: First character to swap (if None, selects random)
        char2: Second character to swap (if None, selects random different char)
        
    Returns:
        String with permuted characters
        
    Example:
        >>> permute_char("hello", "h", "o")
        'oellh'
    """
    if char1 is None:
        char1 = get_random_char(string)
    if char2 is None:
        remaining_string = string.replace(char1, '')
        if remaining_string:
            char2 = get_random_char(remaining_string)
        else:
            char2 = char1

    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        if char == char1:
            result.append(char2)
        elif char == char2:
            result.append(char1)
        else:
            result.append(char)
    return chars_to_string(result)


def duplicate_char(string: str, char_to_duplicate: Optional[str] = None) -> str:
    """
    Duplicate all occurrences of a character in a string.
    
    Args:
        string: Input string
        char_to_duplicate: Character to duplicate (if None, selects random)
        
    Returns:
        String with duplicated character
        
    Example:
        >>> duplicate_char("hello", "l")
        'hellllo'
    """
    if char_to_duplicate is None:
        char_to_duplicate = get_random_char(string)

    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        result.append(char)
        if char == char_to_duplicate:
            result.append(char)
    return chars_to_string(result)


def normalize_diacritic(string: str) -> str:
    """
    Remove diacritics from a string, converting to base characters.
    
    Args:
        string: Input string with potential diacritics
        
    Returns:
        String with diacritics removed
        
    Example:
        >>> normalize_diacritic("á é í ó ú")
        'a e i o u'
    """
    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        if char in diacritic_map:
            result.append(diacritic_map[char])
        else:
            result.append(char)
    return chars_to_string(result)


def diacritize(string: str) -> str:
    """
    Add diacritics to all eligible characters in a string.
    
    For each base character that can have diacritics, randomly selects
    one of the possible diacritic variants.
    
    Args:
        string: Input string with base characters
        
    Returns:
        String with diacritics added to eligible characters
        
    Example:
        >>> result = diacritize("aeiou")
        >>> all(c in "àáâèéêìíîòóôùúû" for c in result)
        True
    """
    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        if char in reverse_diacritic_map:
            result.append(random.choice(reverse_diacritic_map[char]))
        else:
            result.append(char)
    return chars_to_string(result)


def randomly_diacritize(string: str, probability: float = 0.5) -> str:
    """
    Randomly add diacritics to eligible characters with a given probability.
    
    Args:
        string: Input string
        probability: Probability of adding diacritic to each eligible character
        
    Returns:
        String with randomly added diacritics
        
    Example:
        >>> result = randomly_diacritize("aeiou", probability=0.5)
    """
    result = []
    char_list = string_to_chars(string)
    for char in char_list:
        if char in reverse_diacritic_map and random.random() < probability:
            result.append(random.choice(reverse_diacritic_map[char]))
        else:
            result.append(normalize_diacritic(char))
    return chars_to_string(result)


def spell_string(string: str) -> str:
    """
    Spell out a string with spaces between each character.
    
    Args:
        string: Input string
        
    Returns:
        String with characters separated by spaces
        
    Example:
        >>> spell_string("hello")
        'h e l l o'
    """
    return chars_to_string(string_to_chars(string), add_space=True)


def shuffle_chars(char_list: List[str]) -> List[str]:
    """
    Shuffle a list of characters ensuring the result differs from the original.
    
    Args:
        char_list: List of characters to shuffle
        
    Returns:
        Shuffled list of characters (different from input if length > 1)
    """
    if len(char_list) <= 1:
        return char_list
    shuffled_list = char_list[:]
    while True:
        random.shuffle(shuffled_list)
        if shuffled_list != char_list:
            break
    return shuffled_list


def randomly_merge_chars(char_list: List[str]) -> List[str]:
    """
    Randomly merge adjacent characters into multi-character strings.
    
    This function randomly combines adjacent characters (pairs or triplets)
    into single list elements. At least one merge is guaranteed.
    
    Args:
        char_list: List of individual characters
        
    Returns:
        List with some characters merged
        
    Example:
        >>> result = randomly_merge_chars(['h', 'e', 'l', 'l', 'o'])
        >>> len(result) < 5  # Some characters were merged
        True
    """
    merged_list = []
    i = 0
    while i < len(char_list):
        if i < len(char_list) - 1 and random.random() < 0.5:
            merged_list.append(char_list[i] + char_list[i + 1])
            i += 2
        elif i < len(char_list) - 2 and random.random() < 0.5:
            merged_list.append(char_list[i] + char_list[i + 1] + char_list[i + 2])
            i += 3
        else:
            merged_list.append(char_list[i])
            i += 1

    # Ensure at least one merge happened
    if len(merged_list) == len(char_list):
        idx = random.randint(0, len(char_list) - 2)
        merged_list = (
            char_list[:idx]
            + [char_list[idx] + char_list[idx + 1]]
            + char_list[idx + 2:]
        )
    return merged_list


def randomly_insert_char(char_list: List[str]) -> List[str]:
    """
    Insert a random lowercase letter at a random position.
    
    Args:
        char_list: List of characters
        
    Returns:
        List with one additional random character inserted
    """
    idx = random.randint(0, len(char_list))
    char_to_insert = random.choice('abcdefghijklmnopqrstuvwxyz')
    return char_list[:idx] + [char_to_insert] + char_list[idx:]


def randomly_delete_char(char_list: List[str]) -> List[str]:
    """
    Delete a random character from the list.
    
    Args:
        char_list: List of characters
        
    Returns:
        List with one character removed (unchanged if length <= 1)
    """
    if len(char_list) <= 1:
        return char_list
    idx = random.randint(0, len(char_list) - 1)
    return char_list[:idx] + char_list[idx + 1:]


def perturb_string(string: str) -> List[str]:
    """
    Generate multiple perturbed versions of a string for MCQ distractors.
    
    Applies 3 random perturbations from: shuffle, merge, insert, delete.
    Each result is space-separated for readability.
    
    Args:
        string: Input string to perturb
        
    Returns:
        List of 3 perturbed string versions (space-separated)
        
    Example:
        >>> results = perturb_string("hello")
        >>> len(results)
        3
        >>> all(' ' in r for r in results)  # All are space-separated
        True
    """
    char_list = string_to_chars(string)
    perturbation_functions = [
        shuffle_chars,
        randomly_merge_chars,
        randomly_insert_char,
        randomly_delete_char,
    ]
    chosen_functions = random.sample(perturbation_functions, 3)
    results = [chars_to_string(func(char_list), add_space=True) for func in chosen_functions]
    return results
