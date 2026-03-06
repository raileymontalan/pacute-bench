"""
Constants for pacute-bench evaluation.
"""

from typing import Dict, Set

# ============================================================================
# MCQ Constants
# ============================================================================

MCQ_LABEL_MAP: Dict[int, str] = {0: "A", 1: "B", 2: "C", 3: "D"}
NUM_MCQ_OPTIONS: int = 4
NUM_INCORRECT_OPTIONS: int = 3

# ============================================================================
# Word Length Thresholds
# ============================================================================

MIN_WORD_LENGTH_COMPOSITION: int = 3
MIN_WORD_LENGTH_MANIPULATION: int = 5
MIN_WORD_LENGTH_SYLLABIFICATION: int = 3
MIN_WORD_LENGTH_GENERAL_SYLLABLE_COUNTING: int = 7

# ============================================================================
# Affix Types
# ============================================================================

AFFIX_TYPES: list = ["prefix", "suffix", "infix", "circumfix"]

# ============================================================================
# Diacritics and Character Sets
# ============================================================================

VOWELS: Set[str] = set("AEIOUaeiouÀÁÂÈÉÊÌÍÎÒÓÔÙÚÛàáâèéêìíîòóôùúû")
DIACRITICS: Set[str] = set("ÀÁÂÈÉÊÌÍÎÒÓÔÙÚÛàáâèéêìíîòóôùúûÑñ")
ACCENTED_VOWELS: Set[str] = set("ÀÁÂÈÉÊÌÍÎÒÓÔÙÚÛàáâèéêìíîòóôùúû")
UPPERCASE_LETTERS: Set[str] = set("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")
UPPERCASE_DIACRITICS: Set[str] = set("ÀÁÂÈÉÊÌÍÎÒÓÔÙÚÛ")

MABILIS: Set[str] = set("ÁÉÍÓÚáéíóú")   # Acute accent
MALUMI: Set[str] = set("ÀÈÌÒÙàèìòù")    # Grave accent
MARAGSA: Set[str] = set("ÂÊÎÔÛâêîôû")   # Circumflex accent

# ============================================================================
# Diacritic Mappings
# ============================================================================

DIACRITIC_MAP: Dict[str, str] = {
    'á': 'a', 'à': 'a', 'â': 'a',
    'é': 'e', 'è': 'e', 'ê': 'e',
    'í': 'i', 'ì': 'i', 'î': 'i',
    'ó': 'o', 'ò': 'o', 'ô': 'o',
    'ú': 'u', 'ù': 'u', 'û': 'u',
    'ñ': 'n',
    'Á': 'A', 'À': 'A', 'Â': 'A',
    'É': 'E', 'È': 'E', 'Ê': 'E',
    'Í': 'I', 'Ì': 'I', 'Î': 'I',
    'Ó': 'O', 'Ò': 'O', 'Ô': 'O',
    'Ú': 'U', 'Ù': 'U', 'Û': 'U',
    'Ñ': 'N',
}

REVERSE_DIACRITIC_MAP: Dict[str, list] = {
    'a': ['á', 'à', 'â'],
    'e': ['é', 'è', 'ê'],
    'i': ['í', 'ì', 'î'],
    'o': ['ó', 'ò', 'ô'],
    'u': ['ú', 'ù', 'û'],
    'n': ['ñ'],
    'A': ['Á', 'À', 'Â'],
    'E': ['É', 'È', 'Ê'],
    'I': ['Í', 'Ì', 'Î'],
    'O': ['Ó', 'Ò', 'Ô'],
    'U': ['Ú', 'Ù', 'Û'],
    'N': ['Ñ'],
}

# ============================================================================
# Filipino Phonology
# ============================================================================

LETTER_PAIRS: Set[str] = set(["bl", "br", "dr", "pl", "tr"])

# ============================================================================
# Stress Classification
# ============================================================================

STRESS_PRONUNCIATION_MAP: Dict[str, str] = {
    "mabilis": "mabilis (acute: á)",
    "malumi": "malumi (grave: à)",
    "maragsa": "maragsa (circumflex: â)",
    "malumay": "malumay (unmarked)"
}

# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_FREQUENCY_FILE: str = 'data/corpora/pacute_data/word_frequencies.csv'
DEFAULT_RANK_FILLNA: int = 100000
DEFAULT_FREQ_WEIGHT: float = 0.5
DEFAULT_RANDOM_STATE: int = 42
