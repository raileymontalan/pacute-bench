"""
Hierarchical Task Framework for PACUTE

Organizes evaluation tasks into 6 levels that build compositionally,
enabling diagnosis of specific capability failures.

Level 0: Character Recognition - Can the model see individual characters?
Level 1: Character Manipulation - Can it manipulate strings without morphology?
Level 2: Morpheme Decomposition - Can it identify morphological boundaries?
Level 3: Morpheme Manipulation - Can it transform morphological units?
Level 4: Morpheme Composition - Can it build words from morphemes?
Level 5: Complex Morphological Reasoning - Multi-step linguistic operations

Design principle: If a model fails at Level N, we expect failures at Level N+1, N+2, etc.
This creates a diagnostic cascade that pinpoints the source of errors.
"""

import json
import random
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass, field
import pandas as pd

from pacute_bench.utils.strings import (
    spell_string,
    chars_to_string,
    string_to_chars,
)
from pacute_bench.utils.syllabification import syllabify
from pacute_bench.utils.helpers import prepare_mcq_outputs, prepare_gen_outputs


# Simple position-specific operations for hierarchical tasks
def delete_character(word: str, position: int) -> str:
    """Delete character at position (1-indexed)."""
    if position < 1 or position > len(word):
        return word
    return word[:position-1] + word[position:]


def insert_character(word: str, position: int, char: str) -> str:
    """Insert character at position (1-indexed)."""
    if position < 1 or position > len(word) + 1:
        return word
    return word[:position-1] + char + word[position-1:]


def substitute_character(word: str, position: int, new_char: str) -> str:
    """Substitute character at position (1-indexed)."""
    if position < 1 or position > len(word):
        return word
    return word[:position-1] + new_char + word[position:]


def permute_characters(word: str, pos1: int, pos2: int) -> str:
    """Swap characters at two positions (1-indexed)."""
    if pos1 < 1 or pos1 > len(word) or pos2 < 1 or pos2 > len(word):
        return word
    chars = list(word)
    chars[pos1-1], chars[pos2-1] = chars[pos2-1], chars[pos1-1]
    return ''.join(chars)


@dataclass
class HierarchicalTask:
    """A task with hierarchical level metadata."""
    level: int  # 0-5
    category: str  # e.g., "affixation", "composition", "manipulation"
    subcategory: str  # e.g., "character_deletion", "affix_identification"
    prompt_en: str
    prompt_tl: str
    answer: str
    options: Optional[List[str]] = None  # For MCQ format
    word: str = ""  # The word being tested
    metadata: Dict = field(default_factory=dict)  # Additional info for analysis

    def to_mcq_dict(self) -> Dict:
        """Convert to MCQ format dict."""
        if not self.options or len(self.options) != 4:
            raise ValueError("MCQ format requires exactly 4 options")

        return {
            "level": self.level,
            "category": self.category,
            "subcategory": self.subcategory,
            "prompt_en": self.prompt_en,
            "prompt_tl": self.prompt_tl,
            "options": self.options,
            "answer": self.answer,
            "word": self.word,
            "metadata": self.metadata,
        }

    def to_gen_dict(self) -> Dict:
        """Convert to generative format dict."""
        return {
            "level": self.level,
            "category": self.category,
            "subcategory": self.subcategory,
            "prompt_en": self.prompt_en,
            "prompt_tl": self.prompt_tl,
            "answer": self.answer,
            "word": self.word,
            "metadata": self.metadata,
        }


class HierarchicalTaskGenerator:
    """Generator for hierarchical PACUTE tasks."""

    def __init__(self, words_df: pd.DataFrame, affixes_df: Optional[pd.DataFrame] = None):
        """
        Initialize generator.

        Args:
            words_df: DataFrame with columns: word, syllables, stress, etc.
            affixes_df: DataFrame with affix annotations (root, prefix, suffix, etc.)
        """
        self.words_df = words_df
        self.affixes_df = affixes_df

    # ========================================================================
    # LEVEL 0: CHARACTER RECOGNITION
    # ========================================================================

    def generate_level0_character_identification(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 0: Identify a character at a specific position.

        Example: "What is the 3rd character in 'kumain'?" → 'm'
        """
        tasks = []
        words = self.words_df.sample(n)["word"].tolist()

        for word in words:
            if len(word) < 3:
                continue

            position = random.randint(1, len(word))
            answer = word[position - 1]

            prompt_en = f"What is the {position}{'st' if position == 1 else 'nd' if position == 2 else 'rd' if position == 3 else 'th'} character in the word '{word}'?"
            prompt_tl = f"Ano ang ika-{position} na titik sa salitang '{word}'?"

            if format == "mcq":
                # Generate distractors: other characters from the word
                distractors = [c for c in word if c != answer]
                random.shuffle(distractors)
                options = [answer] + distractors[:3]
                while len(options) < 4:
                    options.append(random.choice("abcdefghijklmnopqrstuvwxyz"))
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=0,
                category="recognition",
                subcategory="character_identification",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"position": position}
            ))

        return tasks

    def generate_level0_character_counting(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 0: Count occurrences of a specific character.

        Example: "How many 'a's in 'kumain'?" → 2
        """
        tasks = []
        words = self.words_df.sample(n)["word"].tolist()

        for word in words:
            # Pick a character that appears at least once
            char_counts = {}
            for c in word:
                char_counts[c] = char_counts.get(c, 0) + 1

            char = random.choice(list(char_counts.keys()))
            count = char_counts[char]
            answer = str(count)

            prompt_en = f"How many '{char}' characters are in the word '{word}'?"
            prompt_tl = f"Ilang titik na '{char}' ang nasa salitang '{word}'?"

            if format == "mcq":
                # Generate distractors: count ± 1, count ± 2
                distractors = [count - 2, count - 1, count + 1, count + 2]
                distractors = [str(d) for d in distractors if d >= 0 and d != count]
                options = [answer] + random.sample(distractors, min(3, len(distractors)))
                while len(options) < 4:
                    options.append(str(random.randint(0, len(word))))
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=0,
                category="recognition",
                subcategory="character_counting",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"target_char": char, "count": count}
            ))

        return tasks

    def generate_level0_character_presence(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 0: Check if a character is present in word.

        Example: "Does 'kumain' contain 'u'?" → Yes
        """
        tasks = []
        words = self.words_df.sample(n * 2)["word"].tolist()  # Sample extra for balance

        # Balance positive and negative examples
        for i, word in enumerate(words[:n]):
            if i % 2 == 0:
                # Positive example: char is in word
                char = random.choice(word)
                answer = "Yes"
                answer_tl = "Oo"
            else:
                # Negative example: char not in word
                all_chars = set("abcdefghijklmnopqrstuvwxyzáàâéèêíìîóòôúùû")
                not_in_word = list(all_chars - set(word.lower()))
                if not not_in_word:
                    continue
                char = random.choice(not_in_word)
                answer = "No"
                answer_tl = "Hindi"

            prompt_en = f"Does the word '{word}' contain the character '{char}'?"
            prompt_tl = f"May titik na '{char}' ba sa salitang '{word}'?"

            if format == "mcq":
                options = ["Yes", "No"]
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=0,
                category="recognition",
                subcategory="character_presence",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"target_char": char, "present": answer == "Yes"}
            ))

        return tasks[:n]

    # ========================================================================
    # LEVEL 1: CHARACTER MANIPULATION
    # ========================================================================

    def generate_level1_character_deletion(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 1: Delete a character at position.

        Example: "Delete the 3rd character from 'kumain'" → "kuain"

        Requires: Level 0 (character identification)
        """
        tasks = []
        words = self.words_df.sample(n)["word"].tolist()

        for word in words:
            if len(word) < 4:
                continue

            position = random.randint(1, len(word))
            answer = delete_character(word, position)

            prompt_en = f"Delete the {position}{'st' if position == 1 else 'nd' if position == 2 else 'rd' if position == 3 else 'th'} character from '{word}'."
            prompt_tl = f"Tanggalin ang ika-{position} na titik mula sa salitang '{word}'."

            if format == "mcq":
                # Generate distractors: delete different positions
                distractors = []
                for pos in range(1, len(word) + 1):
                    if pos != position:
                        distractors.append(delete_character(word, pos))
                options = [answer] + random.sample(distractors, min(3, len(distractors)))
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=1,
                category="manipulation",
                subcategory="character_deletion",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"position": position, "operation": "delete"}
            ))

        return tasks

    def generate_level1_character_insertion(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 1: Insert a character at position.

        Example: "Insert 'l' at position 3 in 'kumain'" → "kulain"
        """
        tasks = []
        words = self.words_df.sample(n)["word"].tolist()

        for word in words:
            position = random.randint(1, len(word) + 1)
            char_to_insert = random.choice("abcdefghilmnoprstuy")
            answer = insert_character(word, position, char_to_insert)

            prompt_en = f"Insert '{char_to_insert}' at position {position} in the word'{word}'."
            prompt_tl = f"Ilagay ang titik na '{char_to_insert}' sa ika-{position} na posisyon sa salitang '{word}'."

            if format == "mcq":
                # Generate distractors: insert at different positions
                distractors = []
                for pos in range(1, len(word) + 2):
                    if pos != position:
                        distractors.append(insert_character(word, pos, char_to_insert))
                options = [answer] + random.sample(distractors, min(3, len(distractors)))
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=1,
                category="manipulation",
                subcategory="character_insertion",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"position": position, "char": char_to_insert, "operation": "insert"}
            ))

        return tasks

    def generate_level1_character_substitution(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 1: Substitute a character at position.

        Example: "Replace the 2nd character in 'kumain' with 'l'" → "klmain"
        """
        tasks = []
        words = self.words_df.sample(n)["word"].tolist()

        for word in words:
            if len(word) < 3:
                continue

            position = random.randint(1, len(word))
            old_char = word[position - 1]
            new_char = random.choice([c for c in "abcdefghilmnoprstuy" if c != old_char])
            answer = substitute_character(word, position, new_char)

            prompt_en = f"Replace the {position}{'st' if position == 1 else 'nd' if position == 2 else 'rd' if position == 3 else 'th'} character in the word '{word}' with the character '{new_char}'."
            prompt_tl = f"Palitan ang ika-{position} na titik sa salitang '{word}' gamit ng titik na '{new_char}'."

            if format == "mcq":
                # Generate distractors: substitute at different positions or with different chars
                distractors = []
                for pos in range(1, len(word) + 1):
                    if pos != position:
                        distractors.append(substitute_character(word, pos, new_char))
                for c in "abcdefghilmnoprstuy":
                    if c != new_char:
                        distractors.append(substitute_character(word, position, c))
                options = [answer] + random.sample(distractors, min(3, len(distractors)))
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=1,
                category="manipulation",
                subcategory="character_substitution",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"position": position, "old_char": old_char, "new_char": new_char, "operation": "substitute"}
            ))

        return tasks

    def generate_level1_character_permutation(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 1: Swap two characters.

        Example: "Swap positions 2 and 4 in 'kumain'" → "kmauin"
        """
        tasks = []
        # Filter for words with length >= 4 first, then sample
        valid_words = self.words_df[self.words_df["word"].str.len() >= 4]
        if len(valid_words) < n:
            raise ValueError(f"Not enough words with length >= 4. Need {n}, have {len(valid_words)}")
        words = valid_words.sample(n)["word"].tolist()

        for word in words:

            pos1, pos2 = random.sample(range(1, len(word) + 1), 2)
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1

            answer = permute_characters(word, pos1, pos2)

            prompt_en = f"Swap characters at positions {pos1} and {pos2} in the word '{word}'."
            prompt_tl = f"Ipagpalit ang mga titik sa posisyon {pos1} at {pos2} sa salitang '{word}'."

            if format == "mcq":
                # Generate distractors: swap different positions
                distractors = []
                for _ in range(6):
                    p1, p2 = random.sample(range(1, len(word) + 1), 2)
                    if (p1, p2) != (pos1, pos2) and (p2, p1) != (pos1, pos2):
                        distractors.append(permute_characters(word, p1, p2))
                options = [answer] + random.sample(distractors, min(3, len(distractors)))
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=1,
                category="manipulation",
                subcategory="character_permutation",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"pos1": pos1, "pos2": pos2, "operation": "permute"}
            ))

        return tasks

    # ========================================================================
    # LEVEL 2: MORPHEME DECOMPOSITION
    # ========================================================================

    def generate_level2_affix_identification(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 2: Identify the affix in a word.

        Example: "What is the infix in 'kumain'?" → "um"

        Requires: Understanding of morphological boundaries
        """
        if self.affixes_df is None:
            raise ValueError("affixes_df required for affix identification tasks")

        tasks = []
        affixed_words = self.affixes_df.sample(n)

        for _, row in affixed_words.iterrows():
            word = row["word"]

            # Determine which affix type to ask about
            affix_types = []
            if pd.notna(row.get("prefix")):
                affix_types.append(("prefix", row["prefix"]))
            if pd.notna(row.get("infix")):
                affix_types.append(("infix", row["infix"]))
            if pd.notna(row.get("suffix")):
                affix_types.append(("suffix", row["suffix"]))

            if not affix_types:
                continue

            affix_type, affix = random.choice(affix_types)
            answer = affix

            prompt_en = f"What is the {affix_type} in the word'{word}'?"
            prompt_tl = f"Ano ang {affix_type} sa salitang '{word}'?"

            if format == "mcq":
                # Generate distractors: other affixes of same type
                all_affixes = set()
                for _, r in self.affixes_df.iterrows():
                    if pd.notna(r.get(affix_type)):
                        all_affixes.add(r[affix_type])
                distractors = list(all_affixes - {affix})
                options = [answer] + random.sample(distractors, min(3, len(distractors)))
                while len(options) < 4:
                    options.append(random.choice(["pa", "ka", "ma", "na", "ba", "ta", "la"]))
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=2,
                category="decomposition",
                subcategory=f"affix_identification_{affix_type}",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"affix_type": affix_type, "affix": affix, "root": row.get("root", "")}
            ))

        return tasks

    def generate_level2_root_extraction(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 2: Extract the root from an affixed word.

        Example: "What is the root of 'nagluto'?" → "luto"
        """
        if self.affixes_df is None:
            raise ValueError("affixes_df required for root extraction tasks")

        tasks = []
        affixed_words = self.affixes_df[self.affixes_df["root"].notna()].sample(n)

        for _, row in affixed_words.iterrows():
            word = row["word"]
            answer = row["root"]

            prompt_en = f"What is the root word of the word '{word}'?"
            prompt_tl = f"Ano ang salitang-ugat ng salitang '{word}'?"

            if format == "mcq":
                # Generate distractors: roots of other words
                other_roots = self.affixes_df[self.affixes_df["root"] != answer]["root"].dropna().tolist()
                distractors = random.sample(other_roots, min(3, len(other_roots)))
                options = [answer] + distractors
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=2,
                category="decomposition",
                subcategory="root_extraction",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"root": answer, "affixes": {
                    "prefix": row.get("prefix", ""),
                    "infix": row.get("infix", ""),
                    "suffix": row.get("suffix", "")
                }}
            ))

        return tasks

    def generate_level2_syllable_counting(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 2: Count syllables in a word.

        Example: "How many syllables in 'kumain'?" → "3" (ku-ma-in)
        """
        tasks = []
        words = self.words_df[self.words_df["syllables"].notna()].sample(n)

        for _, row in words.iterrows():
            word = row["word"]
            syllables = syllabify(word)
            count = len(syllables)
            answer = str(count)

            prompt_en = f"How many syllables are in the word '{word}'?"
            prompt_tl = f"Ilang pantig ang nasa salitang '{word}'?"

            if format == "mcq":
                # Generate distractors: count ± 1
                distractors = [str(count - 1), str(count + 1)]
                if count > 2:
                    distractors.append(str(count - 2))
                if count > 1:
                    distractors.append(str(count + 2))
                distractors = [d for d in distractors if int(d) > 0]
                options = [answer] + random.sample(distractors, min(3, len(distractors)))
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=2,
                category="decomposition",
                subcategory="syllable_counting",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"syllables": syllables, "count": count}
            ))

        return tasks

    # ========================================================================
    # LEVEL 3: MORPHEME MANIPULATION
    # ========================================================================

    def generate_level3_affix_removal(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 3: Remove an affix from a word.

        Example: "Remove the infix from 'kumain'" → "kain"

        Requires: Level 2 (identify affix) + Level 1 (delete characters)
        """
        if self.affixes_df is None:
            raise ValueError("affixes_df required for affix removal tasks")

        tasks = []
        affixed_words = self.affixes_df.sample(n)

        for _, row in affixed_words.iterrows():
            word = row["word"]
            root = row.get("root", "")

            if not root:
                continue

            # Determine which affix to remove
            affix_types = []
            if pd.notna(row.get("prefix")):
                affix_types.append("prefix")
            if pd.notna(row.get("infix")):
                affix_types.append("infix")
            if pd.notna(row.get("suffix")):
                affix_types.append("suffix")

            if not affix_types:
                continue

            affix_type = random.choice(affix_types)

            # For simplicity, if removing all affixes, answer is root
            # This is a simplification - proper implementation would need
            # to compute result of removing specific affix
            answer = root

            prompt_en = f"Remove the {affix_type} from the word '{word}'."
            prompt_tl = f"Tanggalin ang {affix_type} mula sa salitang '{word}'."

            if format == "mcq":
                # Generate distractors: other roots
                other_roots = self.affixes_df[self.affixes_df["root"] != root]["root"].dropna().tolist()
                distractors = random.sample(other_roots, min(3, len(other_roots)))
                options = [answer] + distractors
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=3,
                category="manipulation",
                subcategory=f"affix_removal_{affix_type}",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"affix_type": affix_type, "original_word": word, "operation": "remove_affix"}
            ))

        return tasks

    # ========================================================================
    # LEVEL 4: MORPHEME COMPOSITION
    # ========================================================================

    def generate_level4_affix_application(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 4: Apply an affix to a root word.

        Example: "Add infix 'um' to 'kain'" → "kumain"

        Requires: Level 2 (understand morphology) + inverse of Level 3
        """
        if self.affixes_df is None:
            raise ValueError("affixes_df required for affix application tasks")

        tasks = []
        affixed_words = self.affixes_df.sample(n)

        for _, row in affixed_words.iterrows():
            word = row["word"]
            root = row.get("root", "")

            if not root:
                continue

            # Determine which affix to apply
            affix_types = []
            if pd.notna(row.get("prefix")):
                affix_types.append(("prefix", row["prefix"]))
            if pd.notna(row.get("infix")):
                affix_types.append(("infix", row["infix"]))
            if pd.notna(row.get("suffix")):
                affix_types.append(("suffix", row["suffix"]))

            if not affix_types:
                continue

            affix_type, affix = random.choice(affix_types)
            answer = word

            prompt_en = f"Add {affix_type} '{affix}' to the root word '{root}'."
            prompt_tl = f"Ikabit ang {affix_type} na '{affix}' sa salitang-ugat na '{root}'."

            if format == "mcq":
                # Generate distractors: other words with same root or affix
                distractors = []
                for _, r in self.affixes_df.iterrows():
                    if r["word"] != word and (r.get("root") == root or r.get(affix_type) == affix):
                        distractors.append(r["word"])
                options = [answer] + random.sample(distractors, min(3, len(distractors)))
                while len(options) < 4:
                    options.append(affix + root)  # Simple concatenation as distractor
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=4,
                category="composition",
                subcategory=f"affix_application_{affix_type}",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word,
                metadata={"affix_type": affix_type, "affix": affix, "root": root, "operation": "apply_affix"}
            ))

        return tasks

    # ========================================================================
    # LEVEL 5: COMPLEX MORPHOLOGICAL REASONING
    # ========================================================================

    def generate_level5_multi_step_transformation(
        self, n: int, format: Literal["mcq", "gen"] = "gen"
    ) -> List[HierarchicalTask]:
        """
        Level 5: Multi-step morphological operations.

        Example: "Extract the root from 'nagluto', then add suffix '-an'" → "lutuan"

        Requires: Composition of multiple lower-level operations
        """
        if self.affixes_df is None:
            raise ValueError("affixes_df required for multi-step tasks")

        tasks = []
        affixed_words = self.affixes_df[self.affixes_df["root"].notna()].sample(n * 2)

        for i in range(0, len(affixed_words) - 1, 2):
            row1 = affixed_words.iloc[i]
            row2 = affixed_words.iloc[i + 1]

            word1 = row1["word"]
            root1 = row1["root"]

            # Use affix from word2
            affix_types = []
            if pd.notna(row2.get("prefix")):
                affix_types.append(("prefix", row2["prefix"]))
            if pd.notna(row2.get("suffix")):
                affix_types.append(("suffix", row2["suffix"]))

            if not affix_types:
                continue

            affix_type, affix = random.choice(affix_types)

            # The answer is approximately: root1 + affix
            # This is simplified - proper implementation would need proper morphological rules
            if affix_type == "prefix":
                answer = affix + root1
            else:  # suffix
                answer = root1 + affix

            prompt_en = f"Extract the root from the word '{word1}', then add {affix_type} '{affix}'."
            prompt_tl = f"Kunin ang ugat mula sa salitang '{word1}', pagkatapos ay ikabit ang {affix_type} na '{affix}'."

            if format == "mcq":
                # Generate distractors
                distractors = [
                    word1,  # Original word (didn't transform)
                    root1,  # Just the root (incomplete transformation)
                    affix + word1 if affix_type == "prefix" else word1 + affix,  # Applied to wrong form
                ]
                options = [answer] + distractors[:3]
                random.shuffle(options)
            else:
                options = None

            tasks.append(HierarchicalTask(
                level=5,
                category="reasoning",
                subcategory="multi_step_transformation",
                prompt_en=prompt_en,
                prompt_tl=prompt_tl,
                answer=answer,
                options=options,
                word=word1,
                metadata={
                    "original_word": word1,
                    "root": root1,
                    "target_affix_type": affix_type,
                    "target_affix": affix,
                    "operation": "extract_root_then_apply_affix"
                }
            ))

        return tasks[:n]

    # ========================================================================
    # GENERATION API
    # ========================================================================

    def generate_all_levels(
        self,
        n_per_subcategory: int = 20,
        format: Literal["mcq", "gen", "both"] = "both"
    ) -> Dict[int, List[HierarchicalTask]]:
        """
        Generate complete hierarchical task suite.

        Returns:
            Dictionary mapping level (0-5) to list of tasks
        """
        tasks_by_level = {i: [] for i in range(6)}

        formats = ["mcq", "gen"] if format == "both" else [format]

        for fmt in formats:
            # Level 0
            tasks_by_level[0].extend(self.generate_level0_character_identification(n_per_subcategory, fmt))
            tasks_by_level[0].extend(self.generate_level0_character_counting(n_per_subcategory, fmt))
            tasks_by_level[0].extend(self.generate_level0_character_presence(n_per_subcategory, fmt))

            # Level 1
            tasks_by_level[1].extend(self.generate_level1_character_deletion(n_per_subcategory, fmt))
            tasks_by_level[1].extend(self.generate_level1_character_insertion(n_per_subcategory, fmt))
            tasks_by_level[1].extend(self.generate_level1_character_substitution(n_per_subcategory, fmt))
            tasks_by_level[1].extend(self.generate_level1_character_permutation(n_per_subcategory, fmt))

            # Level 2
            if self.affixes_df is not None:
                tasks_by_level[2].extend(self.generate_level2_affix_identification(n_per_subcategory, fmt))
                tasks_by_level[2].extend(self.generate_level2_root_extraction(n_per_subcategory, fmt))
            tasks_by_level[2].extend(self.generate_level2_syllable_counting(n_per_subcategory, fmt))

            # Level 3
            if self.affixes_df is not None:
                tasks_by_level[3].extend(self.generate_level3_affix_removal(n_per_subcategory, fmt))

            # Level 4
            if self.affixes_df is not None:
                tasks_by_level[4].extend(self.generate_level4_affix_application(n_per_subcategory, fmt))

            # Level 5
            if self.affixes_df is not None:
                tasks_by_level[5].extend(self.generate_level5_multi_step_transformation(n_per_subcategory, fmt))

        return tasks_by_level

    def save_tasks(self, tasks_by_level: Dict[int, List[HierarchicalTask]], output_dir: str, format: str = "mcq"):
        """Save tasks to JSONL files organized by level."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        for level, tasks in tasks_by_level.items():
            output_file = os.path.join(output_dir, f"{format}_level{level}.jsonl")
            with open(output_file, "w") as f:
                for task in tasks:
                    if format == "mcq":
                        f.write(json.dumps(task.to_mcq_dict()) + "\n")
                    else:
                        f.write(json.dumps(task.to_gen_dict()) + "\n")
            print(f"Saved {len(tasks)} tasks to {output_file}")
