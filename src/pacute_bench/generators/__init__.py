"""
Dataset generators for pacute-bench benchmarks.
"""

from .composition import create_composition_dataset, create_corpus_composition_dataset
from .manipulation import create_manipulation_dataset
from .morphological_extraction import create_morphological_extraction_dataset
from .morphological_production import create_morphological_production_dataset
from .syllabification import create_syllabification_dataset
from .hierarchical import HierarchicalTaskGenerator

__all__ = [
    "create_composition_dataset",
    "create_corpus_composition_dataset",
    "create_manipulation_dataset",
    "create_morphological_extraction_dataset",
    "create_morphological_production_dataset",
    "create_syllabification_dataset",
    "HierarchicalTaskGenerator",
]
