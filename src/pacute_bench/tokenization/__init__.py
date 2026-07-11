"""
Patok: morphology-aware expand/contract tokenization for Filipino.

This is a standalone carve-out of the Patok tokenizer used in the PACUTE
paper's continued-pretraining experiments (see Table on the CPT results in
the paper). It is a data-preprocessing-time tokenization intervention, not a
training loop: it does not modify model architecture, vocabulary, or the
training objective, and has no cluster/infrastructure dependencies.

Main class:
- MorphologyAwarePatokProcessor: Aho-Corasick-based affix-aware
  contract/expand tokenization.

See README.md in this directory for a minimal usage example.
"""

from .patok_morphology import MorphologyAwarePatokProcessor

__all__ = ["MorphologyAwarePatokProcessor"]
