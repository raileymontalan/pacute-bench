# Patok tokenizer

Morphology-aware expand/contract tokenization for Filipino, used in the PACUTE paper's continued-pretraining (CPT) experiments. This module is preprocessing-time only — it transforms a sequence of token IDs before they're fed to training, and does not modify model architecture, vocabulary, or the training objective. No cluster or infrastructure dependencies.

Requires the `tokenization` extra: `pip install -e ".[tokenization]"` (adds `pyahocorasick`).

## Mechanism

1. **Contract**: randomly merge 2–4 adjacent BPE tokens back into a larger substring, preferentially skipping or re-splitting spans that would obscure a known Filipino affix (`affix_awareness`, default 0.95).
2. **Re-expand**: split the contracted substring by (a) peeling off any matched affix at its correct string boundary, (b) peeling off syllable-reduplication patterns (repeated CV pairs, e.g. `gaganda` → `ga` + `ganda`), then (c) re-tokenizing the residual with the base tokenizer.
3. **Stochastic expansion**: a final unconstrained expansion pass over non-affix tokens (same mechanism as StochasTok).

The affix inventory (`affixes/{prefix,infix,suffix}.txt`) is matched via an Aho-Corasick automaton for efficient multi-pattern search.

## Usage

```python
from transformers import AutoTokenizer
from pacute_bench.tokenization import MorphologyAwarePatokProcessor

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")

processor = MorphologyAwarePatokProcessor(
    tokenizer,
    prefix_file="src/pacute_bench/tokenization/affixes/prefix.txt",
    infix_file="src/pacute_bench/tokenization/affixes/infix.txt",
    suffix_file="src/pacute_bench/tokenization/affixes/suffix.txt",
)

token_ids = tokenizer.encode("kumumakain")
patok_ids = processor.contract_expand(token_ids)
print(processor.decode_tokens(patok_ids))
```

On first use for a given tokenizer, the processor builds a token-expansion table from the tokenizer's vocabulary (`build_expansions`) and caches it to `data/expansions/expansions_<tokenizer_name>.json` relative to the project root — this is a one-time cost per tokenizer (a few minutes for large vocabularies), not shipped in this repo to keep it lightweight.

## Not included here

The NeMo/PBS continued-pretraining pipeline that uses this processor during CPT is intentionally not part of this repo (see the paper's Limitations section) — it contains cluster-specific infrastructure code unrelated to the tokenization algorithm itself.
