# pacute-bench

A self-contained evaluation suite for Filipino morphology benchmarks, designed to run against a [vLLM](https://github.com/vllm-project/vllm) server via its OpenAI-compatible API.

## Benchmarks

### Main benchmark

**PACUTE** (**P**ilipino **A**ffix and **C**haracter-Level **U**nderstanding of **T**okens **E**valuation) is the primary benchmark in this suite, designed to evaluate Filipino morphological and phonological understanding.

| Category | Task | GEN samples | MCQ samples |
|---|---|---|---|
| **PACUTE-Affixation** | Identify/apply Filipino affix inflections (prefix, suffix, infix, circumfix) | 140 | 140 |
| **PACUTE-Composition** | Character-level composition: spelling, counting, length, diacritics | 500 | 900 |
| **PACUTE-Manipulation** | String operations on Filipino words (deletion, insertion, substitution, …) | 800 | 800 |
| **PACUTE-Syllabification** | Syllable counting, stress identification/disambiguation, reduplication | 500 | 500 |

**Generative (GEN)** is the primary evaluation format: the model produces a free-form answer that is scored against the ground truth using exact, contains, and prefix match. **Multiple-choice (MCQ)** is provided as a alternative for base pretrained models that cannot reliably follow generation instructions; it uses log-probability scoring over a fixed option set, requiring no generation.

### Supplementary benchmark

| Category | Task | GEN samples | MCQ samples |
|---|---|---|---|
| **Hierarchical** | 6-level diagnostic cascade (character → morpheme → composition) | 600 | 600 |

The Hierarchical benchmark complements PACUTE by structuring tasks across compositional dependency levels, enabling fine-grained diagnosis of where model capabilities break down.

### Reference benchmarks

The following are existing benchmarks from other researchers, included to situate PACUTE results in a broader context.

| Name | Task | GEN samples | MCQ samples | Source |
|---|---|---|---|---|
| **LangGame** | Word-level reasoning (longest, contains letter, starts/ends with …) | 1,000 | 1,000 | Sims et al. (2025) |
| **Multi-digit Addition** | 3-digit arithmetic (tests numeral tokenization) | 1,000 | 1,000 | Sims et al. (2025) |
| **CUTE** | Character-level understanding (spell, insert, delete, swap, sub …) | 1,400 | — | Edman et al. (2024) |

- Sims, A., Foster, T., Kaleb, K., Nguyen, T.-D. H., Lee, J., Foerster, J. N., Teh, Y. W., & Lu, C. (2025). *StochasTok: Improving Fine-Grained Subword Understanding in LLMs.* arXiv:2506.01687. https://arxiv.org/abs/2506.01687
- Edman, L., Schmid, H., & Fraser, A. (2024). *CUTE: Measuring LLMs' Understanding of Their Tokens.* In Proceedings of EMNLP 2024, pp. 3017–3026. https://aclanthology.org/2024.emnlp-main.177/

## Task Details

### PACUTE-Affixation

Tests understanding and application of Filipino affix morphology. Each sample belongs to one of two task types — **Inflection** (`affix_inflection`: given a root word and an affix, produce the inflected form) or **Identification** (`affix_identification`: given an inflected word, name the affix that was applied) — and one of four affix subcategories:

| Subcategory | Inflection example | Identification example | GEN samples | MCQ samples |
|---|---|---|---|---|
| `prefix` | `um-` + `inom` → `uminom` | Which affix is in `uminom`? → `um-` | 40 | 40 |
| `suffix` | `bukas` + `-an` → `buksan` | Which affix is in `buksan`? → `-an` | 40 | 40 |
| `infix` | `-in-` + `kain` → `kinain` | Which affix is in `kinain`? → `-in-` | 40 | 40 |
| `circumfix` | `pag-` + `sayaw` + `-an` → `pagsayawan` | Which affixes are in `pagsayawan`? → `pag-`, `-an` | 20 | 20 |

GEN and MCQ counts are split evenly between the two task types. MCQ distractors are sourced from affixes with high Levenshtein similarity to make options plausible.

---

### PACUTE-Composition

Tests character-level composition skills — understanding the internal structure of Filipino words without morphological transformations. Samples are drawn from a corpus of Filipino words.

| Subcategory | Description | Example | GEN samples | MCQ samples |
|---|---|---|---|---|
| `spelling` | Spell out a word with spaces between each character | `sila` → `s i l a` | 100 | 100 |
| `character` / `char_exactly` | How many occurrences of a given character are in the word? | How many `a`s in `sila`? → `1` | 100 | 100 |
| `char_most` | Which of four words has the most occurrences of a given character? | Which has the most `a`s? | — | 100 |
| `char_least` | Which of four words has the fewest occurrences of a given character? | Which has the fewest `a`s? | — | 100 |
| `length` / `length_exactly` | How many characters does the word have? | How many characters in `sila`? → `4` | 100 | 100 |
| `length_most` | Which of four words is the longest? | Which word is longest? | — | 100 |
| `length_least` | Which of four words is the shortest? | Which word is shortest? | — | 100 |
| `diacritic` / `diacritic_exactly` | How many diacritics (tuldik) does the word contain? | How many diacritics in `silà`? → `1` | 100 | 100 |
| `uppercase` / `uppercase_exactly` | How many uppercase characters does the word contain? | How many uppercase in `Hindi`? → `1` | 100 | 100 |

GEN tasks ask the model to produce a count or character sequence directly (subcategory names: `spelling`, `character`, `length`, `diacritic`, `uppercase`). The comparative MCQ-only tasks (`char_most`, `char_least`, `length_most`, `length_least`) do not have a GEN counterpart because the question changes form between formats.

---

### PACUTE-Manipulation

Tests the ability to apply specific character-level transformations to Filipino words. Each sample specifies a target word and the operation to perform. MCQ distractors are generated by applying the correct operation with wrong parameters or a different operation entirely.

| Subcategory | Description | Example | GEN samples | MCQ samples |
|---|---|---|---|---|
| `deletion` | Delete a specified character from the word | Delete `m` from `kumain` → `kuain` | 100 | 100 |
| `insertion` | Insert a character after a specified position | Insert `l` after `u` in `kumain` → `kulmain` | 100 | 100 |
| `substitution` | Replace one character with another | Replace `m` with `l` in `kumain` → `kulain` | 100 | 100 |
| `permutation` | Swap two specified characters | Swap `k` and `m` in `kumain` → `mukain` | 100 | 100 |
| `duplication` | Duplicate a specified character | Duplicate `a` in `kumain` → `kumaain` | 100 | 100 |
| `uppercasing` | Convert the word to uppercase | `kumain` → `KUMAIN` | 100 | 100 |
| `lowercasing` | Convert the word to lowercase | `KUMAIN` → `kumain` | 100 | 100 |
| `diacritic_normalization` | Strip diacritic marks (tuldik) from the word | `kumáin` → `kumain` | 100 | 100 |

---

### PACUTE-Syllabification

Tests phonological and prosodic awareness of Filipino words. There are **5 task types**, each with 100 GEN samples (and 100 MCQ samples where applicable), covering syllable counting and stress:

| Subcategory | Description | Example | GEN samples | MCQ samples |
|---|---|---|---|---|
| `stress_identification` | Given sentence context, identify which syllable of the target word carries the stress | *Punong-puno ng tubig at putik ang mga bahay.* — which syllable of `puno` has the stress? → `no` | 100 | 100 (2-option) |
| `stress_disambiguation` | Given sentence context, write the word with the correct diacritic marks (tuldik) | *Pag-asa ang huling namamatay.* — write `huli` with diacritics → `hulí` | 100 | 100 |
| `reduplication_identification` | Name the syllable being reduplicated in a given word | What is the reduplicated syllable in `babae`? → `ba` | 100 | — |
| `reduplication_detection` | Identify which of four words exhibits CV-reduplication | Which word has CV-reduplication? → `babae` | — | 100 |
| `ng_aware_syllable_counting` | Count syllables in words containing `ng`, testing awareness that `ng` is a single consonant digraph | How many syllables in `galing`? → `2` | 100 | 100 |
| `general_syllable_counting` | Count syllables in longer words (≥ 7 characters) | How many syllables in `bulubundukin`? → `5` | 100 | 100 |

> **Note on stress tasks:** Stress identification and disambiguation are context-dependent — they use sentence-level context from a curated corpus to resolve stress ambiguity. The MCQ stress identification task presents only 2 options (the stressed syllable and one other from the same word).

---

### Hierarchical

A diagnostic benchmark that organises tasks into 6 compositional levels. Each level builds on the previous, creating a cascade that makes it easy to pinpoint where a model's capabilities break down.

| Level | Capability | Description | Example | GEN samples | MCQ samples |
|---|---|---|---|---|---|
| 0 | Character Recognition | Identify a character at a specific position | "What is the 3rd character in `kumain`?" → `m` | 100 | 100 |
| 1 | Character Manipulation | Perform simple string edits | "Delete the 3rd character in `kumain`" → `kuain` | 100 | 100 |
| 2 | Morpheme Decomposition | Identify morphological boundaries | "What is the root of `kumain`?" → `kain` | 100 | 100 |
| 3 | Morpheme Manipulation | Transform morphological units | "Change `-um-` to `mag-` in `kumain`" → `magkain` | 100 | 100 |
| 4 | Morpheme Composition | Combine morphemes into well-formed words | "Combine `ka-` + `alis` + `-an`" → `kaalisan` | 100 | 100 |
| 5 | Complex Morphological Reasoning | Multi-step linguistic operations | Apply focus markers, combine affixes | 100 | 100 |

Each level has 100 GEN and 100 MCQ samples. If a model fails at level N, failures at N+1, N+2, … are expected due to the compositional dependency structure.

---

### LangGame

Tests subword understanding through word games. Each sample presents 4 candidate words and asks a question about their surface properties. Total of 1,000 samples.

Question types: **most** (most occurrences of a character), **contains** (contains a substring), **starts** (starts with a string), **ends** (ends with a string), **longest** (longest word), **shortest** (shortest word).

Example: *[how, method, need, very] — which word contains `t`?* → `method`

---

### Multi-digit Addition

Simple 3-digit integer addition problems. Primarily probes whether a model's tokenizer correctly handles multi-digit numerals. MCQ distractors are generated using strategically chosen errors (off-by-one, digit swap, carry errors, ±10, ±100). Total of 1,000 samples.

Example: `295+592=` → `887`

---

### CUTE

Character Understanding Tasks Evaluation — 14 task types, 100 samples each, covering a broad range of character-level operations on Filipino and English words.

| Task type | Description | Example answer |
|---|---|---|
| `spell` | Spell out characters with spaces | Spell out `individual` → `i n d i v i d u a l` |
| `spell_inverse` | Reconstruct a word from spelled-out characters | Write the word `b a b y` → `baby` |
| `contains_char` | Does the word contain a given character? | Is there a `u` in `join`? → `No` |
| `contains_word` | Does the word contain a given substring? | Is there `in` in `He asked, with a twinkle in his eye.`? → `Yes` |
| `orth` | Select the word closer in Levenshtein distance | Closer to `career`: `life` or `care`? → `care` |
| `sem` | Select the word more semantically related | More related to `common`: `widespread` or `comment`? → `widespread` |
| `ins_char` | Insert a character after every instance of a given character | Add `t` after every `a` in `states` → `stattes` |
| `ins_word` | Insert a word after every instance of a given word | Add `among` after every `happy` in `She is happy.` → `She is happy among.` |
| `del_char` | Delete every instance of a given character | Delete every `e` in `reviews` → `rviws` |
| `del_word` | Delete every instance of a given word | Delete every `And` in `And they both enjoyed their soup.` → `they both enjoyed their soup.` |
| `sub_char` | Substitute every instance of one character with another | Substitute `a` with `x` in `agency` → `xgency` |
| `sub_word` | Substitute every instance of one word with another | Substitute `each` with `under` in `Tim and Tom look at each other.` → `Tim and Tom look at under other.` |
| `swap_char` | Swap the positions of two characters | Swap `a` and `e` in `added` → `eddad` |
| `swap_word` | Swap the positions of two words | Swap `loved` and `paint` in `Lily loved to paint with her bright colors.` → `Lily paint to loved with her bright colors.` |

## Setup

### 1. Configure your environment

```bash
cp .env.example .env
# Edit .env — set PROJECT_ROOT, VENV_PATH, LOGS_PATH, and any API keys
```

The `.env` file is gitignored; never commit it.

Key variables:

| Variable | Description |
|---|---|
| `PROJECT_ROOT` | Absolute path to this repo |
| `VENV_PATH` | Python virtual environment to activate |
| `LOGS_PATH` | Directory for PBS job logs and vLLM server logs (default: `$PROJECT_ROOT/logs`) |

### 2. Install the package

```bash
source .env
source "$VENV_PATH/bin/activate"
pip install -e ".[dev]"
```

### 3. Add your models

Edit `configs/models_pt.yaml` (base pretrained) or `configs/models_it.yaml` (instruction-tuned):

```yaml
models:
  my-model-7b-it:
    path: /path/to/my-model        # HuggingFace model path or local directory
    type: it                       # "pt" or "it"
    tokenizer: /path/to/tokenizer  # optional; defaults to path
    thinking: false                # set true for chain-of-thought models
```

## Generating benchmarks

Benchmark JSONL files are included in `data/benchmarks/`. To regenerate from the source corpora:

```bash
cd "$PROJECT_ROOT"
python -m pacute_bench.scripts.generate_benchmarks            # all benchmarks
python -m pacute_bench.scripts.generate_benchmarks --benchmarks pacute hierarchical
```

Options:

```
--benchmarks   pacute hierarchical langgame math cute all  (default: all)
--output-dir   data/benchmarks   (default)
--corpora-dir  data/corpora      (default)
--random-seed  1859              (default)
```

## Running evaluations

### Interactive / single model

First start a vLLM server, then run the evaluation script:

```bash
vllm serve /path/to/model --port 8000

python -m pacute_bench.scripts.run_evaluation \
    --models my-model-7b-it \
    --vllm-url http://localhost:8000
```

Useful flags:

```
--benchmarks   Override the default benchmark list
--eval-mode    auto | mcq | gen | both   (auto = MCQ-only for pt, both for it)
--max-samples  Cap samples per benchmark (useful for smoke-testing)
--overwrite    Re-run even when inference results already exist
--system-prompt  Override the per-benchmark system prompt
--output-dir   Where to write the combined results JSON (default: results/benchmark_evaluation)
```

### PBS cluster (recommended)

```bash
# Submit all models:
bash scripts/submit_evaluations.sh

# Subset options:
bash scripts/submit_evaluations.sh --it-only
bash scripts/submit_evaluations.sh --model my-model-7b-it
bash scripts/submit_evaluations.sh --dry-run          # preview qsub commands
bash scripts/submit_evaluations.sh --overwrite --max-samples 50
```

Each job:
1. Sources `.env` for `PROJECT_ROOT`, `VENV_PATH`, and `LOGS_PATH`
2. Starts a vLLM server (port auto-derived from PBS job ID)
3. Runs `pacute_bench.scripts.run_evaluation` against it
4. Writes PBS stdout/stderr and the vLLM server log to `$LOGS_PATH/`

### Commercial models (OpenAI, Anthropic, Gemini)

Commercial models can be evaluated via their respective batch APIs, which process all requests asynchronously and are more cost-efficient than online inference. Add models to `configs/models_commercial.yaml`:

```yaml
models:
  gpt-4o:
    path: gpt-4o
    type: it
    provider: openai      # openai | anthropic | gemini
```

Set the appropriate API key in `.env`, then run:

```bash
python -m pacute_bench.scripts.run_evaluation \
    --models gpt-4o \
    --eval-mode gen      # MCQ is not supported — batch APIs do not expose log-probabilities
```

> **Note:** Only the GEN format is supported for commercial models. MCQ evaluation requires per-token log-probabilities, which commercial APIs do not expose.

Providers currently supported: **OpenAI** (`OpenAIEvaluator`), **Anthropic** (`AnthropicEvaluator`), **Gemini** (`GeminiEvaluator`). To add a new provider, subclass `BatchEvaluator` in `src/pacute_bench/evaluators/` and implement `_submit_batch` and `_try_collect_batch`.

## Output structure

```
results/
└── <model-name>/
    ├── evaluation_results_<timestamp>.json   # metrics summary
    └── inference/
        ├── pacute-affixation-mcq.jsonl       # per-sample predictions
        ├── cute-gen.jsonl
        └── ...

logs/
├── pb-eval-<model>.o<jobid>                  # PBS stdout/stderr
└── vllm_<model>_<timestamp>.log             # vLLM server log
```

### Result format

MCQ result dict:
```json
{
  "accuracy": 0.72, "f1_score": 0.72, "normalized_accuracy": 0.63,
  "num_samples": 400, "format": "mcq",
  "by_category": { "prefix": { "accuracy": 0.80, ... }, ... }
}
```

Generative result dict:
```json
{
  "exact_match": 0.45, "contains_match": 0.61, "prefix_match": 0.52,
  "num_samples": 400, "format": "generative",
  "by_category": { ... }
}
```

## Project structure

```
pacute-bench/
├── .env                          # local config (gitignored)
├── .env.example                  # template — commit this
├── configs/
│   ├── evaluation.yaml           # per-benchmark system prompts & answer tags
│   ├── models_it.yaml            # instruction-tuned model registry
│   ├── models_pt.yaml            # pretrained model registry
│   └── models_commercial.yaml    # commercial model registry (OpenAI, Anthropic, Gemini)
├── data/
│   ├── benchmarks/               # generated JSONL evaluation files
│   └── corpora/                  # source data for generation
├── scripts/
│   ├── eval_model.pbs            # PBS job script (single model)
│   └── submit_evaluations.sh     # batch PBS submission
├── src/pacute_bench/
│   ├── evaluators/
│   │   ├── base.py               # BaseEvaluator (shared logic)
│   │   ├── vllm.py               # VLLMEvaluator (self-hosted models)
│   │   ├── batch.py              # BatchEvaluator (abstract base for commercial)
│   │   ├── openai.py             # OpenAIEvaluator
│   │   ├── anthropic.py          # AnthropicEvaluator
│   │   └── gemini.py             # GeminiEvaluator
│   ├── generators/               # benchmark dataset generators
│   ├── loaders/                  # benchmark loaders (registry)
│   ├── scripts/
│   │   ├── generate_benchmarks.py   # pacute-generate entry-point
│   │   └── run_evaluation.py        # pacute-eval entry-point
│   └── utils/                    # shared helpers (strings, syllabification …)
└── tests/
```

## Contributors

- Railey Montalan
- David Africa
- Richell Flores
- JP Layacan
- Lance Gamboa
