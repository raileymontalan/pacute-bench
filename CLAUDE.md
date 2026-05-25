# pacute-bench — CLAUDE.md

Filipino morphology evaluation suite. Runs benchmarks against a vLLM server (OpenAI-compatible API). MCQ benchmarks scored via log-probability; generative benchmarks via exact/contains/prefix match.

---

## Setup

```bash
cp .env.example .env       # set PROJECT_ROOT, VENV_PATH, LOGS_PATH
source .env
source "$VENV_PATH/bin/activate"
pip install -e ".[dev]"
```

Python ≥3.9 required.

---

## Entry Points

| Command | What it does |
|---|---|
| `pacute-generate` | Regenerate benchmark JSONL files from source corpora |
| `pacute-eval` | Run evaluation against a vLLM server |

```bash
# Generate benchmarks
python -m pacute_bench.scripts.generate_benchmarks
python -m pacute_bench.scripts.generate_benchmarks --benchmarks pacute hierarchical

# Run evaluation (interactive)
vllm serve /path/to/model --port 8000
python -m pacute_bench.scripts.run_evaluation \
    --models my-model-7b-it \
    --vllm-url http://localhost:8000

# Run evaluation (PBS cluster)
bash scripts/submit_evaluations.sh
bash scripts/submit_evaluations.sh --it-only
bash scripts/submit_evaluations.sh --model my-model-7b-it --dry-run
```

Key eval flags: `--benchmarks`, `--eval-mode auto|mcq|gen|both`, `--max-samples`, `--overwrite`, `--output-dir`.

---

## Benchmarks

### PACUTE (5 categories)

| Benchmark | Format | Samples | Task |
|---|---|---|---|
| `pacute-composition` | MCQ + GEN | 950 / 550 | Character-level composition (spelling, counting, finding) |
| `pacute-manipulation` | MCQ + GEN | 800 / 800 | String operations on Filipino words |
| `pacute-syllabification` | MCQ + GEN | 200 / 200 | Stress identification and disambiguation |
| `pacute-morphological-extraction` | MCQ + GEN | 400 / 400 | Identify affix, root, or reduplicant of a word |
| `pacute-morphological-production` | MCQ + GEN | 150 / 150 | Produce inflected form from root + affix |

### Other benchmarks

| Benchmark | Format | Task |
|---|---|---|
| `hierarchical` | MCQ + GEN | 6-level character→morpheme→composition cascade |
| `langgame` | MCQ + GEN | Word-property reasoning (length, letters, order) |
| `multi-digit-addition` | MCQ + GEN | 3-digit arithmetic (tests numeral tokenization) |
| `cute` | GEN | Character-level manipulation (spell, insert, delete, swap) |

### Legacy (not part of PACUTE group)

| Benchmark | Format | Task |
|---|---|---|
| `pacute-affixation` | MCQ + GEN | Filipino prefix/suffix/infix inflection (legacy) |

Benchmark JSONL files live in `data/benchmarks/`. Source corpora in `data/corpora/`.

---

## Project Structure

```
src/pacute_bench/
  evaluator.py              # VLLMEvaluator — inference + scoring
  generators/               # Benchmark JSONL generators (one per benchmark family)
  loaders/                  # Benchmark loaders + registry
  scripts/
    generate_benchmarks.py  # pacute-generate entry point
    run_evaluation.py       # pacute-eval entry point
  utils/                    # Shared helpers (strings, syllabification, sampling)

configs/
  evaluation.yaml           # Per-benchmark system prompts and answer tags
  models_it.yaml            # Instruction-tuned model registry
  models_pt.yaml            # Pretrained model registry

data/
  benchmarks/               # Generated JSONL files (checked in)
  corpora/                  # Source data for generation

scripts/
  eval_model.pbs            # PBS job script (single model)
  submit_evaluations.sh     # Batch PBS submission
```

---

## Adding a Model

Edit `configs/models_it.yaml` (instruction-tuned) or `configs/models_pt.yaml` (pretrained):

```yaml
models:
  my-model-7b-it:
    path: /path/to/model
    type: it            # "pt" or "it"
    tokenizer: /path/to/tokenizer   # optional; defaults to path
    thinking: false     # true for chain-of-thought models
```

---

## Output Structure

```
results/<model-name>/
  evaluation_results_<timestamp>.json   # metrics summary
  inference/
    pacute-composition-mcq.jsonl        # per-sample predictions
    pacute-morphological-extraction-gen.jsonl
    cute-gen.jsonl
    ...

logs/
  pb-eval-<model>.o<jobid>              # PBS stdout/stderr
  vllm_<model>_<timestamp>.log          # vLLM server log
```

MCQ result keys: `accuracy`, `f1_score`, `normalized_accuracy`, `num_samples`, `by_category`.
GEN result keys: `exact_match`, `contains_match`, `prefix_match`, `num_samples`, `by_category`.

---

## Code Style

No formatter config checked in — match the style of the file being edited.

Run tests:
```bash
pytest
```

---

## Karpathy's Guidelines

### 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```
