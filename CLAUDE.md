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

Python ≥3.11 required.

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
bash scripts/submit_evaluations.sh --queue <your-queue>
bash scripts/submit_evaluations.sh --queue <your-queue> --it-only
bash scripts/submit_evaluations.sh --queue <your-queue> --model my-model-7b-it --dry-run
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
    sample_human_baseline.py     # stratified sample for human annotation
    generate_annotation_sheets.py  # Excel workbooks for annotators
    score_human_baselines.py     # scoring + IAA for filled workbooks
    rescore_from_inference.py    # re-aggregate results from inference JSONL
  utils/                    # Shared helpers (strings, syllabification, sampling)

configs/
  evaluation.yaml           # Per-benchmark system prompts and answer tags
  models_it.yaml            # Instruction-tuned model registry
  models_pt.yaml            # Pretrained model registry

data/
  benchmarks/               # Generated JSONL files (checked in)
  corpora/                  # Source data for generation

scripts/
  submit_evaluations.sh     # Batch submission (SLURM + PBS, auto-detected)
  eval_model.slurm          # SLURM job script (single vLLM model)
  eval_model.pbs            # PBS equivalent
  eval_commercial.slurm     # SLURM job script (commercial API models)
  eval_commercial.pbs       # PBS equivalent
  setup_env.slurm           # One-time venv setup on a GPU node
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
  evaluation_results.json               # metrics summary
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
