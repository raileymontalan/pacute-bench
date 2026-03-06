# pacute-bench

A self-contained evaluation suite for Filipino morphology benchmarks, designed to run against a [vLLM](https://github.com/vllm-project/vllm) server via its OpenAI-compatible API.

## Benchmarks

| Name | Format | Task |
|---|---|---|
| **PACUTE-Affixation** | MCQ + GEN | Identify correct prefix/suffix/infix inflections |
| **PACUTE-Composition** | MCQ + GEN | Combine morphemes to form well-formed words |
| **PACUTE-Manipulation** | MCQ + GEN | String operations (reversal, deletion, substitution) on Filipino words |
| **PACUTE-Syllabification** | MCQ + GEN | Break words into correct syllable boundaries |
| **Hierarchical** | MCQ + GEN | 6-level diagnostic cascade (character → morpheme → composition) |
| **LangGame** | MCQ + GEN | Word-level reasoning (longest, contains letter, starts/ends with …) |
| **Multi-digit Addition** | MCQ + GEN | 3-digit arithmetic (tests numeral tokenization) |
| **CUTE** | GEN | Character-level understanding (spell, insert, delete, swap, sub …) |

MCQ benchmarks are evaluated using log-probability scoring (no generation needed). Generative benchmarks use exact/contains/prefix match against the ground truth.

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

Edit `configs/models_pt.yaml` (pretrained) or `configs/models_it.yaml` (instruction-tuned):

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
│   └── models_pt.yaml            # pretrained model registry
├── data/
│   ├── benchmarks/               # generated JSONL evaluation files
│   └── corpora/                  # source data for generation
├── scripts/
│   ├── eval_model.pbs            # PBS job script (single model)
│   └── submit_evaluations.sh     # batch PBS submission
├── src/pacute_bench/
│   ├── evaluator.py              # VLLMEvaluator class
│   ├── generators/               # benchmark dataset generators
│   ├── loaders/                  # benchmark loaders (registry)
│   ├── scripts/
│   │   ├── generate_benchmarks.py   # pacute-generate entry-point
│   │   └── run_evaluation.py        # pacute-eval entry-point
│   └── utils/                    # shared helpers (strings, syllabification …)
└── tests/
```
