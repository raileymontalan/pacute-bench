#!/bin/bash
# SLURM job script — evaluate a single commercial model (OpenAI/Anthropic/Gemini/xAI) via batch API.
# No GPU required. Submits all requests as a batch, then polls until complete.
# Do NOT run directly; submit via scripts/submit_evaluations_slurm.sh --commercial-only.
#
# Required env vars:
#   MODEL_NAME   Key from configs/models_commercial.yaml  (e.g. claude-opus-4-6)
#   PROJECT_ROOT Absolute path to pacute-bench repo root
#
# Optional env vars:
#   OUTPUT_DIR   Where to write evaluation results (default: $RESULTS_PATH from .env, else $LOGS_PATH)
#   OVERWRITE    "true" to re-run benchmarks that already have results
#   MAX_SAMPLES  Integer to cap samples per benchmark (default: all)
#   BENCHMARKS   Comma-separated benchmark names (default: all 9 gen benchmarks below)

#SBATCH --job-name=pb-commercial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=26:00:00
#SBATCH --output=logs/pb-commercial_%j.out
#SBATCH --error=logs/pb-commercial_%j.out
# Note: --output/--error are overridden dynamically by submit_evaluations_slurm.sh

MODEL_NAME="${MODEL_NAME:-claude-sonnet-4-6}"
OVERWRITE="${OVERWRITE:-false}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
# Default: all generative benchmarks. Commercial APIs do not expose log-probabilities,
# so MCQ evaluation is not supported — always run gen-only.
BENCHMARKS="${BENCHMARKS:-pacute-composition-gen,pacute-manipulation-gen,pacute-syllabification-gen,pacute-morphological-extraction-gen,pacute-morphological-production-gen,hierarchical-gen,langgame-gen,multi-digit-addition-gen,cute-gen}"

set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT must be set}"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.env"
mkdir -p "$LOGS_PATH"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_PATH:-$LOGS_PATH}}"
mkdir -p "$OUTPUT_DIR"

source "$VENV_PATH/bin/activate"
uv pip install -e "$PROJECT_ROOT" --quiet

echo "Running commercial evaluation for $MODEL_NAME…"

EXTRA_FLAGS=""
[[ "$OVERWRITE"  == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --overwrite"
[[ -n "$MAX_SAMPLES" ]]       && EXTRA_FLAGS="$EXTRA_FLAGS --max-samples $MAX_SAMPLES"
[[ -n "$BENCHMARKS" ]]        && EXTRA_FLAGS="$EXTRA_FLAGS --benchmarks ${BENCHMARKS//,/ }"

python3 -m pacute_bench.scripts.run_evaluation \
    --models "$MODEL_NAME" \
    --model-config configs/models_commercial.yaml \
    --eval-mode gen \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_FLAGS

printf "\n=== Job Runtime: %dh %dm %ds ===\n" $((SECONDS/3600)) $(((SECONDS%3600)/60)) $((SECONDS%60))
echo "Done: $MODEL_NAME"
