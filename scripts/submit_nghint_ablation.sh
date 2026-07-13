#!/bin/bash
# ng-hint syllabification ablation — submit jobs or show results.
# Runs pacute-syllabification-gen-nghint for 4 models that show ng-digraph
# overcounting errors at varying syllabification performance levels.
# Standard pacute-syllabification-gen results already exist for comparison.
#
# Usage:
#   bash scripts/submit_nghint_ablation.sh [QUEUE]          # submit jobs
#   bash scripts/submit_nghint_ablation.sh --analyze        # print results table

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
source "$PROJECT_ROOT/.env"

ANALYZE=false
QUEUE="AISG_debug"
for arg in "$@"; do
    case "$arg" in
        --analyze) ANALYZE=true ;;
        *)         QUEUE="$arg" ;;
    esac
done

BENCHMARKS="pacute-syllabification-gen-nghint"

# Rationale for model selection (existing syll-gen contains_match in parens):
#  gemma-3-27b-it            – 30.0% — explicitly named in paper's ng-overcounting error analysis
#  sea-lion-qwen-v4.5-27b-it – 36.5% — SEA-trained, mid-range
#  qwen3.6-27b-it            – 47.0% — strong multilingual, upper-mid range
#  gemma-4-31b-it            – 50.5% — best open-weight IT, to test hint impact at ceiling
MODELS=(
    gemma-3-27b-it
    sea-lion-qwen-v4.5-27b-it
    qwen3.6-27b-it
    gemma-4-31b-it
)

if $ANALYZE; then
    cd "$PROJECT_ROOT"
    python3 -m pacute_bench.scripts.analyze_ablation \
        --nghint \
        --results-dir "$RESULTS_PATH" \
        --models "${MODELS[@]}"
else
    mkdir -p "$LOGS_PATH"
    for MODEL in "${MODELS[@]}"; do
        JOB_NAME="pb-nghint-$(echo "$MODEL" | tr '.' '-')"
        VARS="MODEL_NAME=${MODEL},PROJECT_ROOT=${PROJECT_ROOT},BENCHMARKS='${BENCHMARKS}'"
        echo -n "  Submitting $MODEL … "
        qsub \
            -N "$JOB_NAME" \
            -q "$QUEUE" \
            -l "select=1:mem=64gb:ncpus=4:ngpus=1" \
            -l "walltime=12:00:00" \
            -o "$LOGS_PATH/" -e "$LOGS_PATH/" -j oe \
            -v "$VARS" \
            "$SCRIPT_DIR/eval_model.pbs"
    done
fi
