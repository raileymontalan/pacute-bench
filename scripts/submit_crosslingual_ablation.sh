#!/bin/bash
# Cross-lingual (EN vs TL item-text) ablation — submit jobs or show results.
# Runs the 12 TL benchmark variants for 5 representative models.
# EN results already exist; only TL needs to be collected.
#
# Usage:
#   bash scripts/submit_crosslingual_ablation.sh [QUEUE]          # submit jobs
#   bash scripts/submit_crosslingual_ablation.sh --analyze        # print results table

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

BENCHMARKS="hierarchical-mcq-tl,hierarchical-gen-tl,pacute-composition-mcq-tl,pacute-composition-gen-tl,pacute-manipulation-mcq-tl,pacute-manipulation-gen-tl,pacute-morphological-extraction-mcq-tl,pacute-morphological-extraction-gen-tl,pacute-morphological-production-mcq-tl,pacute-morphological-production-gen-tl,pacute-syllabification-mcq-tl,pacute-syllabification-gen-tl"

# Rationale for model selection:
#  gpt2                      – PT, no multilingual training → sanity check (expect ~0 delta)
#  gemma-3-4b-it             – small IT, multilingual Gemma family
#  qwen3.6-27b-it            – large IT, Qwen with strong multilingual coverage
#  sea-lion-qwen-v4.5-27b-it – SEA-trained IT, strongest hypothesis for TL > EN
#  gemma-4-31b-it            – large IT, Gemma (best open-weight syll performer)
MODELS=(
    gpt2
    gemma-3-4b-it
    qwen3.6-27b-it
    sea-lion-qwen-v4.5-27b-it
    gemma-4-31b-it
)

if $ANALYZE; then
    cd "$PROJECT_ROOT"
    python3 -m pacute_bench.scripts.analyze_ablation \
        --crosslingual \
        --results-dir "$RESULTS_PATH" \
        --models "${MODELS[@]}"
else
    mkdir -p "$LOGS_PATH"
    for MODEL in "${MODELS[@]}"; do
        JOB_NAME="pb-cxl-$(echo "$MODEL" | tr '.' '-')"
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
