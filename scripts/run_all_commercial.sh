#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/pacute-bench
source env/bin/activate

BENCHMARKS=(
    pacute-affixation-gen
    pacute-composition-gen
    pacute-manipulation-gen
    pacute-syllabification-gen
    hierarchical-gen
    langgame-gen
    multi-digit-addition-gen
    cute-gen
)
BENCH_ARGS="${BENCHMARKS[*]}"

# Anthropic models (async via AISI proxy)
ANTHROPIC_MODELS=(claude-opus-4-6 claude-sonnet-4-6 claude-haiku-4-5)

# OpenAI models (batch API via AISI proxy)
OPENAI_MODELS=(gpt-4o gpt-4.1 gpt-4.1-mini gpt-4.1-nano o3 o4-mini)

ALL_MODELS=("${ANTHROPIC_MODELS[@]}" "${OPENAI_MODELS[@]}")

echo "=============================================="
echo "PACUTE Commercial Model Evaluation"
echo "Models: ${ALL_MODELS[*]}"
echo "Benchmarks: ${BENCH_ARGS}"
echo "Started: $(date)"
echo "=============================================="

for model in "${ALL_MODELS[@]}"; do
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ">>> Starting: $model  ($(date))"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    python -m pacute_bench.scripts.run_evaluation \
        --models "$model" \
        --benchmarks $BENCH_ARGS \
        --eval-mode gen \
        2>&1 | tee -a "results/${model}_run.log"
    echo ">>> Finished: $model  ($(date))"
done

echo ""
echo "=============================================="
echo "ALL DONE: $(date)"
echo "=============================================="
