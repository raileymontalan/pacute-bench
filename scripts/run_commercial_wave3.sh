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

# Wave 3: GPT-5 family (need max_completion_tokens, batch may be slow)
MODELS=(
    gpt-5
    gpt-5-mini
    gpt-5.4
)

echo "=============================================="
echo "PACUTE Commercial Evaluation — Wave 3"
echo "Models: ${MODELS[*]}"
echo "Started: $(date)"
echo "=============================================="

for model in "${MODELS[@]}"; do
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
echo "WAVE 3 DONE: $(date)"
echo "=============================================="
