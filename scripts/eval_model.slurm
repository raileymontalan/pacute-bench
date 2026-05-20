#!/bin/bash
# SLURM job script — evaluate a single model on all benchmarks.
# Do NOT run directly; submit via scripts/submit_evaluations_slurm.sh.
#
# Required env vars (set before sbatch, or via --export):
#   MODEL_NAME   Short model key from configs/models_*.yaml  (e.g. qwen-2.5-7b-it)
#   PROJECT_ROOT Absolute path to pacute-bench repo root
#
# Optional env vars:
#   OUTPUT_DIR   Where to write evaluation results (default: $RESULTS_PATH from .env, else $LOGS_PATH)
#   OVERWRITE    "true" to re-run benchmarks that already have results
#   MAX_SAMPLES  Integer to cap samples per benchmark (default: all)
#   BENCHMARKS   Comma-separated benchmark names (default: all)
#   VLLM_PORT    Port for the vLLM server (default: auto-derived from job ID)

#SBATCH --job-name=pb-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/pb-eval_%j.out
#SBATCH --error=logs/pb-eval_%j.out
# Note: --output/--error/--gres are overridden dynamically by submit_evaluations_slurm.sh


# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-gpt2}"
OVERWRITE="${OVERWRITE:-false}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BENCHMARKS="${BENCHMARKS:-}"
# OUTPUT_DIR resolved after .env is sourced (RESULTS_PATH may be set there)

# Auto-derive a unique port from SLURM_JOB_ID so concurrent jobs don't collide.
if [[ -z "${VLLM_PORT:-}" ]]; then
    VLLM_PORT=$(( 8000 + (SLURM_JOB_ID % 1000) ))
fi

set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT must be set}"
cd "$PROJECT_ROOT"

# Source .env for VENV_PATH, LOGS_PATH, RESULTS_PATH, and API keys
source "$PROJECT_ROOT/.env"
mkdir -p "$LOGS_PATH"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_PATH:-$LOGS_PATH}}"
mkdir -p "$OUTPUT_DIR"

# Load CUDA modules if MODULEFILES_PATH is defined in .env
if [[ -n "${MODULEFILES_PATH:-}" ]]; then
    module use "$MODULEFILES_PATH"
    module swap cuda/13.0.3 2>/dev/null || true
fi

source "$VENV_PATH/bin/activate"

# Set LD_LIBRARY_PATH for CUDA libs bundled in the venv
SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cusparselt/lib:${SITE_PACKAGES}/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export VLLM_USE_DEEP_GEMM=0

uv pip install -e "$PROJECT_ROOT" --quiet

# ── UUID → integer GPU remapping ──────────────────────────────────────────────
UUIDS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n')
declare -a VISIBLE_DEVICES=()
for UUID in $UUIDS; do
    ID=$(nvidia-smi --id="$UUID" --query-gpu=index --format=csv,noheader 2>/dev/null || echo "$UUID")
    VISIBLE_DEVICES+=("$ID")
done
VISIBLE_DEVICES_STR=$(IFS=,; echo "${VISIBLE_DEVICES[*]}")
export CUDA_VISIBLE_DEVICES="$VISIBLE_DEVICES_STR"
N_GPUS=${#VISIBLE_DEVICES[@]}
echo "CUDA_VISIBLE_DEVICES → $CUDA_VISIBLE_DEVICES  ($N_GPUS GPUs)"

# ── Resolve all model fields from YAML ────────────────────────────────────────
_model_info=$(python3 - <<PYEOF
import yaml, sys, shlex
for cfg in ['configs/models_pt.yaml', 'configs/models_it.yaml']:
    try:
        data = yaml.safe_load(open(cfg))
        info = data['models'].get('${MODEL_NAME}')
        if info:
            print('MODEL_PATH=' + shlex.quote(str(info['path'])))
            print('MODEL_TYPE=' + shlex.quote(str(info['type'])))
            print('TP=' + shlex.quote(str(info.get('tp', 1))))
            print('REASONING_PARSER=' + shlex.quote(str(info.get('reasoning_parser', ''))))
            print('ENABLE_EXPERT_PARALLEL=' + str(info.get('enable_expert_parallel', False)).lower())
            print('LANGUAGE_MODEL_ONLY=' + str(info.get('language_model_only', False)).lower())
            print('REASONING_PARSER_PLUGIN=' + shlex.quote(str(info.get('reasoning_parser_plugin', ''))))
            print('CHAT_TEMPLATE_KWARGS=' + shlex.quote(str(info.get('chat_template_kwargs', ''))))
            print('MAX_MODEL_LEN=' + shlex.quote(str(info.get('max_model_len', 8192))))
            sys.exit(0)
    except Exception:
        pass
print('MODEL_PATH=UNKNOWN')
sys.exit(1)
PYEOF
)
eval "$_model_info"

if [[ "${MODEL_PATH}" == "UNKNOWN" ]]; then
    echo "ERROR: model '${MODEL_NAME}' not found in any config YAML" >&2
    exit 1
fi
echo "Model path              : $MODEL_PATH"
echo "Model type              : $MODEL_TYPE"
echo "TP size (YAML)          : $TP"
echo "N GPUs (allocated)      : $N_GPUS"
echo "Reasoning parser        : ${REASONING_PARSER:-(none)}"
echo "Expert parallel         : $ENABLE_EXPERT_PARALLEL"
echo "Language model only     : $LANGUAGE_MODEL_ONLY"
echo "Reasoning parser plugin : ${REASONING_PARSER_PLUGIN:-(none)}"
echo "Chat template kwargs    : ${CHAT_TEMPLATE_KWARGS:-(none)}"
echo "Max model len           : $MAX_MODEL_LEN"

# ── Start vLLM server ─────────────────────────────────────────────────────────
VLLM_LOG="${LOGS_PATH}/vllm_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$VLLM_LOG")"

export VLLM_ENGINE_CORE_CLIENT_TIMEOUT=3600

VLLM_SERVE_ARGS=(
    "$MODEL_PATH"
    --port "$VLLM_PORT"
    --tensor-parallel-size "$N_GPUS"
    --max-model-len "$MAX_MODEL_LEN"
    --enable-prefix-caching
    --gpu-memory-utilization 0.85
    --trust-remote-code
    --attention-backend FLASH_ATTN
)
[[ -n "${REASONING_PARSER:-}" ]]        && VLLM_SERVE_ARGS+=(--reasoning-parser "$REASONING_PARSER")
[[ -n "${REASONING_PARSER_PLUGIN:-}" ]] && VLLM_SERVE_ARGS+=(--reasoning-parser-plugin "$REASONING_PARSER_PLUGIN")
[[ -n "${CHAT_TEMPLATE_KWARGS:-}" ]]    && VLLM_SERVE_ARGS+=(--default-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
[[ "${ENABLE_EXPERT_PARALLEL:-}" == "true" ]] && VLLM_SERVE_ARGS+=(--enable-expert-parallel)
[[ "${LANGUAGE_MODEL_ONLY:-}" == "true" ]]   && VLLM_SERVE_ARGS+=(--language-model-only)

echo "Starting vLLM: $MODEL_PATH (TP=$N_GPUS, port=$VLLM_PORT, max_model_len=$MAX_MODEL_LEN)"
vllm serve "${VLLM_SERVE_ARGS[@]}" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "Stopping vLLM server (PID $VLLM_PID)…"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ── Wait for /health endpoint ─────────────────────────────────────────────────
VLLM_URL="http://localhost:${VLLM_PORT}"
echo -n "Waiting for vLLM server"
ELAPSED=0; TIMEOUT=600
until curl -sf "${VLLM_URL}/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo " DIED — see $VLLM_LOG" >&2; exit 1
    fi
    sleep 5; ELAPSED=$((ELAPSED + 5)); echo -n "."
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo " TIMEOUT — see $VLLM_LOG" >&2; exit 1
    fi
done
echo " ready (${ELAPSED}s)"

# ── Detect the model ID actually registered by vLLM ──────────────────────────
REGISTERED_MODEL=$(curl -sf "${VLLM_URL}/v1/models" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
echo "Registered model ID: $REGISTERED_MODEL"

# ── Build optional flags ──────────────────────────────────────────────────────
EXTRA_FLAGS=""
[[ "$OVERWRITE"   == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --overwrite"
[[ -n "$MAX_SAMPLES" ]]        && EXTRA_FLAGS="$EXTRA_FLAGS --max-samples $MAX_SAMPLES"
[[ -n "$BENCHMARKS" ]]         && EXTRA_FLAGS="$EXTRA_FLAGS --benchmarks ${BENCHMARKS//,/ }"

# ── Run evaluation ────────────────────────────────────────────────────────────
echo "Running evaluation for $MODEL_NAME…"
python3 -m pacute_bench.scripts.run_evaluation \
    --models "$MODEL_NAME" \
    --model-config configs/models_pt.yaml configs/models_it.yaml \
    --vllm-url "$VLLM_URL" \
    --vllm-model-id "$REGISTERED_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_FLAGS

printf "\n=== Job Runtime: %dh %dm %ds ===\n" $((SECONDS/3600)) $(((SECONDS%3600)/60)) $((SECONDS%60))
echo "Done: $MODEL_NAME"
