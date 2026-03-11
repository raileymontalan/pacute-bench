#!/bin/bash
# Submit one PBS job per model defined in the model config YAMLs.
#
# Usage:
#   bash scripts/submit_evaluations.sh [OPTIONS]
#
# Options:
#   --pt-only              Only submit PT (pretrained) models
#   --it-only              Only submit IT (instruction-tuned) models
#   --commercial-only      Only submit commercial API models (OpenAI/Anthropic, no GPU)
#   --model <name>         Submit only this model (repeatable)
#   --overwrite            Re-run benchmarks that already have results
#   --max-samples <n>      Cap samples per benchmark
#   --benchmarks <b...>    Only run these benchmarks (e.g. pacute-syllabification-gen)
#   --port <n>             vLLM server port (default: auto)
#   --queue <q>            PBS queue (default: AISG_debug)
#   --walltime <hh:mm:ss>  Job walltime (default: 12:00:00)
#   --filter <pattern>    Only submit models whose name contains <pattern>
#   --dry-run              Print qsub commands without submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="$SCRIPT_DIR/eval_model.pbs"

# ── Load environment from .env ────────────────────────────────────────────────
# PROJECT_ROOT, VENV_PATH, LOGS_PATH are all defined there.
_ENV_FILE="$(dirname "$SCRIPT_DIR")/.env"
if [[ ! -f "$_ENV_FILE" ]]; then
    echo "ERROR: .env not found at $_ENV_FILE" >&2
    echo "Copy .env.example to .env and fill in your paths." >&2
    exit 1
fi
# shellcheck source=../.env
source "$_ENV_FILE"

PT_ONLY=false
IT_ONLY=false
COMMERCIAL_ONLY=false
DRY_RUN=false
OVERWRITE=false
MAX_SAMPLES=""
BENCHMARKS=""
VLLM_PORT=""
PBS_QUEUE="AISG_debug"
WALLTIME="12:00:00"
SINGLE_MODELS=()
FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pt-only)     PT_ONLY=true ;;
        --it-only)     IT_ONLY=true ;;
        --commercial-only) COMMERCIAL_ONLY=true ;;
        --dry-run)     DRY_RUN=true ;;
        --overwrite)   OVERWRITE=true ;;
        --port)        VLLM_PORT="$2"; shift ;;
        --max-samples) MAX_SAMPLES="$2"; shift ;;
        --benchmarks)  shift
                       while [[ $# -gt 0 && "$1" != --* ]]; do
                           BENCHMARKS="${BENCHMARKS:+${BENCHMARKS},}$1"
                           shift
                       done
                       continue ;;
        --queue)       PBS_QUEUE="$2"; shift ;;
        --walltime)    WALLTIME="$2"; shift ;;
        --model)       SINGLE_MODELS+=("$2"); shift ;;
        --filter)      FILTER="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ── Config files to read ──────────────────────────────────────────────────────
CONFIG_FILES=()
if $COMMERCIAL_ONLY; then
    CONFIG_FILES=("$PROJECT_ROOT/configs/models_commercial.yaml")
elif $IT_ONLY; then
    CONFIG_FILES=("$PROJECT_ROOT/configs/models_it.yaml")
elif $PT_ONLY; then
    CONFIG_FILES=("$PROJECT_ROOT/configs/models_pt.yaml")
else
    CONFIG_FILES=(
        "$PROJECT_ROOT/configs/models_pt.yaml"
        "$PROJECT_ROOT/configs/models_it.yaml"
    )
fi

# ── Build model list ──────────────────────────────────────────────────────────
if [[ ${#SINGLE_MODELS[@]} -gt 0 ]]; then
    ALL_MODELS=("${SINGLE_MODELS[@]}")
else
    ALL_MODELS=()
    for cfg in "${CONFIG_FILES[@]}"; do
        [[ ! -f "$cfg" ]] && continue
        while IFS= read -r model; do
            ALL_MODELS+=("$model")
        done < <(python3 -c "
import yaml
with open('$cfg') as f:
    data = yaml.safe_load(f)
for name in data.get('models', {}):
    print(name)
")
    done
fi

# ── GPU heuristic ─────────────────────────────────────────────────────────────
# Extracts the largest "NNb" or "NNN" token from the model path/name as a
# parameter count (in billions), then maps it to a GPU count via ranges:
#   >= 120B → 8 GPUs  |  >= 27B → 4 GPUs  |  >= 7B → 2 GPUs  |  else → 1 GPU
# Note: models with an explicit "NNm" (millions) suffix are treated as <1B and
# skip the integer fallback to avoid misreading e.g. "270m" as 270B.
gpus_for_model() {
    local model_id="$1"
    local size_b
    # Extract the largest numeric value followed by b/B (e.g. 70b, 235B, 3.5b)
    size_b=$(echo "$model_id" | grep -oiE '[0-9]+(\.[0-9]+)?b' \
               | sed 's/[bB]$//' \
               | awk 'BEGIN{m=0} {if($1+0>m) m=$1+0} END{print m}')
    if [[ -z "$size_b" || "$size_b" == "0" ]]; then
        # If path has an explicit Nm/NM suffix (millions), skip fallback — model is <1B
        local has_m
        has_m=$(echo "$model_id" | grep -oiE '[0-9]+(\.[0-9]+)?m' | head -1)
        if [[ -z "$has_m" ]]; then
            # Fallback: plain integers in the path (e.g. deepseek-v3 has no explicit Nb)
            size_b=$(echo "$model_id" | grep -oE '[0-9]+' \
                       | awk 'BEGIN{m=0} {if($1+0>m) m=$1+0} END{print m}')
        else
            size_b=0
        fi
    fi
    size_b=${size_b:-0}
    if   awk "BEGIN{exit !($size_b >= 120)}"; then echo 8
    elif awk "BEGIN{exit !($size_b >= 27)}";  then echo 4
    elif awk "BEGIN{exit !($size_b >= 7)}";   then echo 2
    else echo 1
    fi
}

get_model_path() {
    local name="$1"
    python3 - <<PYEOF
import yaml, sys
for cfg in ['$PROJECT_ROOT/configs/models_pt.yaml', '$PROJECT_ROOT/configs/models_it.yaml', '$PROJECT_ROOT/configs/models_commercial.yaml']:
    try:
        data = yaml.safe_load(open(cfg))
        info = data['models'].get('$name')
        if info:
            print(info['path'])
            sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PYEOF
}

get_model_min_gpus() {
    local name="$1"
    python3 - <<PYEOF
import yaml, sys
for cfg in ['$PROJECT_ROOT/configs/models_pt.yaml', '$PROJECT_ROOT/configs/models_it.yaml', '$PROJECT_ROOT/configs/models_commercial.yaml']:
    try:
        data = yaml.safe_load(open(cfg))
        info = data['models'].get('$name')
        if info and 'min_gpus' in info:
            print(info['min_gpus'])
            sys.exit(0)
    except Exception:
        pass
print(0)
PYEOF
}

# ── PBS variable string ───────────────────────────────────────────────────────
pbs_vars() {
    local model="$1"
    local vars="MODEL_NAME=${model},PROJECT_ROOT=${PROJECT_ROOT}"
    [[ -n "$VLLM_PORT" ]]   && vars="${vars},VLLM_PORT=${VLLM_PORT}"
    $OVERWRITE              && vars="${vars},OVERWRITE=true"
    [[ -n "$MAX_SAMPLES" ]] && vars="${vars},MAX_SAMPLES=${MAX_SAMPLES}"
    [[ -n "$BENCHMARKS" ]]  && vars="${vars},BENCHMARKS=${BENCHMARKS}"
    echo "$vars"
}

# ── Apply --filter ───────────────────────────────────────────────────────────
if [[ -n "$FILTER" ]]; then
    FILTERED=()
    for m in "${ALL_MODELS[@]}"; do
        [[ "$m" == *"$FILTER"* ]] && FILTERED+=("$m")
    done
    ALL_MODELS=("${FILTERED[@]}")
fi

# ── Create log directory ──────────────────────────────────────────────────────
mkdir -p "$LOGS_PATH"

# ── Submit ────────────────────────────────────────────────────────────────────
echo "Submitting ${#ALL_MODELS[@]} model evaluation job(s) to queue: $PBS_QUEUE"
echo "PBS script: $PBS_SCRIPT"
echo ""

SUBMITTED=0
SKIPPED=0

for model_name in "${ALL_MODELS[@]}"; do
    model_path=$(get_model_path "$model_name" 2>/dev/null) || {
        echo "  SKIP: '$model_name' not found in config YAMLs"
        SKIPPED=$((SKIPPED + 1))
        continue
    }

    if $COMMERCIAL_ONLY; then
        # Commercial models: no GPU needed, poll-based batch job
        VARS=$(pbs_vars "$model_name")
        JOB_NAME="pb-commercial-$(echo "$model_name" | tr '.' '-')"
        CMD=(
            qsub
            -N "$JOB_NAME"
            -q "$PBS_QUEUE"
            -l "select=1:mem=8gb:ncpus=2"
            -l "walltime=${WALLTIME}"
            -o "${LOGS_PATH}/"
            -e "${LOGS_PATH}/"
            -j oe
            -v "$VARS"
            "$SCRIPT_DIR/eval_commercial.pbs"
        )
        printf "  %-32s  (commercial API)  " "$model_name"
    else
        N_GPUS=$(gpus_for_model "$model_path")
        MIN_GPUS=$(get_model_min_gpus "$model_name")
        [[ "$MIN_GPUS" -gt "$N_GPUS" ]] && N_GPUS="$MIN_GPUS"
        NCPUS=$((N_GPUS * 4))
        MEM=$((N_GPUS * 64))gb
        VARS=$(pbs_vars "$model_name")
        JOB_NAME="pb-eval-$(echo "$model_name" | tr '.' '-')"
        CMD=(
            qsub
            -N "$JOB_NAME"
            -q "$PBS_QUEUE"
            -l "select=1:mem=${MEM}:ncpus=${NCPUS}:ngpus=${N_GPUS}"
            -l "walltime=${WALLTIME}"
            -o "${LOGS_PATH}/"
            -e "${LOGS_PATH}/"
            -j oe
            -v "$VARS"
            "$PBS_SCRIPT"
        )
        printf "  %-32s  ngpus=%-2s  " "$model_name" "$N_GPUS"
    fi

    if $DRY_RUN; then
        echo "[DRY RUN] ${CMD[*]}"
    else
        JOB_ID=$("${CMD[@]}")
        echo "$JOB_ID"
        SUBMITTED=$((SUBMITTED + 1))
    fi
done

echo ""
echo "Submitted: $SUBMITTED  Skipped: $SKIPPED"
