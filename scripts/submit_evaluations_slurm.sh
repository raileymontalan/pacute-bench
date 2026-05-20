#!/bin/bash
# Submit one SLURM job per model defined in the model config YAMLs.
#
# Usage:
#   bash scripts/submit_evaluations_slurm.sh [OPTIONS]
#
# Options:
#   --pt-only              Only submit PT (pretrained) models
#   --it-only              Only submit IT (instruction-tuned) models
#   --commercial-only      Only submit commercial API models (OpenAI/Anthropic, no GPU)
#   --model <name>         Submit only this model (repeatable)
#   --overwrite            Re-run benchmarks that already have results
#   --max-samples <n>      Cap samples per benchmark
#   --benchmarks <b...>    Only run these benchmarks (e.g. pacute-syllabification-gen)
#   --output-dir <path>    Where to write evaluation results (default: $RESULTS_PATH from .env)
#   --port <n>             vLLM server port (default: auto)
#   --partition <p>        SLURM partition (default: high)
#   --walltime <hh:mm:ss>  Job walltime (default: 12:00:00)
#   --filter <pattern>     Only submit models whose name contains <pattern>
#   --dry-run              Print sbatch commands without submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/eval_model.slurm"

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
OUTPUT_DIR="${RESULTS_PATH:-}"
VLLM_PORT=""
SLURM_PARTITION="high"
WALLTIME="12:00:00"
SINGLE_MODELS=()
FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pt-only)          PT_ONLY=true ;;
        --it-only)          IT_ONLY=true ;;
        --commercial-only)  COMMERCIAL_ONLY=true ;;
        --dry-run)          DRY_RUN=true ;;
        --overwrite)        OVERWRITE=true ;;
        --port)             VLLM_PORT="$2"; shift ;;
        --max-samples)      MAX_SAMPLES="$2"; shift ;;
        --benchmarks)       shift
                            while [[ $# -gt 0 && "$1" != --* ]]; do
                                BENCHMARKS="${BENCHMARKS:+${BENCHMARKS},}$1"
                                shift
                            done
                            continue ;;
        --output-dir)       OUTPUT_DIR="$2"; shift ;;
        --partition)        SLURM_PARTITION="$2"; shift ;;
        --walltime)         WALLTIME="$2"; shift ;;
        --model)            SINGLE_MODELS+=("$2"); shift ;;
        --filter)           FILTER="$2"; shift ;;
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

# ── YAML helpers ──────────────────────────────────────────────────────────────
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

get_model_tp() {
    local name="$1"
    python3 - <<PYEOF
import yaml, sys
for cfg in ['$PROJECT_ROOT/configs/models_pt.yaml', '$PROJECT_ROOT/configs/models_it.yaml']:
    try:
        data = yaml.safe_load(open(cfg))
        info = data['models'].get('$name')
        if info:
            print(info.get('tp', 1))
            sys.exit(0)
    except Exception:
        pass
print(1)
PYEOF
}

# ── Apply --filter ────────────────────────────────────────────────────────────
if [[ -n "$FILTER" ]]; then
    FILTERED=()
    for m in "${ALL_MODELS[@]}"; do
        [[ "$m" == *"$FILTER"* ]] && FILTERED+=("$m")
    done
    ALL_MODELS=("${FILTERED[@]}")
fi

mkdir -p "$LOGS_PATH"

echo "Submitting ${#ALL_MODELS[@]} model evaluation job(s) to partition: $SLURM_PARTITION"
echo "SLURM script: $SLURM_SCRIPT"
echo ""

SUBMITTED=0
SKIPPED=0

for model_name in "${ALL_MODELS[@]}"; do
    model_path=$(get_model_path "$model_name" 2>/dev/null) || {
        echo "  SKIP: '$model_name' not found in config YAMLs"
        SKIPPED=$((SKIPPED + 1))
        continue
    }

    JOB_NAME="pb-eval-$(echo "$model_name" | tr '.' '-')"

    if $COMMERCIAL_ONLY; then
        declare -a EXPORT_VARS=(
            MODEL_NAME="$model_name"
            PROJECT_ROOT="$PROJECT_ROOT"
            OVERWRITE="$( $OVERWRITE && echo true || echo false )"
        )
        [[ -n "$OUTPUT_DIR" ]]  && EXPORT_VARS+=(OUTPUT_DIR="$OUTPUT_DIR")
        [[ -n "$MAX_SAMPLES" ]] && EXPORT_VARS+=(MAX_SAMPLES="$MAX_SAMPLES")
        [[ -n "$BENCHMARKS" ]]  && EXPORT_VARS+=(BENCHMARKS="$BENCHMARKS")

        CMD=(
            env "${EXPORT_VARS[@]}"
            sbatch
            --job-name="$JOB_NAME"
            --partition="$SLURM_PARTITION"
            --nodes=1
            --ntasks-per-node=1
            --cpus-per-task=2
            --mem=8G
            --time="${WALLTIME:-26:00:00}"
            --output="${LOGS_PATH}/${JOB_NAME}_%j.out"
            "$SCRIPT_DIR/eval_commercial.slurm"
        )
        printf "  %-32s  (commercial API)  " "$model_name"
    else
        N_GPUS=$(get_model_tp "$model_name")
        NCPUS=$((N_GPUS * 4))
        MEM=$((N_GPUS * 64))G

        declare -a EXPORT_VARS=(
            MODEL_NAME="$model_name"
            PROJECT_ROOT="$PROJECT_ROOT"
            OVERWRITE="$( $OVERWRITE && echo true || echo false )"
        )
        [[ -n "$OUTPUT_DIR" ]]  && EXPORT_VARS+=(OUTPUT_DIR="$OUTPUT_DIR")
        [[ -n "$VLLM_PORT" ]]   && EXPORT_VARS+=(VLLM_PORT="$VLLM_PORT")
        [[ -n "$MAX_SAMPLES" ]] && EXPORT_VARS+=(MAX_SAMPLES="$MAX_SAMPLES")
        [[ -n "$BENCHMARKS" ]]  && EXPORT_VARS+=(BENCHMARKS="$BENCHMARKS")

        CMD=(
            env "${EXPORT_VARS[@]}"
            sbatch
            --job-name="$JOB_NAME"
            --partition="$SLURM_PARTITION"
            --nodes=1
            --ntasks-per-node=1
            --cpus-per-task="$NCPUS"
            --mem="$MEM"
            --gres="gpu:${N_GPUS}"
            --time="$WALLTIME"
            --output="${LOGS_PATH}/${JOB_NAME}_%j.out"
            "$SLURM_SCRIPT"
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
