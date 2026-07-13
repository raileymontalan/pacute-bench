#!/bin/bash
# Submit one job per model defined in the model config YAMLs.
# Supports both PBS and SLURM schedulers (auto-detected, or set with --scheduler).
#
# Usage:
#   bash scripts/submit_evaluations.sh [OPTIONS]
#
# Options:
#   --scheduler pbs|slurm  Scheduler to use (default: auto-detect via which sbatch/qsub)
#   --pt-only              Only submit PT (pretrained) models
#   --it-only              Only submit IT (instruction-tuned) models
#   --commercial-only      Only submit commercial API models (OpenAI/Anthropic, no GPU)
#   --model <name>         Submit only this model (repeatable)
#   --overwrite            Re-run benchmarks that already have results
#   --max-samples <n>      Cap samples per benchmark
#   --benchmarks <b...>    Only run these benchmarks (e.g. pacute-syllabification-gen)
#   --output-dir <path>    Where to write evaluation results (default: $RESULTS_PATH from .env)
#   --port <n>             vLLM server port (default: auto)
#   --queue <q>            PBS queue (required for PBS; e.g. --queue gpu)
#   --partition <p>        SLURM partition (default: gpu)
#   --walltime <hh:mm:ss>  Job walltime (default: 12:00:00)
#   --filter <pattern>     Only submit models whose name contains <pattern>
#   --dry-run              Print submit commands without submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_ENV_FILE="$(dirname "$SCRIPT_DIR")/.env"
if [[ ! -f "$_ENV_FILE" ]]; then
    echo "ERROR: .env not found at $_ENV_FILE" >&2
    echo "Copy .env.example to .env and fill in your paths." >&2
    exit 1
fi
# shellcheck source=../.env
source "$_ENV_FILE"

# ── Defaults ──────────────────────────────────────────────────────────────────
SCHEDULER=""
PT_ONLY=false
IT_ONLY=false
COMMERCIAL_ONLY=false
DRY_RUN=false
OVERWRITE=false
MAX_SAMPLES=""
BENCHMARKS=""
OUTPUT_DIR="${RESULTS_PATH:-}"
VLLM_PORT=""
PBS_QUEUE=""
SLURM_PARTITION="gpu"
WALLTIME="12:00:00"
SINGLE_MODELS=()
FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scheduler)        SCHEDULER="$2"; shift ;;
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
        --queue)            PBS_QUEUE="$2"; shift ;;
        --partition)        SLURM_PARTITION="$2"; shift ;;
        --walltime)         WALLTIME="$2"; shift ;;
        --model)            SINGLE_MODELS+=("$2"); shift ;;
        --filter)           FILTER="$2"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

# ── Auto-detect scheduler ─────────────────────────────────────────────────────
if [[ -z "$SCHEDULER" ]]; then
    if command -v sbatch &>/dev/null; then
        SCHEDULER="slurm"
    elif command -v qsub &>/dev/null; then
        SCHEDULER="pbs"
    else
        echo "ERROR: no scheduler found. Install sbatch/qsub or pass --scheduler pbs|slurm" >&2
        exit 1
    fi
fi

if [[ "$SCHEDULER" == "pbs" && -z "$PBS_QUEUE" ]]; then
    echo "ERROR: --queue <queue_name> is required for PBS (e.g. --queue gpu)" >&2
    exit 1
fi

# ── Config files ──────────────────────────────────────────────────────────────
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
        done < <(YAML_CFG="$cfg" python3 -c "
import yaml, os
with open(os.environ['YAML_CFG']) as f:
    data = yaml.safe_load(f)
for name in data.get('models', {}):
    print(name)
")
    done
fi

# ── YAML helpers ──────────────────────────────────────────────────────────────
get_model_path() {
    local name="$1"
    MODEL_NAME="$name" PROJECT_ROOT="$PROJECT_ROOT" python3 - <<'PYEOF'
import yaml, sys, os
name = os.environ['MODEL_NAME']
project_root = os.environ.get('PROJECT_ROOT', '')
for cfg_path in [
    f'{project_root}/configs/models_pt.yaml',
    f'{project_root}/configs/models_it.yaml',
    f'{project_root}/configs/models_commercial.yaml',
]:
    try:
        with open(cfg_path) as fh:
            data = yaml.safe_load(fh)
        info = data['models'].get(name)
        if info:
            print(info['path'])
            sys.exit(0)
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Warning: {cfg_path}: {exc}", file=sys.stderr)
sys.exit(1)
PYEOF
}

get_model_tp() {
    local name="$1"
    MODEL_NAME="$name" PROJECT_ROOT="$PROJECT_ROOT" python3 - <<'PYEOF'
import yaml, sys, os
name = os.environ['MODEL_NAME']
project_root = os.environ.get('PROJECT_ROOT', '')
for cfg_path in [
    f'{project_root}/configs/models_pt.yaml',
    f'{project_root}/configs/models_it.yaml',
]:
    try:
        with open(cfg_path) as fh:
            data = yaml.safe_load(fh)
        info = data['models'].get(name)
        if info:
            print(info.get('tp', 1))
            sys.exit(0)
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Warning: {cfg_path}: {exc}", file=sys.stderr)
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

TARGET="$( [[ "$SCHEDULER" == "pbs" ]] && echo "queue: $PBS_QUEUE" || echo "partition: $SLURM_PARTITION" )"
echo "Submitting ${#ALL_MODELS[@]} model evaluation job(s) [$SCHEDULER] to $TARGET"
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
        JOB_NAME="pb-commercial-$(echo "$model_name" | tr '.' '-')"
        if [[ "$SCHEDULER" == "pbs" ]]; then
            VARS="MODEL_NAME=${model_name},PROJECT_ROOT=${PROJECT_ROOT}"
            $OVERWRITE              && VARS="${VARS},OVERWRITE=true"
            [[ -n "$MAX_SAMPLES" ]] && VARS="${VARS},MAX_SAMPLES=${MAX_SAMPLES}"
            [[ -n "$BENCHMARKS" ]]  && VARS="${VARS},BENCHMARKS='${BENCHMARKS}'"
            [[ -n "$OUTPUT_DIR" ]]  && VARS="${VARS},OUTPUT_DIR=${OUTPUT_DIR}"
            CMD=(qsub -N "$JOB_NAME" -q "$PBS_QUEUE"
                 -l "select=1:mem=8gb:ncpus=2" -l "walltime=${WALLTIME}"
                 -o "${LOGS_PATH}/" -e "${LOGS_PATH}/" -j oe
                 -v "$VARS" "$SCRIPT_DIR/eval_commercial.pbs")
        else
            EXPORT_VARS=(MODEL_NAME="$model_name" PROJECT_ROOT="$PROJECT_ROOT"
                         OVERWRITE="$( $OVERWRITE && echo true || echo false )")
            [[ -n "$OUTPUT_DIR" ]]  && EXPORT_VARS+=(OUTPUT_DIR="$OUTPUT_DIR")
            [[ -n "$MAX_SAMPLES" ]] && EXPORT_VARS+=(MAX_SAMPLES="$MAX_SAMPLES")
            [[ -n "$BENCHMARKS" ]]  && EXPORT_VARS+=(BENCHMARKS="$BENCHMARKS")
            CMD=(env "${EXPORT_VARS[@]}" sbatch --job-name="$JOB_NAME"
                 --partition="$SLURM_PARTITION" --nodes=1 --ntasks-per-node=1
                 --cpus-per-task=2 --mem=8G --time="${WALLTIME:-26:00:00}"
                 --output="${LOGS_PATH}/%j-${JOB_NAME}.out"
                 --error="${LOGS_PATH}/%j-${JOB_NAME}.out"
                 "$SCRIPT_DIR/eval_commercial.slurm")
        fi
        printf "  %-32s  (commercial API)  " "$model_name"
    else
        N_GPUS=$(get_model_tp "$model_name")
        NCPUS=$((N_GPUS * 4))
        JOB_NAME="pb-eval-$(echo "$model_name" | tr '.' '-')"
        if [[ "$SCHEDULER" == "pbs" ]]; then
            MEM=$((N_GPUS * 64))gb
            VARS="MODEL_NAME=${model_name},PROJECT_ROOT=${PROJECT_ROOT}"
            [[ -n "$VLLM_PORT" ]]   && VARS="${VARS},VLLM_PORT=${VLLM_PORT}"
            [[ -n "$OUTPUT_DIR" ]]  && VARS="${VARS},OUTPUT_DIR=${OUTPUT_DIR}"
            $OVERWRITE              && VARS="${VARS},OVERWRITE=true"
            [[ -n "$MAX_SAMPLES" ]] && VARS="${VARS},MAX_SAMPLES=${MAX_SAMPLES}"
            [[ -n "$BENCHMARKS" ]]  && VARS="${VARS},BENCHMARKS='${BENCHMARKS}'"
            CMD=(qsub -N "$JOB_NAME" -q "$PBS_QUEUE"
                 -l "select=1:mem=${MEM}:ncpus=${NCPUS}:ngpus=${N_GPUS}"
                 -l "walltime=${WALLTIME}"
                 -o "${LOGS_PATH}/" -e "${LOGS_PATH}/" -j oe
                 -v "$VARS" "$SCRIPT_DIR/eval_model.pbs")
        else
            MEM=$((N_GPUS * 64))G
            EXPORT_VARS=(MODEL_NAME="$model_name" PROJECT_ROOT="$PROJECT_ROOT"
                         OVERWRITE="$( $OVERWRITE && echo true || echo false )")
            [[ -n "$OUTPUT_DIR" ]]  && EXPORT_VARS+=(OUTPUT_DIR="$OUTPUT_DIR")
            [[ -n "$VLLM_PORT" ]]   && EXPORT_VARS+=(VLLM_PORT="$VLLM_PORT")
            [[ -n "$MAX_SAMPLES" ]] && EXPORT_VARS+=(MAX_SAMPLES="$MAX_SAMPLES")
            [[ -n "$BENCHMARKS" ]]  && EXPORT_VARS+=(BENCHMARKS="$BENCHMARKS")
            CMD=(env "${EXPORT_VARS[@]}" sbatch --job-name="$JOB_NAME"
                 --partition="$SLURM_PARTITION" --nodes=1 --ntasks-per-node=1
                 --cpus-per-task="$NCPUS" --mem="$MEM" --gres="gpu:${N_GPUS}"
                 --time="$WALLTIME"
                 --output="${LOGS_PATH}/%j-${JOB_NAME}.out"
                 --error="${LOGS_PATH}/%j-${JOB_NAME}.out"
                 "$SCRIPT_DIR/eval_model.slurm")
        fi
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
