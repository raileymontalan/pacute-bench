#!/bin/bash
#
# One-time pacute-bench environment setup on a GPU node.
#
# Run this once before submitting evaluation jobs. It installs
# the Python venv with vLLM on a GPU node so CUDA compilation
# works correctly against the custom driver.
#
# Usage:
#   sbatch setup_env.slurm

#SBATCH --job-name=pacute-bench-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodelist=smc-pod-[14,10,9,8,5,3]
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.out

set -euo pipefail
cd $SLURM_SUBMIT_DIR

mkdir -p logs

module use "${MODULEFILES_PATH}"
module swap cuda/13.0.3

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

echo "Creating venv and installing pacute-bench + vLLM..."
uv sync

echo ""
echo "Verifying vLLM install..."
.venv/bin/python -c "import vllm; print(f'vLLM {vllm.__version__} OK')"

echo ""
echo "Setup complete. Virtual environment is ready at .venv/"
