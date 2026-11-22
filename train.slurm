#!/bin/bash
#SBATCH --job-name=nanotabicl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Submit from the repository root. Slurm copies this script into its spool,
# so the script's own directory is not the checkout directory.
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"
if [[ ! -f train.py || ! -x .venv/bin/python ]]; then
    echo "Submit from the repository root after uv sync --locked has finished." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv must be available on PATH inside the Slurm job." >&2
    exit 1
fi

if (( $# == 0 )); then
    set -- --run-dir runs/cuda-seed0
fi

# Reuse the installed uv environment; no dependency downloads on compute nodes.
# A failed CUDA preflight stops the job before training starts.
srun --ntasks=1 uv run --frozen --no-sync python -u preflight.py --device cuda
exec srun --ntasks=1 uv run --frozen --no-sync python -u train.py --device cuda "$@"
