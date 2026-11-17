#!/bin/bash
# Submit from the course repository root, after creating rag/results/slurm/.
# Usage: sbatch ../cache_transcripts.sbatch practice
# Set ASR_CUDA=none before submission to use the PyTorch reference.
#SBATCH --job-name=rag-transcripts
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=rag/results/slurm/transcripts-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the course repository root}"
split=${1:-practice}
case "$split" in
    practice|dev) ;;
    *) echo "Split must be practice or dev" >&2; exit 2 ;;
esac

source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate mls-asr
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
source "${RAG_ENV:-$HOME/mls-rag-env}/bin/activate"
set -u

export PATH=/opt/cuda-12.8.0/bin:$PATH
export ASR_CUDA=${ASR_CUDA:-all}
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Job=%s split=%s ASR_CUDA=%s\n' "$SLURM_JOB_ID" "$split" "$ASR_CUDA"
hostname
nvidia-smi -L
make -C hw1-asr lib
exec python -u rag/scripts/evaluate_rag.py --device cpu --split "$split" \
    --transcript-cache "rag/.cache/${split}_transcripts.jsonl"

