#!/bin/bash
# Submit from the course repository root, after creating rag/results/slurm/.
# Usage: sbatch ../run_experiment.sbatch [run_retrieval_experiment.py arguments]
#SBATCH --job-name=rag-experiment
#SBATCH --partition=Teaching
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=rag/results/slurm/experiment-%j.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the course repository root}"
source "${RAG_ENV:-$HOME/mls-rag-env}/bin/activate"
set -u
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Job=%s\n' "$SLURM_JOB_ID"
hostname
exec python -u rag/extensions/run_retrieval_experiment.py "$@"

