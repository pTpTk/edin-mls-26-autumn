#!/bin/bash
# Usage: sbatch ../judge_answers.sbatch practice RUN_DIR --allow-subset
# Array 0..3 = exact k=1,5,10,20; 4..7 = HNSW k=1,5,10,20.
# Default concurrency 1 limits simultaneous ELM judge jobs.
#SBATCH --job-name=rag-judge
#SBATCH --partition=Teaching
#SBATCH --array=0-7%1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=rag/results/slurm/judge-%A_%a.out

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the course repository root}"
split=${1:?Usage: judge_answers.sbatch SPLIT RUN_DIR [evaluator flags]}
run_dir=${2:?Candidate directory required}
shift 2
source "${RAG_ENV:-$HOME/mls-rag-env}/bin/activate"
set -u
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

task=${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}
if [[ ! "$task" =~ ^[0-7]$ ]]; then
    echo "Array task must be between 0 and 7" >&2
    exit 2
fi
methods=(exact hnsw)
depths=(1 5 10 20)
method=${methods[$((task / 4))]}
k=${depths[$((task % 4))]}
printf 'Job=%s task=%s method=%s k=%s\n' "$SLURM_JOB_ID" "$task" "$method" "$k"
exec python -u rag/scripts/evaluate_answers.py --split "$split" \
    --candidates "$run_dir/${method}_k${k}.jsonl" \
    --out-dir "$run_dir/judge_${method}_k${k}" "$@"

