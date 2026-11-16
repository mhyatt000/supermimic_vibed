#!/bin/bash
# Eval/video watcher of one training run as its own 1-GPU job (for runs whose GPUs are all busy training, see
# tools/sbatch_ddp.sh): evaluates every new checkpoint of <run_dir>, renders videos, and exits once the trainer
# has written <run_dir>/done. The trainer's W&B logger picks eval.csv and videos/ up from the run directory.
#   sbatch --dependency=after:<train job id> tools/sbatch_watch.sh <run_dir> [extra evaluate.py args...]
# On the trainer's node (-w g4, when a GPU is free there) the trainer also waits for the last evaluation
# before it closes the W&B run; from another node it cannot see the watcher and does not wait.
#SBATCH --job-name=sm-watch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
LOG=${1:?usage: sbatch tools/sbatch_watch.sh <run_dir> [evaluate.py args...]}
shift
mkdir -p "$LOG"
export SM_CACHE_DIR=/tmp/$USER/supermimic_cache_watch
echo "[watch] $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} run=$LOG"
tools/uv_run.sh supermimic/evaluate.py --run_dir "$LOG" --watch --video "$@" > "$LOG/eval_stdout.log" 2>&1
rc=$?
echo "[watch] exit $rc"
exit $rc
