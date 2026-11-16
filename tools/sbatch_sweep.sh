#!/bin/bash
# Finetune sweep, one GPU per arm: {cosine,constant} LR x {proprio,privileged} actor observation,
# Muon+Adam, starting from checkpoints/lafan1_g1_it10000.pt. Each array task runs the trainer plus
# the eval/video watcher on the same GPU; both report to one W&B run through the trainer.
# Submit from the repository root:
#   sbatch tools/sbatch_sweep.sh [extra train.py args...]
#   ITERS=300 GROUP=smoke PROJECT=supermimic-vibed-smoke sbatch --array=1 --time=00:40:00 tools/sbatch_sweep.sh --ckpt_every 100
# A single arm outside the 2x2, e.g. from scratch (INIT= disables the checkpoint and the finetune warm-ups):
#   INIT= SCHED=constant ACTOR_OBS=privileged_official GROUP=scratch-muon sbatch --array=0 tools/sbatch_sweep.sh
#SBATCH --job-name=sm-ft
#SBATCH --partition=gpu
#SBATCH --nodelist=g4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/slurm/%x-%A_%a.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

SCHEDS=(cosine cosine constant constant)
OBS=(proprio privileged proprio privileged)
i=${SLURM_ARRAY_TASK_ID:-0}
sched=${SCHED:-${SCHEDS[$i]}}
obs=${ACTOR_OBS:-${OBS[$i]}}

ITERS=${ITERS:-20000}
CRITIC_ONLY_ITERS=${CRITIC_ONLY_ITERS:-200}
INIT=${INIT-checkpoints/lafan1_g1_it10000.pt}
# finetune: short world-model warm-up, then fit the critic (the checkpoint has none) before the policy moves
INIT_ARGS=()
[ -n "$INIT" ] && INIT_ARGS=(--init_from "$INIT" --wm_only_iters 100 --critic_only_iters "$CRITIC_ONLY_ITERS")
PROJECT=${PROJECT:-supermimic-vibed}
GROUP=${GROUP:-ft-muon}
RUN=${obs}_${sched}
LOG=logs/$GROUP/${RUN}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
mkdir -p "$LOG"
cp -r supermimic "$LOG/code"  # the tree is uncommitted: keep what this run actually executed
# one kernel cache per arm: the arms compile different graphs and start cold at the same time
export SM_CACHE_DIR=/tmp/$USER/supermimic_cache_$RUN

echo "[sweep] $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} arm=$RUN init=${INIT:-scratch} iters=$ITERS log=$LOG"
# cosine decays over the whole run; the repository default (250k) would be flat over a finetune
tools/uv_run.sh supermimic/train.py --log_dir "$LOG" \
  "${INIT_ARGS[@]}" --optimizer muon \
  --lr_schedule "$sched" --lr_decay_iters "$ITERS" --iterations "$ITERS" --actor_obs "$obs" \
  --ckpt_every_min 15 \
  --wandb_project "$PROJECT" --wandb_group "$GROUP" --wandb_name "$RUN" "$@" \
  > "$LOG/stdout.log" 2>&1 &
TRAIN=$!
tools/uv_run.sh supermimic/evaluate.py --run_dir "$LOG" --watch --video --train_pid $TRAIN \
  > "$LOG/eval_stdout.log" 2>&1 &
wait $TRAIN
rc=$?
wait
echo "[sweep] train exit $rc"
exit $rc
