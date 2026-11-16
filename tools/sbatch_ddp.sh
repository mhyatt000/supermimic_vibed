#!/bin/bash
# One training run on N GPUs of one node: N processes under torchrun, one per GPU, each with its own simulator
# and replay buffer, stepping the same networks with rank-averaged gradients (supermimic/dist.py; README,
# Multi-GPU). Every train.py size is per rank: 4 ranks collect and fit 4x the data per iteration.
# Same knobs and conventions as tools/sbatch_sweep.sh; the eval/video watcher is a separate 1-GPU job.
# Submit both from the repository root:
#   jid=$(RUN=proprio_cosine_x4 sbatch --parsable tools/sbatch_ddp.sh [extra train.py args...])
#   sbatch --dependency=after:$jid tools/sbatch_watch.sh logs/ddp/proprio_cosine_x4_$jid
# Other GPU counts: sbatch --gres=gpu:2 --cpus-per-task=32 --mem=128G ...   (16 CPUs + 64G per GPU)
# Functional test, 2 ranks on the one RTX 2080 of a debug node (NRANKS > GPUs: gloo instead of NCCL):
#   NRANKS=2 PROJECT= INIT= GROUP=smoke ITERS=40 sbatch -p debug -w g5 --gres=gpu:1 -c 10 --mem=28G -t 30 tools/sbatch_ddp.sh \
#     --num_envs 64 --buffer_steps 200 --wm_batch 64 --pi_batch 64 --wm_only_iters 10 --log_every 10 --ckpt_every 20
#SBATCH --job-name=sm-ddp
#SBATCH --partition=gpu
#SBATCH --nodelist=g4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

sched=${SCHED:-cosine}
obs=${ACTOR_OBS:-proprio}
ITERS=${ITERS:-20000}
CRITIC_ONLY_ITERS=${CRITIC_ONLY_ITERS:-200}
INIT=${INIT-checkpoints/lafan1_g1_it10000.pt}
INIT_ARGS=()
[ -n "$INIT" ] && INIT_ARGS=(--init_from "$INIT" --wm_only_iters 100 --critic_only_iters "$CRITIC_ONLY_ITERS")
NGPU=$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES:?no GPU granted}" | grep -c .)
N=${NRANKS:-$NGPU}
PROJECT=${PROJECT-supermimic-vibed}  # PROJECT= turns W&B off
GROUP=${GROUP:-ddp}
RUN=${RUN:-${obs}_${sched}_x$N}
WB_ARGS=()
[ -n "$PROJECT" ] && WB_ARGS=(--wandb_project "$PROJECT" --wandb_group "$GROUP" --wandb_name "$RUN")
LOG=logs/$GROUP/${RUN}_$SLURM_JOB_ID
mkdir -p "$LOG"
cp -r supermimic "$LOG/code"  # the tree is uncommitted: keep what this run actually executed
export SM_CACHE_DIR=/tmp/$USER/supermimic_cache_ddp_$RUN
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( ${SLURM_CPUS_PER_TASK:-8} / N ))}  # torchrun would otherwise force 1

# the watcher job only stops on <run>/done: write it also when the ranks were killed and could not
finish() { [ -e "$LOG/done" ] || echo aborted > "$LOG/done"; }
trap finish EXIT
trap 'exit 143' TERM

echo "[ddp] $(hostname) gpus=$CUDA_VISIBLE_DEVICES ranks=$N run=$RUN init=${INIT:-scratch} iters=$ITERS log=$LOG"
# torchrun stops the other ranks when one fails. stdout.log: rank 0; ranks/: every rank's own stdout and stderr
tools/uv_run.sh -m torch.distributed.run --standalone --nproc_per_node "$N" \
  --tee 3 --log_dir "$LOG/ranks" --local_ranks_filter 0 \
  supermimic/train.py --log_dir "$LOG" \
  "${INIT_ARGS[@]}" --optimizer muon \
  --lr_schedule "$sched" --lr_decay_iters "$ITERS" --iterations "$ITERS" --actor_obs "$obs" \
  --ckpt_every_min 15 \
  "${WB_ARGS[@]}" "$@" \
  > "$LOG/stdout.log" 2>&1 &
wait $!
rc=$?
echo "[ddp] train exit $rc"
exit $rc
