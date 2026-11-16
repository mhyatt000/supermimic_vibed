#!/bin/bash
# BONES-SEED, from the gated Hugging Face dataset to the cache the trainer reads, as one CPU-only job:
#   1. tools/download_bones_seed.sh     g1.tar.gz (23.5 GB) into data/bones_seed/   (skipped once complete; resumes)
#   2. tools/bones_seed_convert.py      the archive, read as a stream, to data/bones_seed/cache (50 Hz, about 6.4 GiB)
#   3. tools/bones_seed_filter.py       Appendix E name and feasibility filters: manifest.txt, rejected.csv,
#                                       filter_report.json (counts next to the paper's 129,667 and 129,514)
#   4. tools/bones_seed_verify.py       units, up axis, Euler order, frame rate and stance, asked of the data itself
# Needs the Hugging Face account of the cached login to have been granted access on
# https://huggingface.co/datasets/bones-studio/seed (form + licence); without it step 1 fails and the job stops.
#   sbatch tools/sbatch_bones_prep.sh            # g0: CPU only, outbound internet, no GPU requested
#   bash tools/sbatch_bones_prep.sh              # or directly on the head node
# Afterwards train with  --motion_dir data/bones_seed/cache  (see supermimic/motion_cache.py).
#SBATCH --job-name=bones-prep
#SBATCH --partition=debug
#SBATCH --nodelist=g0
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
OUT=data/bones_seed
W=$((${SLURM_CPUS_PER_TASK:-$(nproc)} - 2))
step() { echo "== $(date '+%F %T') $*"; "$@" || { echo "== FAILED ($?): $*"; exit 1; }; }

if [ ! -f "$OUT/g1.tar.gz" ] || [ "$(stat -c %s "$OUT/g1.tar.gz")" != 23499973647 ]; then
  step bash tools/download_bones_seed.sh
fi
if [ ! -f "$OUT/cache/motion_cache.json" ]; then
  step tools/uv_run.sh tools/bones_seed_convert.py --src "$OUT/g1.tar.gz" --out "$OUT/cache" --workers "$W"
fi
step tools/uv_run.sh tools/bones_seed_filter.py --cache "$OUT/cache" --workers "$W" "$@"
# the verdicts are printed and written to verify_report.json; a failed check is reported, the cache stays usable
tools/uv_run.sh tools/bones_seed_verify.py --cache "$OUT/cache" || echo "== tools/bones_seed_verify.py: at least one check did not pass, read the lines above"
echo "== $(date '+%F %T') done: $(du -sh "$OUT" | cut -f1) in $OUT"
