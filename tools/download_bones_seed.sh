#!/bin/bash
# Download the G1 retarget of BONES-SEED (https://huggingface.co/datasets/bones-studio/seed) into data/bones_seed/.
# The dataset is gated (auto-approved form + BONES-SEED License, https://bones.studio/info/seed-license): the
# Hugging Face account of the cached login (`hf auth whoami`) must have been granted access on the dataset page.
# Only what the G1 pipeline needs is fetched: g1.tar.gz (23.5 GB; the two SOMA archives, 45 GB each, are skipped),
# the metadata table, the licence and the dataset card. The archive is NOT extracted: tools/bones_seed_convert.py
# reads it as a stream. `hf download` resumes a partial file, so re-running this script continues the download.
#   bash tools/download_bones_seed.sh                       # head node
#   sbatch tools/download_bones_seed.sh                     # CPU-only job (g0 has outbound internet)
#SBATCH --job-name=bones-dl
#SBATCH --partition=debug
#SBATCH --nodelist=g0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
OUT=data/bones_seed
mkdir -p "$OUT"
LOG=$OUT/download.log
echo "[$(date '+%F %T')] start on $(hostname)" >> "$LOG"
for attempt in 1 2 3 4 5 6 7 8; do
  .venv/bin/hf download bones-studio/seed --repo-type dataset --local-dir "$OUT" \
    g1.tar.gz metadata/seed_metadata_v004.parquet LICENSE.md README.md >> "$LOG" 2>&1 && break
  echo "[$(date '+%F %T')] attempt $attempt failed, retrying in 60 s" >> "$LOG"
  sleep 60
done
if [ -f "$OUT/g1.tar.gz" ] && [ "$(stat -c %s "$OUT/g1.tar.gz")" = 23499973647 ]; then
  echo "[$(date '+%F %T')] done: $(du -sh "$OUT" | cut -f1) in $OUT" >> "$LOG"
else
  echo "[$(date '+%F %T')] INCOMPLETE: g1.tar.gz missing or wrong size" >> "$LOG"
  exit 1
fi
