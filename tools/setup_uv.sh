#!/bin/bash
# Docker-free setup: clone Isaac Lab at the pinned commit and install its locked kit-less dependency set into
# .venv with uv (the `uv sync` of IsaacLab/docker/Dockerfile.kitless, plus the wandb extra). About 12 GB.
# Needs uv (https://docs.astral.sh/uv/), git, gcc/g++ (torch.compile) and ffmpeg on PATH (videos).
# Afterwards run everything through tools/uv_run.sh, e.g. SM_PYTHON=tools/uv_run.sh tools/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
ISAACLAB_COMMIT=46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
export GIT_LFS_SKIP_SMUDGE=1  # the LFS assets are not needed
if [ ! -d IsaacLab ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git
fi
git -C IsaacLab checkout "$ISAACLAB_COMMIT"
# --frozen: install IsaacLab/uv.lock as is (torch 2.11 + cu130, warp 1.17, newton, mujoco-warp 3.15)
cd IsaacLab
UV_PROJECT_ENVIRONMENT="$REPO/.venv" uv sync --frozen --inexact --extra all --extra importers --extra test --extra wandb
"$REPO/.venv/bin/python" -c "import torch, warp, newton, mujoco_warp, wandb; print('torch', torch.__version__, '| warp', warp.__version__)"
