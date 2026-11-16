#!/bin/bash
# Docker-free launcher: run a script with the uv-managed venv (.venv), which is installed from
# IsaacLab/uv.lock exactly as IsaacLab/docker/Dockerfile.kitless does (see the header of that file).
# Usage: tools/uv_run.sh <script.py> [args...]
#   srun -p gpu -w g4 --gres=gpu:1 --cpus-per-task=16 --mem=64G tools/uv_run.sh supermimic/train.py --log_dir logs/run
# Do not add `ulimit -v` (see README, Deviations).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --- environment of the kit-less image (Dockerfile.kitless, runtime stage)
export ISAACLAB_PATH="$REPO/IsaacLab"
export VIRTUAL_ENV="$REPO/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export LANG=C.UTF-8
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
# keep a stray `uv run`/`uv sync` from rebuilding or replacing the venv
export UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV"
export UV_NO_SYNC=1

# --- caches (docker/run_container.sh keeps them in the repo's .cache/, which is NFS here).
# Warp kernels and the torch.compile (inductor + triton) cache go to node-local disk: shared by
# all jobs of this user on the node, never by two nodes, and file locks work. Override with
# SM_CACHE_DIR (e.g. a per-job directory) if needed.
SM_CACHE_DIR="${SM_CACHE_DIR:-${TMPDIR:-/tmp}/${USER:-$(id -un)}/supermimic_cache}"
export SM_CACHE_DIR
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$SM_CACHE_DIR/warp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SM_CACHE_DIR/inductor}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SM_CACHE_DIR/mpl}"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"

# --- C compiler for torch.compile (the Docker image adds build-essential): prefer the system gcc
if [ -x /usr/bin/gcc ] && [ -x /usr/bin/g++ ]; then
  export CC="${CC:-/usr/bin/gcc}"
  export CXX="${CXX:-/usr/bin/g++}"
fi

# --- headless rendering needs nothing here: tools/render.py defaults MUJOCO_GL=egl itself (and
# mujoco then sets PYOPENGL_PLATFORM=egl); the GPU nodes ship the glvnd EGL vendor files, and
# ffmpeg comes from PATH. MUJOCO_GL is deliberately not exported globally: it would make every
# `import mujoco` (training included) load libEGL, which nodes without it (head0) cannot do.

# ffmpeg (tools/render.py) is the user's pixi-global build on the shared home; make sure a job
# submitted from a shell without ~/.pixi/bin on PATH still finds it
command -v ffmpeg >/dev/null 2>&1 || export PATH="$PATH:$HOME/.pixi/bin"

exec "$VIRTUAL_ENV/bin/python" "$@"
