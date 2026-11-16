"""End-to-end test of the BONES-SEED tools on stand-in data (the real dataset is gated).

LAFAN1 clips are re-encoded in the BONES-SEED G1 CSV format (header row; root_translate in centimetres;
root_rotate as extrinsic-xyz Euler angles in degrees; ``<joint>_dof`` in degrees; columns deliberately shuffled) and
packed as ``g1/csv/<date>/<name>.csv`` in a tar.gz. tools/bones_seed_convert.py must give back the frames that
MotionLib computes from the original files, and tools/bones_seed_filter.py must keep the clean clips and drop
clips renamed or edited to break each rule.

  TMPDIR=<scratch> tools/uv_run.sh tools/test_bones_seed_pipeline.py --clips 6
"""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tarfile
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
ap.add_argument("--clips", type=int, default=6)
ap.add_argument("--pattern", type=str, default="walk*.csv", help="which LAFAN1 files")
ap.add_argument("--out", type=str, default="", help="keep the stand-in archive and cache here")
args = ap.parse_args()

import glob  # noqa: E402

import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from supermimic.kinematics import G1Kinematics  # noqa: E402
from supermimic.motion import MotionLib  # noqa: E402
from supermimic.motion_cache import open_frames, read_clips, read_manifest  # noqa: E402
from supermimic.robot import G1_JOINT_NAMES  # noqa: E402

files = sorted(glob.glob(os.path.join(args.motion_dir, args.pattern)))[: args.clips]
tmp = args.out or tempfile.mkdtemp(prefix="bones_standin_")
os.makedirs(tmp, exist_ok=True)
tar_path, cache = os.path.join(tmp, "g1.tar.gz"), os.path.join(tmp, "cache")
rng = np.random.default_rng(0)


def encode(raw: np.ndarray) -> bytes:
    """LAFAN1 rows (p m, quat xyzw, 29 joints rad) -> a BONES-SEED G1 CSV."""
    eul = Rotation.from_quat(raw[:, 3:7]).as_euler("xyz", degrees=True)
    cols = {"Frame": np.arange(len(raw), dtype=np.float64)}
    for k, a in enumerate("XYZ"):
        cols[f"root_translate{a}"], cols[f"root_rotate{a}"] = raw[:, k] * 100.0, eul[:, k]
    for k, j in enumerate(G1_JOINT_NAMES):
        cols[f"{j}_dof"] = np.rad2deg(raw[:, 7 + k])
    order = ["Frame"] + list(rng.permutation([c for c in cols if c != "Frame"]))  # readers must go by name
    buf = io.StringIO()
    np.savetxt(buf, np.stack([cols[c] for c in order], 1), delimiter=",", header=",".join(order), comments="", fmt="%.9g")
    return buf.getvalue().encode()


# variants of the first clip that each rule must catch: (suffix or name, edit)
def lift(raw, dz):
    out = raw.copy()
    out[:, 2] += dz
    return out


def hover(raw):  # stand still 5 cm above the floor for 2 s
    out = np.repeat(raw[:1], 60, 0)
    out[:, 2] += 0.05
    return out


expected = {}
with tarfile.open(tar_path, "w:gz") as tar:

    def add(date, name, raw, verdict):
        data = encode(raw)
        info = tarfile.TarInfo(f"g1/csv/{date}/{name}.csv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        expected[name] = verdict

    raws = [np.loadtxt(f, delimiter=",") for f in files]
    for k, (f, raw) in enumerate(zip(files, raws)):
        add(f"2401{k % 3 + 1:02d}", os.path.splitext(os.path.basename(f))[0], raw, "keep")
    short = raws[0][:300]
    add("240104", "walk_up_stairs_001", short, "name")
    add("240104", "Sitting_down_chair_R_002", short, "name")
    add("240104", "look_over_fence_003", short, "name")
    add("240104", "transition_position_004", short, "keep")  # contains "sit" but none of the keywords
    add("240105", "raised_005", lift(short, 0.40), "elevated_support")
    add("240105", "sunk_006", lift(short, -0.20), "ground_penetration")
    add("240105", "hover_007", hover(short), "floating_feet")
    add("240105", "one_frame_008", short[:1], "error")

run = lambda *a: subprocess.run([sys.executable, *a], check=True, cwd=REPO)  # noqa: E731
run(os.path.join(REPO, "tools", "bones_seed_convert.py"), "--src", tar_path, "--out", cache, "--src_fps", "30", "--workers", "4", "--raw_sample_every", "2")
run(os.path.join(REPO, "tools", "bones_seed_filter.py"), "--cache", cache, "--workers", "4")

# ---- conversion: the clean clips equal MotionLib on the original files
names, start, length = read_clips(cache)
mm = open_frames(cache)
kin = G1Kinematics("cpu")
ref = MotionLib(files, kin, "cpu")
worst = 0.0
for c, n in enumerate(ref.names):
    k = names.index(n)
    x = torch.from_numpy(np.array(mm[int(start[k]) : int(start[k] + length[k])]))
    a, b = int(ref.clip_start[c]), int(ref.clip_end[c]) + 1
    assert len(x) == b - a, (n, len(x), b - a)
    from supermimic.rotations import quat_to_mat

    # MotionLib computes the resampling phase in float32 (frame n is source time 0.6 n), the converter in float64:
    # on long clips the phase differs by up to 5e-4 source frames, so compare the first 200 frames tightly
    d = torch.cat(((x[:, :3] - ref.root_p[a:b]).abs(), (quat_to_mat(x[:, 3:7]) - ref.root_R[a:b]).abs().flatten(1), (x[:, 7:] - ref.q[a:b]).abs()), 1)
    worst = max(worst, d.max().item())
    assert d[:200].max() < 1e-5 and d.max() < 2e-3, (n, d[:200].max().item(), d.max().item())
print(f"[test] conversion matches MotionLib on {len(ref.names)} clips (largest difference {worst:.2e}, float32 resampling phase of MotionLib)")

# ---- filters
kept = set(read_manifest(cache))
rej = {}
for line in open(os.path.join(cache, "rejected.csv")).read().splitlines()[1:]:
    n, p, reason, _ = line.split(",", 3)
    rej[n] = "name" if p == "name" else reason
errs = {line.split(",")[1] for line in open(os.path.join(cache, "convert_errors.csv")).read().splitlines()[1:]}
originals = {os.path.splitext(os.path.basename(f))[0] for f in files}
dropped = {}
for n, verdict in expected.items():
    got = "keep" if n in kept else "error" if n in errs else rej.get(n, "missing")
    if n in originals and got not in ("keep", "error", "missing", "name"):
        dropped[n] = got  # a real LAFAN1 clip that breaks a feasibility rule (two of the 40 do): reported, not a failure
        continue
    assert verdict in got.split("+"), (n, verdict, got)
print(f"[test] filter verdicts as expected: { {n: v for n, v in expected.items() if n not in originals} }; {len(originals) - len(dropped)} of {len(originals)} "
      f"original clips kept, dropped by the feasibility filter: {dropped}")  # fmt: skip
print(f"PASS (stand-in archive and cache in {tmp})" if args.out else "PASS")
if not args.out:
    import shutil

    shutil.rmtree(tmp)
