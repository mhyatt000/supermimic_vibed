"""CachedMotionLib (supermimic/motion_cache.py) against MotionLib on the LAFAN1 clips, on the CPU.

Builds a cache of the LAFAN1 CSVs with MotionLib's own resampling, loads it through ``MotionLib.from_dir`` and
compares every attribute and method with the MotionLib of the same clips; then checks the reward, termination and
critic features of ``supermimic.mdp`` on both, a manifest, and the seeded evaluation subset.
  tools/uv_run.sh tools/test_motion_cache.py [--clips 6] [--out <dir>]
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.motion import CONTROL_DT, MotionLib
from supermimic.motion_cache import MANIFEST, CacheWriter, CachedMotionLib, choose_clips, load_subset
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import quat_slerp, so3_exp

ap = argparse.ArgumentParser()
ap.add_argument("--clips", type=int, default=6, help="number of LAFAN1 clips (0: all)")
ap.add_argument("--out", type=str, default=None, help="keep the cache here (default: a temporary directory)")
ap.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
args = ap.parse_args()


def lafan1_frames(f: str, src_fps: float = 30.0) -> np.ndarray:
    """MotionLib's loading and 50 Hz resampling of one LAFAN1 CSV, as cache rows (p, quat wxyz, q)."""
    raw = torch.tensor(np.loadtxt(f, delimiter=","), dtype=torch.float32)
    p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
    quat = torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1)
    T_src = raw.shape[0]
    n = int(np.floor((T_src - 1) / src_fps / CONTROL_DT)) + 1
    t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
    i0 = torch.clamp(t.floor().long(), max=T_src - 2)
    a = (t - i0.float()).clamp(0, 1)
    p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
    j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
    return torch.cat((p, quat_slerp(quat[i0], quat[i0 + 1], a), j), -1).numpy()


files = sorted(glob.glob(os.path.join(args.motion_dir, "*.csv")))
if args.clips:
    files = files[:: max(len(files) // args.clips, 1)][: args.clips]
out = args.out or tempfile.mkdtemp(prefix="motion_cache_test_")
w = CacheWriter(out, {"source": "LAFAN1 G1 (test)"})
for f in files:
    w.add(os.path.splitext(os.path.basename(f))[0], lafan1_frames(f), src_dir="lafan1")
w.close()

kin = G1Kinematics("cpu")
ref = MotionLib(files, kin, "cpu")
small = MotionLib.from_dir(out, kin, "cpu")
assert type(small) is CachedMotionLib and small.materialized  # a small library stores the derived quantities
lib = CachedMotionLib(out, kin, "cpu", materialize=False)  # the path of a large one: computed per look-up
assert not torch.is_tensor(lib.body_p) and torch.is_tensor(small.body_p)
worst = 0.0


def same(name: str, a, b, tol: float = 0.0):
    global worst
    a, b = (x.float() if x.dtype == torch.bool else x for x in (a, b))
    assert a.shape == b.shape and a.dtype == b.dtype, (name, a.shape, b.shape, a.dtype, b.dtype)
    err = (a - b).abs().max().item() if a.numel() else 0.0
    worst = max(worst, err)
    assert err <= tol, f"{name}: max abs difference {err:.3e} > {tol:.1e}"
    print(f"  {name:<22s} max abs difference {err:.2e}")


assert lib.names == ref.names and lib.num_frames == ref.num_frames and lib.num_bins == ref.num_bins
for k in ("clip_len", "clip_start", "clip_end", "clip_id", "bin_clip", "bin_start"):
    same(k, getattr(lib, k), getattr(ref, k))
allf = torch.arange(ref.num_frames)
for k in ("root_p", "q", "root_v", "root_w", "qd"):
    same(k, getattr(lib, k), getattr(ref, k), 1e-5)
for k in ("root_R", "root_v_local", "root_w_local", "root_gz", "frame_bin", "body_p", "body_R", "body_v", "body_w"):
    same(k, getattr(lib, k)[allf], getattr(ref, k), 1e-4)
same("root_R.cpu()", lib.root_R.cpu(), ref.root_R, 1e-6)
for k in ("root_R", "root_v_local", "root_w_local", "root_gz", "frame_bin", "body_p", "body_R", "body_v", "body_w"):
    same(f"stored {k}", getattr(small, k), getattr(lib, k)[allf])  # stored and computed per look-up: the same numbers
    assert getattr(small, k).dtype == getattr(ref, k).dtype and getattr(small, k).shape == getattr(ref, k).shape, k

g = torch.Generator().manual_seed(0)
f = torch.randint(0, ref.num_frames, (4096,), generator=g)
f = torch.cat((f, ref.clip_start, ref.clip_end, ref.clip_end - 1))  # clip boundaries too
B = len(f)
p = ref.root_p[f] + 0.1 * torch.randn(B, 3, generator=g)
R = so3_exp(0.3 * torch.randn(B, 3, generator=g)) @ ref.root_R[f]
for i, (a, b) in enumerate(zip(lib.goal(f, p, R), ref.goal(f, p, R))):
    same(f"goal()[{i}]", a, b, 1e-5)
same("goal_official()", lib.goal_official(f, R), ref.goal_official(f, R), 1e-5)
same("future_frames()", lib.future_frames(f, 8), ref.future_frames(f, 8))
for i, (a, b) in enumerate(zip(lib.ref_state(f), ref.ref_state(f))):
    same(f"ref_state()[{i}]", a, b, 1e-5)

# the analytic heads of mdp.py on a perturbed state
dq = torch.tensor(default_joint_pos_list())
S = (p, R, ref.root_v[f] + 0.2 * torch.randn(B, 3, generator=g), ref.root_w[f] + 0.2 * torch.randn(B, 3, generator=g),
     ref.q[f] + 0.05 * torch.randn(B, 29, generator=g), ref.qd[f] + 0.5 * torch.randn(B, 29, generator=g))  # fmt: skip
a, a_prev = torch.randn(B, 29, generator=g), torch.randn(B, 29, generator=g)
m_ref, m_lib = MDP(kin, ref, dq), MDP(kin, lib, dq)
r0, t0 = m_ref.reward(S, f, a, a_prev, terms=True)
r1, t1 = m_lib.reward(S, f, a, a_prev, terms=True)
same("mdp.reward", r1, r0, 1e-5)
same("mdp.reward (stored)", MDP(kin, small, dq).reward(S, f, a, a_prev), r1)
for k in t0:
    same(f"mdp.reward[{k}]", t1[k], t0[k], 1e-4)
same("mdp.termination", m_lib.termination(S, f), m_ref.termination(S, f))
same("mdp.soft_termination", m_lib.soft_termination(S, f), m_ref.soft_termination(S, f), 1e-6)
xi_n, gw = torch.zeros(B, 33), ref.goal(f, p, R)[1]
same("mdp.critic_features", m_lib.critic_features(S, f, xi_n, gw), m_ref.critic_features(S, f, xi_n, gw), 1e-5)

# manifest: only the listed clips are loaded, in file order
keep = ref.names[::2]
open(os.path.join(out, MANIFEST), "w").write("\n".join(reversed(keep)) + "\n")
part = MotionLib.from_dir(out, kin, "cpu")
assert part.names == keep, (part.names, keep)
for c, name in enumerate(keep):
    cr = ref.names.index(name)
    fr = torch.arange(int(ref.clip_start[cr]), int(ref.clip_end[cr]) + 1)
    fp = torch.arange(int(part.clip_start[c]), int(part.clip_end[c]) + 1)
    assert len(fr) == len(fp)
    for k in ("root_p", "q", "root_v", "root_w", "qd"):
        assert (getattr(part, k)[fp] - getattr(ref, k)[fr]).abs().max() <= 1e-5, (name, k)
    assert (part.body_p[fp] - ref.body_p[fr]).abs().max() <= 1e-4
assert bool((part.frame_bin[part.bin_start] == torch.arange(part.num_bins)).all())
print(f"  manifest: {len(part.names)} of {len(ref.names)} clips loaded, values match")

# bins hold start frames only: a clip of 50 k + 1 frames has k bins (MotionLib: k + 1, the last one a lone final frame)
odd = os.path.join(out, "odd_lengths")
w = CacheWriter(odd, {"source": "test"})
x0, lengths = lafan1_frames(files[0]), [51, 2, 101, 50, 75, 151]
for k, n in enumerate(lengths):
    w.add(f"clip{k}", x0[100 * k : 100 * k + n])
w.close()
o = CachedMotionLib(odd, kin, "cpu", materialize=False)
assert o.num_bins == sum(max(-(-(n - 1) // 50), 1) for n in lengths) == 10, o.num_bins
assert bool((o.bin_start < o.clip_end[o.bin_clip]).all())  # at least one step can be taken from every bin
fb = o.frame_bin[torch.arange(o.num_frames)]
assert bool((o.bin_clip[fb] == o.clip_id).all()) and bool((o.frame_bin[o.bin_start] == torch.arange(o.num_bins)).all())
shutil.rmtree(odd)
print(f"  bins of clips with lengths {lengths}: {o.num_bins}, none holds only a clip's last frame")

# evaluation subset: fixed by (names, n, seed), independent of the order of the names
n_sub = max(len(keep) // 2, 1)
s0, s1 = choose_clips(keep, n_sub, 0), choose_clips(list(reversed(keep)), n_sub, 0)
assert sorted(s0) == sorted(s1) and len(s0) == n_sub and choose_clips(keep, 0, 0) == keep
sub = load_subset(out, kin, "cpu", n_sub, 0)
assert sub.names == s0 and type(sub) is CachedMotionLib
sub_csv = load_subset(os.path.dirname(files[0]), kin, "cpu", 2, 0)
assert type(sub_csv) is MotionLib and len(sub_csv.names) == 2
print(f"  subset: {sub.names} (cache), {sub_csv.names} (csv)")
if not args.out:
    shutil.rmtree(out)
else:
    os.remove(os.path.join(out, MANIFEST))  # the cache that is kept holds every clip
print(f"PASS: CachedMotionLib matches MotionLib on {len(ref.names)} clips, {ref.num_frames} frames (largest difference {worst:.2e})")
