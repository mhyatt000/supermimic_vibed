"""Size and speed of supermimic.motion_cache.CachedMotionLib at a given scale, on a synthetic cache.

``--make`` writes a cache of ``--clips`` clips and ``--hours`` hours whose clips are slices of the clips of ``--src``
(a real cache, e.g. the LAFAN1 one of tools/test_motion_cache.py --out), with a long-tailed length distribution.
The benchmark then loads it on ``--device`` and reports the stored bytes, the start-up time, the peak memory during
loading and the cost of the per-step look-ups at ``--batch`` frames.

  tools/uv_run.sh tools/bench_motion_cache.py --src logs/bones_test/lafan_cache --out logs/bones_test/synth_full --make \
      --clips 129514 --hours 264.56 --device cpu
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=str, required=True, help="the synthetic cache")
ap.add_argument("--src", type=str, default=None, help="--make: cache to take the motion from")
ap.add_argument("--make", action="store_true")
ap.add_argument("--clips", type=int, default=129514)
ap.add_argument("--hours", type=float, default=264.56)
ap.add_argument("--device", type=str, default="cpu", help="cpu, cuda:0, or none (only --make)")
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

import torch  # noqa: E402

from supermimic.kinematics import G1Kinematics  # noqa: E402
from supermimic.motion_cache import CacheWriter, CachedMotionLib, is_cache, open_frames, read_clips  # noqa: E402

if args.make and not is_cache(args.out):
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    _, s_start, s_len = read_clips(args.src)
    src = np.array(open_frames(args.src))
    mean = args.hours * 3600 * 50 / args.clips
    L = rng.lognormal(0.0, 0.7, args.clips)  # long tail: most clips a few seconds, some a minute
    L = np.clip(np.round(L / L.mean() * mean), 10, s_len.max()).astype(np.int64)
    w = CacheWriter(args.out, {"source": f"synthetic: slices of {args.src}", "synthetic": True})
    for k in range(args.clips):
        c = rng.choice(np.flatnonzero(s_len >= L[k]))
        a = int(s_start[c] + rng.integers(0, s_len[c] - L[k] + 1))
        w.add(f"synth_{k:06d}", src[a : a + L[k]], src_dir="synthetic")
    w.close()
    print(f"[bench] wrote {args.out}: {args.clips} clips, {w.n} frames ({w.n / 50 / 3600:.2f} h), lengths {L.min()}..{L.max()} (median {int(np.median(L))}), "
          f"{os.path.getsize(os.path.join(args.out, 'frames.f32')) / 2**30:.2f} GiB, {time.time() - t0:.0f} s", flush=True)  # fmt: skip
if args.device == "none":
    sys.exit(0)

dev = args.device
cuda = dev.startswith("cuda")
kin = G1Kinematics(dev)
if cuda:
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
t0 = time.time()
lib = CachedMotionLib(args.out, kin, dev, materialize=False)  # the large-library path at any size
load_s = time.time() - t0
stored = sum(t.numel() * t.element_size() for t in vars(lib).values() if torch.is_tensor(t))
print(f"[bench] {len(lib.names)} clips, {lib.num_frames} frames, {lib.num_bins} bins; all tensors of the library: {stored / 2**30:.2f} GiB "
      f"({stored / lib.num_frames:.1f} bytes per frame); loaded in {load_s:.1f} s")  # fmt: skip
if cuda:
    print(f"[bench] GPU: allocated {(torch.cuda.memory_allocated() - base) / 2**30:.2f} GiB after loading, peak {(torch.cuda.max_memory_allocated() - base) / 2**30:.2f} GiB "
          f"during loading, reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB")  # fmt: skip
else:
    print(f"[bench] CPU: peak resident memory of the process {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GiB")


def clock(fn, n=20):
    fn()
    if cuda:
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.time() - t) / n * 1e3


B = args.batch
f = torch.randint(0, lib.num_frames - 1, (B,), device=dev)
p, R = lib.root_p[f], lib.root_R[f]
bin_G = torch.rand(lib.num_bins, device=dev)
w_avg = torch.full((lib.num_bins,), 1.0 / lib.num_bins, device=dev)


def sample():  # TrackingSim.sample_frames
    soft = torch.softmax(-25.0 * bin_G, 0)
    w_avg.mul_(0.999).add_(0.001 * soft)
    probs = 0.5 / lib.num_bins + 0.4 * soft + 0.1 * w_avg / w_avg.sum()
    return lib.bin_start[torch.multinomial(probs, B, replacement=True)]


with torch.no_grad():
    print(f"[bench] per call at {B} frames on {dev}: reference bodies (FK) {clock(lambda: lib.body_state(f)):.2f} ms, goal {clock(lambda: lib.goal(f, p, R)):.2f} ms, "
          f"ref_state {clock(lambda: lib.ref_state(f)):.2f} ms, reset sampling over {lib.num_bins} bins {clock(sample):.2f} ms")  # fmt: skip
