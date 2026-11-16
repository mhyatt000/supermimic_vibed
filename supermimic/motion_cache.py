"""Reference motion library for datasets too large for :class:`MotionLib` (BONES-SEED: 47.6 M frames at 50 Hz).

:class:`MotionLib` keeps 1356 bytes per frame on the GPU (the tracked-body poses and velocities are 1008 of them),
64.6 GB for the filtered BONES-SEED set. :class:`CachedMotionLib` keeps the generalized coordinates and velocities
only (292 bytes per frame, 13.9 GB) and computes the rest when a batch of frames is indexed: the root rotation matrix
from the stored quaternion, the root-frame velocities, and the tracked bodies by one forward-kinematics pass (the
:meth:`G1Kinematics.forward` that MotionLib runs once at start-up on the same inputs, so the values agree).
It is a drop-in for MotionLib: same attributes (the derived ones are indexable objects instead of tensors) and methods.
A library of at most ``MATERIALIZE_FRAMES`` frames (an evaluation subset, a test cache) stores the derived quantities
as MotionLib does: the look-ups then cost what they cost there, which matters where they are not compiled (evaluate.py).

It reads a cache directory written once by ``tools/bones_seed_convert.py``:
  motion_cache.json   marker and description; ``MotionLib.from_dir`` selects this class when it exists
  frames.f32          float32 (num_frames, 36), 50 Hz: root position (m), root quaternion wxyz, 29 joint angles (rad)
  clips.csv           name, dir, start, length (rows of frames.f32), src_frames; in file order
  manifest.txt        optional, written by tools/bones_seed_filter.py: the clips to load, one name per line
"""

from __future__ import annotations

import csv
import glob
import json
import os
import time

import numpy as np
import torch

from .kinematics import G1Kinematics
from .motion import BIN_FRAMES, CONTROL_DT, MotionLib
from .rotations import quat_to_mat, so3_log

MARKER, FRAMES, CLIPS, MANIFEST = "motion_cache.json", "frames.f32", "clips.csv", "manifest.txt"
FRAME_DIM = 36
_CHUNK = 1 << 20  # frames per block when loading or materializing
# libraries up to this size also store the derived quantities (1356 instead of 292 bytes per frame: 2.5 GiB at the limit)
MATERIALIZE_FRAMES = int(os.environ.get("SM_MOTION_MATERIALIZE_FRAMES", 2_000_000))


def is_cache(path: str) -> bool:
    return os.path.isfile(os.path.join(path, MARKER))


def read_clips(path: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """``(names, start, length)`` of every clip in the cache, in file order."""
    with open(os.path.join(path, CLIPS), newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r["name"] for r in rows], np.array([int(r["start"]) for r in rows], np.int64), np.array([int(r["length"]) for r in rows], np.int64)


def read_manifest(path: str) -> list[str] | None:
    p = os.path.join(path, MANIFEST)
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return [l.strip() for l in fh if l.strip()]


def open_frames(path: str) -> np.memmap:
    n = os.path.getsize(os.path.join(path, FRAMES)) // (4 * FRAME_DIM)
    return np.memmap(os.path.join(path, FRAMES), dtype="<f4", mode="r", shape=(n, FRAME_DIM))


def choose_clips(names: list[str], n: int, seed: int) -> list[str]:
    """A fixed random subset of ``n`` names: it depends on the set of names, ``n`` and ``seed`` only (not on their order)."""
    if n <= 0 or n >= len(names):
        return list(names)
    pool = sorted(names)
    chosen = {pool[i] for i in np.random.default_rng(seed).choice(len(pool), n, replace=False)}
    return [x for x in names if x in chosen]


class CacheWriter:
    """Writes a cache directory clip by clip. The marker is written last, so a partial cache is never loaded."""

    def __init__(self, path: str, meta: dict):
        os.makedirs(path, exist_ok=True)
        self.path, self.meta, self.rows, self.n = path, dict(meta), [], 0
        if os.path.exists(os.path.join(path, MARKER)):
            os.remove(os.path.join(path, MARKER))
        self.fh = open(os.path.join(path, FRAMES + ".tmp"), "wb")

    def add(self, name: str, frames: np.ndarray, src_dir: str = "", src_frames: int = 0):
        frames = np.ascontiguousarray(frames, dtype="<f4")
        assert frames.ndim == 2 and frames.shape[1] == FRAME_DIM and len(frames) >= 2, (name, frames.shape)
        self.fh.write(frames.tobytes())
        self.rows.append((name, src_dir, self.n, len(frames), src_frames))
        self.n += len(frames)

    def close(self):
        self.fh.close()
        names = [r[0] for r in self.rows]
        if len(set(names)) < len(names):  # the same motion name under two directories: qualify all names
            self.rows = [(f"{r[1]}/{r[0]}", *r[1:]) for r in self.rows]
            assert len({r[0] for r in self.rows}) == len(self.rows), "clip names are not unique"
            self.meta["names"] = "dir/name"
        with open(os.path.join(self.path, CLIPS + ".tmp"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(("name", "dir", "start", "length", "src_frames"))
            w.writerows(self.rows)
        os.replace(os.path.join(self.path, FRAMES + ".tmp"), os.path.join(self.path, FRAMES))
        os.replace(os.path.join(self.path, CLIPS + ".tmp"), os.path.join(self.path, CLIPS))
        self.meta.update(fps=1 / CONTROL_DT, layout="root_pos_m(3) root_quat_wxyz(4) joint_pos_rad(29)", dtype="<f4",
                         num_clips=len(self.rows), num_frames=self.n, hours=self.n * CONTROL_DT / 3600)  # fmt: skip
        with open(os.path.join(self.path, MARKER + ".tmp"), "w") as fh:
            json.dump(self.meta, fh, indent=1)
        os.replace(os.path.join(self.path, MARKER + ".tmp"), os.path.join(self.path, MARKER))


class _Lazy:
    """A per-frame reference quantity computed when indexed with a tensor of frames, instead of being stored."""

    def __init__(self, lib: CachedMotionLib, fn):
        self._lib, self._fn = lib, fn

    def __getitem__(self, f: torch.Tensor) -> torch.Tensor:
        return self._fn(self._lib, f)

    def cpu(self) -> torch.Tensor:
        """All frames, materialized on the CPU (evaluate.py --record; small libraries only)."""
        lib = self._lib
        blocks = [self._fn(lib, torch.arange(s, min(s + _CHUNK, lib.num_frames), device=lib.device)).cpu() for s in range(0, lib.num_frames, _CHUNK)]
        return torch.cat(blocks)


def _root_R(lib, f):
    return quat_to_mat(lib.root_quat[f])


def _root_v_local(lib, f):
    return (quat_to_mat(lib.root_quat[f]).transpose(-1, -2) @ lib.root_v[f].unsqueeze(-1)).squeeze(-1)


def _root_w_local(lib, f):
    return (quat_to_mat(lib.root_quat[f]).transpose(-1, -2) @ lib.root_w[f].unsqueeze(-1)).squeeze(-1)


def _root_gz(lib, f):
    return quat_to_mat(lib.root_quat[f])[..., 2, 2]


def _frame_bin(lib, f):
    c = lib.clip_id[f]
    return lib.bin_first[c] + torch.minimum((f - lib.clip_start[c]) // BIN_FRAMES, lib.bin_count[c] - 1)


class CachedMotionLib(MotionLib):
    def __init__(self, path: str, kin: G1Kinematics, device: str, clips: list[str] | None = None, materialize: bool | None = None):
        """``clips``: names to load; default: those of ``manifest.txt`` if it exists, else every clip of the cache.
        ``materialize``: store the derived quantities instead of computing them per look-up; default: by size."""
        t0 = time.time()
        self.device, self.kin, self.path, self._bodies = device, kin, path, None
        names, start, length = read_clips(path)
        manifest = read_manifest(path)
        want = clips if clips is not None else manifest
        if want is None:
            sel = np.arange(len(names))
            print(f"[motion] {path}: no {MANIFEST} (run tools/bones_seed_filter.py), loading all {len(names)} clips unfiltered", flush=True)
        else:
            want = set(want)
            sel = np.array([i for i, n in enumerate(names) if n in want], np.int64)
            if len(sel) != len(want):
                missing = sorted(want - set(names))
                raise RuntimeError(f"{len(missing)} requested clips are not in {path}/{CLIPS}, e.g. {missing[:3]}")
        self.names = [names[i] for i in sel]
        lens = torch.from_numpy(length[sel])
        C, N = len(sel), int(lens.sum())
        self.num_frames = N
        self.clip_len = lens.to(device)
        self.clip_start = (torch.cumsum(lens, 0) - lens).to(device)
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.clip_id = torch.repeat_interleave(torch.arange(C, device=device), self.clip_len)

        f32 = lambda d: torch.empty((N, d), dtype=torch.float32, device=device)  # noqa: E731
        self.root_p, self.root_quat, self.root_v, self.root_w, self.q, self.qd = f32(3), f32(4), f32(3), f32(3), f32(29), f32(29)
        mm = open_frames(path)
        src_start, src_len, out = start[sel], length[sel], 0
        cum = np.cumsum(src_len)
        i = 0
        while i < C:  # blocks of whole clips, about _CHUNK frames each
            j = max(int(np.searchsorted(cum, (cum[i - 1] if i else 0) + _CHUNK, side="right")), i + 1)
            a, b = int(src_start[i]), int(src_start[j - 1] + src_len[j - 1])
            n = int(src_len[i:j].sum())
            if b - a == n:  # the clips are adjacent in the file
                x = np.array(mm[a:b])
            else:
                rows = np.concatenate([np.arange(s, s + l) for s, l in zip(src_start[i:j], src_len[i:j])])
                x = np.array(mm[a:b])[rows - a] if b - a < 4 * n else np.array(mm[rows])
            self._fill(torch.from_numpy(x).to(device), torch.from_numpy(src_len[i:j]).to(device), out)
            out += n
            i = j
        del mm

        # 1 s bins of start frames, consecutive per clip as in MotionLib. A clip's last frame is not a start frame
        # (no step can be taken from it), so a clip of 50 k + 1 frames has k bins where MotionLib has k + 1, the last
        # holding only that frame: an episode reset there would be scored against the next clip's first frame, and
        # among 130,000 clips the return-based sampler would keep returning to those bins.
        nb = torch.clamp((self.clip_len - 1 + BIN_FRAMES - 1) // BIN_FRAMES, min=1)
        self.bin_count = nb
        self.num_bins = int(nb.sum())
        self.bin_first = torch.cumsum(nb, 0) - nb  # first bin of each clip
        self.bin_clip = torch.repeat_interleave(torch.arange(C, device=device), nb)
        self.bin_start = self.clip_start[self.bin_clip] + (torch.arange(self.num_bins, device=device) - self.bin_first[self.bin_clip]) * BIN_FRAMES

        self.root_R, self.root_gz, self.frame_bin = _Lazy(self, _root_R), _Lazy(self, _root_gz), _Lazy(self, _frame_bin)
        self.root_v_local, self.root_w_local = _Lazy(self, _root_v_local), _Lazy(self, _root_w_local)
        self.body_p, self.body_R = _Lazy(self, lambda lib, f: lib.body_state(f)[0]), _Lazy(self, lambda lib, f: lib.body_state(f)[1])
        self.body_v, self.body_w = _Lazy(self, lambda lib, f: lib.body_state(f)[2]), _Lazy(self, lambda lib, f: lib.body_state(f)[3])
        self.materialized = N <= MATERIALIZE_FRAMES if materialize is None else bool(materialize)
        if self.materialized:
            blocks = [torch.arange(s, min(s + _CHUNK, N), device=device) for s in range(0, N, _CHUNK)]
            parts = [self.body_state(f) for f in blocks]
            self._bodies = tuple(torch.cat([p[i] for p in parts]) for i in range(4))
            self.body_p, self.body_R, self.body_v, self.body_w = self._bodies
            for k in ("root_R", "root_gz", "root_v_local", "root_w_local", "frame_bin"):
                setattr(self, k, torch.cat([getattr(self, k)[f] for f in blocks]))
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        stored = sum(t.numel() * t.element_size() for t in vars(self).values() if torch.is_tensor(t))
        print(f"[motion] {path}: {C} clips, {N} frames ({N * CONTROL_DT / 3600:.2f} h), {self.num_bins} bins, {stored / 2**30:.2f} GiB on {device} "
              f"({'derived quantities stored' if self.materialized else 'derived quantities computed per look-up'}), loaded in {time.time() - t0:.1f} s", flush=True)  # fmt: skip

    def _fill(self, x: torch.Tensor, lens: torch.Tensor, out: int):
        """Store a block of whole clips ``x`` (n, 36) at frames ``out..out+n`` with MotionLib's finite-difference
        velocities: central inside a clip, one-sided at its first and last frame."""
        n = x.shape[0]
        first = torch.cumsum(lens, 0) - lens
        cid = torch.repeat_interleave(torch.arange(len(lens), device=x.device), lens)
        f = torch.arange(n, device=x.device)
        lo, hi = torch.maximum(f - 1, first[cid]), torch.minimum(f + 1, (first + lens - 1)[cid])
        den = torch.where(hi - lo == 2, 2 * CONTROL_DT, CONTROL_DT).unsqueeze(-1)
        p, quat, q = x[:, :3], x[:, 3:7], x[:, 7:]
        R = quat_to_mat(quat)
        s = slice(out, out + n)
        self.root_p[s], self.root_quat[s], self.q[s] = p, quat, q
        self.root_v[s] = (p[hi] - p[lo]) / den
        self.qd[s] = (q[hi] - q[lo]) / den
        self.root_w[s] = so3_log(R[hi] @ R[lo].transpose(-1, -2)) / den

    def body_state(self, f: torch.Tensor):
        """Tracked-body positions, rotations, linear and angular velocities of reference frames ``f`` (one FK pass)."""
        if self._bodies is not None:
            return tuple(t[f] for t in self._bodies)
        return self.kin.forward(self.root_p[f], quat_to_mat(self.root_quat[f]), self.root_v[f], self.root_w[f], self.q[f], self.qd[f])


def load_subset(path: str, kin: G1Kinematics, device: str, num_clips: int, seed: int) -> MotionLib:
    """A library of a fixed random subset of ``num_clips`` clips of ``path`` (a cache or a directory of LAFAN1 CSVs):
    the checkpoint-evaluation subset. Only the chosen clips are read and put on the device."""
    if is_cache(path):
        manifest = read_manifest(path)
        names = manifest if manifest is not None else read_clips(path)[0]
        return CachedMotionLib(path, kin, device, clips=choose_clips(names, num_clips, seed))
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]  # noqa: E731
    chosen = set(choose_clips([stem(f) for f in files], num_clips, seed))
    return MotionLib([f for f in files if stem(f) in chosen], kin, device)
