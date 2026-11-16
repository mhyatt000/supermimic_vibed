"""Convert the G1 retarget of BONES-SEED to the 50 Hz cache read by supermimic/motion_cache.py.

Source: ``g1.tar.gz`` of https://huggingface.co/datasets/bones-studio/seed (tools/download_bones_seed.sh), read as a
stream and never extracted (142,220 CSVs ``g1/csv/<date>/<motion>.csv``); a directory of such CSVs works too.

Source format, taken from two official readers and checked per file by column NAME, never by position
(NVlabs/GR00T-WholeBodyControl gear_sonic/scripts/convert_soma_csv_to_motion_lib.py::load_bones_csv, and the
bones-studio/seed-viewer documentation): a header row, then one row per frame at 120 fps with
  Frame                       frame index
  root_translateX/Y/Z         pelvis position in centimetres, Z up
  root_rotateX/Y/Z            pelvis orientation, Euler angles in degrees, R = Rz(Z) Ry(Y) Rx(X)
                              (scipy ``from_euler("xyz")``, the three.js order 'ZYX')
  <joint name>_dof            29 joint angles in degrees, named as supermimic.robot.G1_JOINT_NAMES
Conversion: centimetres to metres, degrees to radians, Euler angles to a wxyz quaternion, then resampling to 50 Hz
with MotionLib's interpolation (positions and joints linear, orientation slerp, last partial step dropped).
tools/bones_seed_verify.py checks these conventions on the converted data; for it, the unconverted root and joint
columns of every ``--raw_sample_every``-th file are kept in ``raw_sample.npz``.

  tools/uv_run.sh tools/bones_seed_convert.py --src data/bones_seed/g1.tar.gz --out data/bones_seed/cache
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import sys
import tarfile
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--src", type=str, default=os.path.join(REPO, "data", "bones_seed", "g1.tar.gz"), help="tar archive (.tar, .tar.gz) or directory of CSVs")
ap.add_argument("--out", type=str, default=os.path.join(REPO, "data", "bones_seed", "cache"))
ap.add_argument("--src_fps", type=float, default=120.0)
ap.add_argument("--euler", type=str, default="xyz", help="scipy Euler sequence of root_rotateX/Y/Z (lower case: extrinsic)")
ap.add_argument("--pos_scale", type=float, default=0.01, help="root_translate units to metres")
ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 2, 1))
ap.add_argument("--limit", type=int, default=0, help="stop after this many clips (tests)")
ap.add_argument("--raw_sample_every", type=int, default=300, help="keep the source values of every n-th file for tools/bones_seed_verify.py (0: none)")
OUT_FPS = 50.0
ROOT_T = ["root_translateX", "root_translateY", "root_translateZ"]
ROOT_R = ["root_rotateX", "root_rotateY", "root_rotateZ"]


def convert(job):
    """One CSV (bytes) -> ``(dir, name, frames (n,36) float32 | None, src_frames, error, header, raw | None)``;
    ``raw``: the source rows (root_translate XYZ, root_rotate XYZ, 29 joints; source units and rate) if asked for."""
    import pandas as pd
    from scipy.spatial.transform import Rotation

    from supermimic.robot import G1_JOINT_NAMES

    path, data, src_fps, euler, pos_scale, keep_raw = job
    d, name = os.path.basename(os.path.dirname(path)), os.path.splitext(os.path.basename(path))[0]
    try:
        df = pd.read_csv(io.BytesIO(data))
        df.columns = [c.strip() for c in df.columns]
        header = list(df.columns)
        cols = ROOT_T + ROOT_R + [f"{j}_dof" for j in G1_JOINT_NAMES]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return d, name, None, len(df), f"missing columns: {missing[:4]}", header, None
        x = df[cols].to_numpy(np.float64)
        T = len(x)
        if T < 2 or not np.isfinite(x).all():
            return d, name, None, T, "fewer than 2 frames" if T < 2 else "non-finite values", header, None
        p, j = x[:, :3] * pos_scale, np.deg2rad(x[:, 6:])
        quat = Rotation.from_euler(euler, x[:, 3:6], degrees=True).as_quat()[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        # resample: frame n of the output is source time n / OUT_FPS (MotionLib's rule)
        n = int(np.floor((T - 1) * OUT_FPS / src_fps + 1e-9)) + 1
        if n < 2:
            return d, name, None, T, "shorter than two 50 Hz frames", header, None
        t = np.arange(n) * (src_fps / OUT_FPS)
        i0 = np.minimum(np.floor(t + 1e-9).astype(np.int64), T - 2)
        a = np.clip(t - i0, 0.0, 1.0)[:, None]
        q0, q1 = quat[i0], quat[i0 + 1]
        dot = (q0 * q1).sum(-1, keepdims=True)
        q1, dot = np.where(dot < 0, -q1, q1), np.minimum(np.abs(dot), 1.0)
        th = np.arccos(dot)
        s = np.sin(th)
        ok = s > 1e-6
        w0 = np.where(ok, np.sin((1 - a) * th) / np.where(ok, s, 1.0), 1 - a)
        w1 = np.where(ok, np.sin(a * th) / np.where(ok, s, 1.0), a)
        qi = w0 * q0 + w1 * q1
        qi /= np.linalg.norm(qi, axis=-1, keepdims=True)
        out = np.concatenate((p[i0] * (1 - a) + p[i0 + 1] * a, qi, j[i0] * (1 - a) + j[i0 + 1] * a), -1).astype(np.float32)
        return d, name, out, T, None, header, x.astype(np.float32) if keep_raw else None
    except Exception as e:  # noqa: BLE001  (one bad file must not stop the conversion)
        return d, name, None, 0, repr(e)[:200], None, None


def members(src: str, limit: int):
    """``(path, bytes)`` of every CSV, in archive order (or sorted, for a directory)."""
    n = 0
    if os.path.isdir(src):
        for root, _, files in sorted(os.walk(src)):
            for f in sorted(files):
                if f.endswith(".csv"):
                    yield os.path.join(root, f), open(os.path.join(root, f), "rb").read()
                    n += 1
                    if limit and n >= limit:
                        return
        return
    with tarfile.open(src, "r|*") as tar:  # a stream: nothing is written to disk
        for m in tar:
            if m.isfile() and m.name.endswith(".csv"):
                yield m.name, tar.extractfile(m).read()
                n += 1
                if limit and n >= limit:
                    return


if __name__ == "__main__":
    args = ap.parse_args()
    from supermimic.motion_cache import CacheWriter
    from supermimic.robot import G1_JOINT_NAMES

    t0 = time.time()
    meta = {"source": os.path.abspath(args.src), "source_bytes": os.path.getsize(args.src) if os.path.isfile(args.src) else None,
            "dataset": "BONES-SEED G1 retarget (https://huggingface.co/datasets/bones-studio/seed), Motion Data by Bones Studio",
            "src_fps": args.src_fps, "euler": args.euler, "pos_scale": args.pos_scale, "joint_names": G1_JOINT_NAMES,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
    writer = CacheWriter(args.out, meta)
    gate = threading.BoundedSemaphore(8 * args.workers)  # bounds the CSVs held in memory ahead of the workers

    def jobs():
        for k, (path, data) in enumerate(members(args.src, args.limit)):
            gate.acquire()
            yield path, data, args.src_fps, args.euler, args.pos_scale, bool(args.raw_sample_every) and k % args.raw_sample_every == 0

    errors, header0, n_src, n_hdr, raws = [], None, 0, 0, []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, (d, name, frames, T, err, header, raw) in enumerate(pool.imap(convert, jobs(), chunksize=4)):
            gate.release()
            if header is not None:
                header0 = header0 or header
                n_hdr += header != header0
            if err:
                errors.append((d, name, T, err))
            else:
                writer.add(name, frames, src_dir=d, src_frames=T)
                n_src += T
                if raw is not None:
                    raws.append((f"{d}/{name}", raw))
            if (k + 1) % 5000 == 0:
                print(f"[convert] {k + 1} files, {writer.n} frames ({writer.n / OUT_FPS / 3600:.1f} h), {len(errors)} errors, {time.time() - t0:.0f} s", flush=True)
    writer.meta.update(header=header0, files_with_other_header=n_hdr, num_errors=len(errors), src_frames=n_src,
                       src_hours=n_src / args.src_fps / 3600, convert_seconds=round(time.time() - t0))  # fmt: skip
    writer.close()
    if raws:
        np.savez_compressed(os.path.join(args.out, "raw_sample.npz"), names=np.array([n for n, _ in raws]), lengths=np.array([len(r) for _, r in raws]),
                            data=np.concatenate([r for _, r in raws]), columns=np.array(ROOT_T + ROOT_R + [f"{j}_dof" for j in G1_JOINT_NAMES]),
                            src_fps=args.src_fps)  # fmt: skip
    with open(os.path.join(args.out, "convert_errors.csv"), "w") as fh:
        fh.write("dir,name,src_frames,error\n" + "".join(f"{d},{n},{T},{e}\n" for d, n, T, e in errors))
    print(f"[convert] done: {len(writer.rows)} clips, {writer.n} frames at 50 Hz ({writer.n / OUT_FPS / 3600:.2f} h; source {n_src / args.src_fps / 3600:.2f} h), "
          f"{len(errors)} errors (convert_errors.csv), {n_hdr} files with a different header, {os.path.getsize(os.path.join(args.out, 'frames.f32')) / 2**30:.2f} GiB, "
          f"{time.time() - t0:.0f} s -> {args.out}", flush=True)  # fmt: skip
    print(f"[convert] header: {header0}", flush=True)
