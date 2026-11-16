"""Evidence that tools/bones_seed_convert.py read the BONES-SEED G1 files correctly. Run after the filter.

Nothing here trusts the assumed conventions; each check asks the data which convention makes physical sense:
  columns    the header found in the files: every expected column present by name (joint order is never positional)
  up axis    which root_translate axis is the height (small spread, about 0.8 m once scaled)
  angle unit share of joint samples inside the robot's joint limits if the values are degrees, and if radians
  Euler      for each of the six orders in which root_rotateX/Y/Z can be composed: on frames where the root is tilted,
             how often the lowest collision geometry of the robot touches the floor (within 3 cm). Only the right
             order puts a tilted body on the ground.
  frame rate vertical acceleration of the centre of mass while both feet are in the air. At a wrong frame rate it is
             off by the square of the ratio. A coarse check: a retarget to a 1.3 m robot is not ballistic (the LAFAN1
             G1 retarget, documented as 30 fps, reads -4.2 m/s^2, not -9.81), so it only tells a rate wrong by a
             factor of two (four times, or a quarter of, the value) from a right one. The finer evidence is the total
             duration: tools/bones_seed_filter.py prints the hours kept next to the paper's 264.56 h.
  stance     foot clearance and sole tilt while standing still (joint zero offsets and signs of the legs)
  posture    share of frames with the pelvis upright, median pelvis height
``--render N`` also writes ``verify_reference.mp4`` (N kept clips played back kinematically) for a look by eye.

  tools/uv_run.sh tools/bones_seed_verify.py --cache data/bones_seed/cache --render 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "tools")]

ap = argparse.ArgumentParser()
ap.add_argument("--cache", type=str, default=os.path.join(REPO, "data", "bones_seed", "cache"))
ap.add_argument("--frames", type=int, default=20000, help="frames sampled per check")
ap.add_argument("--render", type=int, default=0, help="render this many kept clips to verify_reference.mp4")
ap.add_argument("--render_seconds", type=float, default=8.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from bones_seed_filter import DT, FEAS, _sim  # noqa: E402
from supermimic.motion_cache import MARKER, open_frames, read_clips, read_manifest  # noqa: E402
from supermimic.robot import G1_JOINT_NAMES  # noqa: E402

rng = np.random.default_rng(args.seed)
meta = json.load(open(os.path.join(args.cache, MARKER)))
names, start, length = read_clips(args.cache)
mm = open_frames(args.cache)
m, d, floor, feet = _sim()
collision = [g for g in range(m.ngeom) if g != floor and m.geom_contype[g] != 0]
report, verdicts = {}, []


def verdict(name: str, ok: bool | None, text: str):
    ok = None if ok is None else bool(ok)
    verdicts.append((name, ok))
    print(f"[verify] {'PASS ' if ok else 'CHECK' if ok is not None else 'SKIP '} {name}: {text}", flush=True)


def lowest(qpos: np.ndarray) -> np.ndarray:
    """Clearance of the lowest collision geometry above the floor, per row of qpos."""
    out = np.empty(len(qpos))
    for t, x in enumerate(qpos):
        d.qpos[:] = x
        mujoco.mj_kinematics(m, d)
        out[t] = min(mujoco.mj_geomDistance(m, d, g, floor, 10.0, None) for g in collision)
    return out


# ---------------- columns
cols = [f"root_translate{a}" for a in "XYZ"] + [f"root_rotate{a}" for a in "XYZ"] + [f"{j}_dof" for j in G1_JOINT_NAMES]
header = meta.get("header") or []
missing, extra = [c for c in cols if c not in header], [c for c in header if c not in cols]
report["columns"] = {"header": header, "missing": missing, "not_used": extra, "files_with_other_header": meta.get("files_with_other_header"),
                     "convert_errors": meta.get("num_errors")}  # fmt: skip
verdict("columns", not missing and bool(header), f"{len(header)} columns in the files; all {len(cols)} expected names present: {not missing}; "
        f"not used: {extra}; files whose header differs from the first file's: {meta.get('files_with_other_header')}; unreadable files: {meta.get('num_errors')}")  # fmt: skip

# ---------------- checks on the unconverted sample
raw_path = os.path.join(args.cache, "raw_sample.npz")
if os.path.isfile(raw_path):
    raw = np.load(raw_path)
    X = raw["data"].astype(np.float64)
    scale, euler = meta.get("pos_scale", 0.01), meta.get("euler", "xyz")
    # up axis
    med, spread = np.median(X[:, :3], 0) * scale, (np.percentile(X[:, :3], 90, 0) - np.percentile(X[:, :3], 10, 0)) * scale
    up = int(np.argmin(spread))
    report["up_axis"] = {"median_m": med.tolist(), "p10_p90_spread_m": spread.tolist()}
    verdict("up axis and length unit", up == 2 and 0.5 < med[2] < 1.0, f"root_translate X,Y,Z x {scale}: median {np.round(med, 3).tolist()} m, 10-90 % spread "
            f"{np.round(spread, 3).tolist()} m; Z has the smallest spread and a standing pelvis height (G1: about 0.79 m)")  # fmt: skip
    # angle unit
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    J = X[:, 6:]
    in_deg = float(((np.deg2rad(J) >= lo - 0.035) & (np.deg2rad(J) <= hi + 0.035)).mean())
    in_rad = float(((J >= lo - 0.035) & (J <= hi + 0.035)).mean())
    worst = np.maximum(lo - np.deg2rad(J), np.deg2rad(J) - hi).max(0)
    report["angle_unit"] = {"inside_limits_if_degrees": in_deg, "inside_limits_if_radians": in_rad,
                            "largest_limit_violation_rad": {G1_JOINT_NAMES[k]: round(float(worst[k]), 3) for k in np.argsort(-worst)[:5]}}  # fmt: skip
    verdict("joint angle unit", in_deg > 0.99 and in_deg > in_rad, f"joint samples inside the joint limits (+-2 deg): {100 * in_deg:.2f} % read as degrees, "
            f"{100 * in_rad:.2f} % read as radians; largest violations (rad): {report['angle_unit']['largest_limit_violation_rad']}")  # fmt: skip
    # Euler order: tilted frames, each composition order
    tilted = np.flatnonzero(np.abs(X[:, 3:5]).max(1) > 20.0)
    if len(tilted) >= 500:
        pick = np.sort(rng.choice(tilted, min(args.frames, len(tilted)), replace=False))
        table = {}
        for order in itertools.permutations(range(3)):  # R = R_a R_b R_c
            seq = "".join("xyz"[k] for k in reversed(order))  # scipy extrinsic: first letter is applied first
            quat = Rotation.from_euler(seq, X[pick][:, [3 + k for k in reversed(order)]], degrees=True).as_quat()[:, [3, 0, 1, 2]]
            z = lowest(np.concatenate((X[pick, :3] * scale, quat, np.deg2rad(X[pick, 6:])), 1))
            label = "R" + " R".join("xyz"[k] for k in order)
            table[label] = {"scipy_extrinsic": seq, "on_floor_within_3cm": float((np.abs(z) < 0.03).mean()), "median_abs_clearance_m": float(np.median(np.abs(z)))}
        assumed = "R" + " R".join(reversed(euler)) if euler.islower() else "R" + " R".join(euler.lower())
        best = max(table, key=lambda k: table[k]["on_floor_within_3cm"])
        report["euler"] = {"tilted_frames": int(len(tilted)), "sampled": int(len(pick)), "assumed": assumed, "best": best, "orders": table}
        txt = "; ".join(f"{k} {100 * v['on_floor_within_3cm']:.1f} %" for k, v in sorted(table.items(), key=lambda kv: -kv[1]["on_floor_within_3cm"]))
        verdict("Euler order of the root", best == assumed, f"lowest collision geometry within 3 cm of the floor on {len(pick)} frames with the root tilted over 20 deg: {txt}. "
                f"The converter uses {assumed}")  # fmt: skip
    else:
        verdict("Euler order of the root", None, f"only {len(tilted)} tilted frames in the sample")
else:
    verdict("unconverted sample", None, f"{raw_path} not found (bones_seed_convert.py --raw_sample_every)")

# ---------------- checks on the converted cache (kept clips)
manifest = read_manifest(args.cache)
kept = np.array([n in set(manifest) for n in names]) if manifest is not None else np.ones(len(names), bool)
feas_path = os.path.join(args.cache, FEAS)
if os.path.isfile(feas_path):
    feas = np.memmap(feas_path, dtype=np.float16, mode="r", shape=(len(mm), 3))
    acc, stance_rows, n_up, n_all, heights = [], [], 0, 0, []
    for c in rng.permutation(np.flatnonzero(kept))[:20000]:
        a, b = int(start[c]), int(start[c] + length[c])
        z, x = np.asarray(feas[a:b], np.float32), np.asarray(mm[a:b, :7])
        q = x[:, 3:7]
        upz = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        n_up, n_all = n_up + int((upz > np.cos(np.deg2rad(30))).sum()), n_all + len(x)
        heights.append(x[::10, 2])
        # flight: upright, both feet and the lowest link clear of the floor for 0.2 to 0.8 s (longer is not a jump but
        # a body resting on something); one value per flight: the quadratic fit of the height of the centre of mass
        fl = (z[:, 1] > 0.03) & (z[:, 2] > 0.03) & (z[:, 0] > 0.03) & (upz > np.cos(np.deg2rad(30))) & (x[:, 2] > 0.6)
        edges = np.diff(np.concatenate(([0], fl.astype(np.int8), [0])))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if 10 <= e - s <= 40 and len(acc) < 3000:
                com = np.empty(e - s)
                for k in range(s, e):
                    d.qpos[:] = mm[a + k]
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_comPos(m, d)
                    com[k - s] = d.subtree_com[0, 2]
                acc.append(2 * np.polyfit(np.arange(e - s) * DT, com, 2)[0])
        if len(x) > 2:
            speed = np.linalg.norm(np.gradient(x[:, :3], DT, axis=0), axis=-1)
            st = np.flatnonzero((upz > np.cos(np.deg2rad(10))) & (x[:, 2] > 0.65) & (speed < 0.05) & (np.maximum(z[:, 1], z[:, 2]) < 0.02))
            stance_rows.append(a + st[:: max(len(st) // 5, 1)][:5])
    heights = np.concatenate(heights)
    report["posture"] = {"upright_within_30deg": n_up / max(n_all, 1), "median_pelvis_height_m": float(np.median(heights))}
    verdict("posture", n_up / max(n_all, 1) > 0.7 and 0.6 < np.median(heights) < 0.9, f"pelvis within 30 deg of upright in {100 * n_up / max(n_all, 1):.1f} % of frames, "
            f"median pelvis height {np.median(heights):.3f} m")  # fmt: skip
    if acc:
        acc = np.array(acc)
        g = float(np.median(acc))
        report["free_fall"] = {"flights": int(len(acc)), "median_vertical_acceleration": g, "p25": float(np.percentile(acc, 25)), "p75": float(np.percentile(acc, 75))}
        verdict("frame rate (free fall)", -14.0 < g < -3.0 and len(acc) >= 30, f"vertical acceleration of the centre of mass in {len(acc)} flight phases: median {g:.2f} m/s^2 "
                f"(quartiles {np.percentile(acc, 25):.1f} .. {np.percentile(acc, 75):.1f}); gravity is -9.81 and the LAFAN1 G1 retarget reads -4.2. At a true source rate r "
                f"instead of {meta.get('src_fps')} the value is multiplied by (r/{meta.get('src_fps')})^2, so a rate wrong by a factor of two falls outside -14 .. -3")  # fmt: skip
    else:
        verdict("frame rate (free fall)", None, "no flight phases in the sampled clips")
    rows = np.sort(np.concatenate(stance_rows))[: args.frames] if stance_rows else np.zeros(0, np.int64)
    if len(rows) >= 100:
        clear, tilt = [], []
        ids = [m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        for r in rows:
            d.qpos[:] = mm[r]
            mujoco.mj_kinematics(m, d)
            clear.append([min(mujoco.mj_geomDistance(m, d, g, floor, 10.0, None) for g in f) for f in feet])
            tilt.append([np.degrees(np.arccos(np.clip(d.xmat[i][8], -1, 1))) for i in ids])
        clear, tilt = np.array(clear), np.array(tilt)
        report["stance"] = {"frames": int(len(rows)), "foot_clearance_mean_m": float(clear.mean()), "foot_clearance_std_m": float(clear.std()),
                            "sole_tilt_median_deg": float(np.median(tilt)), "sole_tilt_p90_deg": float(np.percentile(tilt, 90)), "paper_clearance": "-0.004 +- 0.004"}  # fmt: skip
        verdict("stance", abs(clear.mean()) < 0.015 and np.median(tilt) < 5, f"standing still, {len(rows)} frames: foot collision geometry {clear.mean():+.4f} +- {clear.std():.4f} m "
                f"from the floor (paper, its foot geometry: -0.004 +- 0.004), sole tilt median {np.median(tilt):.1f} deg, 90th percentile {np.percentile(tilt, 90):.1f} deg")  # fmt: skip
    else:
        verdict("stance", None, f"only {len(rows)} standing frames found")
else:
    verdict("converted data", None, f"{feas_path} not found: run tools/bones_seed_filter.py first")

# ---------------- a video of the reference alone
if args.render > 0:
    pool = np.flatnonzero(kept & (length >= 150))
    chosen = np.sort(rng.choice(pool, min(args.render, len(pool)), replace=False))
    T = int(min(args.render_seconds * 50, max(length[chosen])))
    quat2mat = lambda q: Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()  # noqa: E731
    blocks = [np.array(mm[int(start[c]) : int(start[c] + length[c])]) for c in chosen]
    offs = np.cumsum([0] + [len(b) for b in blocks[:-1]])
    f = np.stack([offs[k] + np.minimum(np.arange(T), len(b) - 1) for k, b in enumerate(blocks)], 1)
    allx = np.concatenate(blocks)
    short = [names[c].split("/")[-1] for c in chosen]
    npz, mp4 = os.path.join(args.cache, "verify_reference.npz"), os.path.join(args.cache, "verify_reference.mp4")
    np.savez_compressed(npz, names=np.array(short), root_p=allx[f][..., :3], root_quat=allx[f][..., 3:7], q=allx[f][..., 7:], f=f, alive=np.ones(f.shape, bool),
                        ref_root_p=allx[:, :3], ref_root_R=quat2mat(allx[:, 3:7]), ref_q=allx[:, 7:], clip_start=offs, clip_len=np.array([len(b) for b in blocks]))  # fmt: skip
    cmd = [sys.executable, os.path.join(REPO, "tools", "render.py"), npz, mp4, "--seconds", str(args.render_seconds), "--title", "REFERENCE ONLY (conversion check)"]
    try:
        subprocess.run(cmd, check=True, timeout=900, stdin=subprocess.DEVNULL)
        report["render"] = {"video": mp4, "clips": short}
        print(f"[verify] rendered {short} to {mp4}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[verify] render failed, skipped: {e!r}", flush=True)
    os.remove(npz)

report["verdicts"] = {k: ("pass" if v else "check" if v is not None else "skipped") for k, v in verdicts}
json.dump(report, open(os.path.join(args.cache, "verify_report.json"), "w"), indent=1)
bad = [k for k, v in verdicts if v is False]
print(f"[verify] {sum(v is True for _, v in verdicts)} passed, {len(bad)} to check {bad}, {sum(v is None for _, v in verdicts)} skipped; report: {os.path.join(args.cache, 'verify_report.json')}")
sys.exit(1 if bad else 0)
