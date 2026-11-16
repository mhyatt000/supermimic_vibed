"""The two motion filters of the SuperMimic paper (Appendix E) on a converted BONES-SEED cache.

Writes into the cache directory:
  manifest.txt        the kept clips, one name per line (what supermimic/motion_cache.py loads)
  rejected.csv        name, pass (name | feasibility), reason, detail for every dropped clip
  filter_report.json  counts after each pass next to the paper's, keyword hit counts, thresholds
  feasibility.f16     float16 (num_frames, 3): lowest body-link height, left and right foot clearance (kept so
                      the rules can be re-run with other thresholds in seconds: ``--reuse``)

Pass 1, name filter. A clip is dropped if its name contains one of the keywords (case-insensitive substring, the
rule of SONIC's script). SONIC_KEYWORDS is copied verbatim from NVlabs/GR00T-WholeBodyControl,
gear_sonic/data_process/filter_and_copy_bones_data.py (commit 1e1ec4f). The paper says its list is "adapted from
the keyword list in SONIC's released data-processing code, plus a few terms it misses" and names stair, ladder, box,
chair, sit, climb, "jump off 50cm", "look over fence" as examples; the exact list is not published. EXTRA_KEYWORDS
are our reading of the terms SONIC's list does not already cover; ``--extra`` replaces them.

Pass 2, feasibility filter. Every frame goes through MuJoCo forward kinematics; z_min(t) is the height of the lowest
body link (link frame origin, floor at z = 0). A clip is dropped for
  elevated support    z_min > 0.20 m for at least 0.6 s
  ground penetration  z_min < -0.12 m at any frame, or z_min < -0.08 m for over 80 % of the clip
  floating feet       for at least 0.5 s both feet are more than 0.02 m above the floor (mj_geomDistance from the
                      foot collision geoms to the floor plane) while the pelvis is upright, at standing height and
                      nearly stationary
The paper gives no numbers for "upright", "standing height" and "nearly stationary"; ours are --float_tilt_deg,
--float_pelvis_z and --float_speed.

  tools/uv_run.sh tools/bones_seed_filter.py --cache data/bones_seed/cache
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PAPER = {"raw": 142220, "after_name": 129667, "after_feasibility": 129514, "hours": 264.56}
SONIC_KEYWORDS = [
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table", "_sit_", "sit_", "ladder", "crutch",
    "_bed_", "_ride_", "scooter", "stepdown", "acrobatics_", "box_HSPU", "cartwheel", "50cm_box_", "on_box", "fall_from",
    "handstand_ff_", "on_1m", "form_box", "off_1m", "230m", "jump_over_obstacle_", "lift_crate_come_up_", "jump_to_shoulder_roll",
    "kozak_dance", "stair", "handstand", "box_jump", "monkey_jump", "safety_roll", "box_dips", "walking_on_edge", "push_obstacle",
]  # fmt: skip
EXTRA_KEYWORDS = ["box", "50cm", "jump_off", "fence"]
FEAS = "feasibility.f16"
DT = 0.02

ap = argparse.ArgumentParser()
ap.add_argument("--cache", type=str, default=os.path.join(REPO, "data", "bones_seed", "cache"))
ap.add_argument("--extra", nargs="*", default=EXTRA_KEYWORDS, help="keywords added to SONIC's list")
ap.add_argument("--no_name_filter", action="store_true")
ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 2, 1))
ap.add_argument("--reuse", action="store_true", help="reuse feasibility.f16 instead of recomputing the kinematics")
ap.add_argument("--dry_run", action="store_true", help="report only: do not write manifest.txt / rejected.csv")
ap.add_argument("--elevated_z", type=float, default=0.20)
ap.add_argument("--elevated_s", type=float, default=0.6)
ap.add_argument("--penetration_z", type=float, default=-0.12)
ap.add_argument("--penetration_frac_z", type=float, default=-0.08)
ap.add_argument("--penetration_frac", type=float, default=0.8)
ap.add_argument("--float_z", type=float, default=0.02)
ap.add_argument("--float_s", type=float, default=0.5)
ap.add_argument("--float_tilt_deg", type=float, default=20.0, help="pelvis upright: its z axis within this angle of vertical")
ap.add_argument("--float_pelvis_z", type=float, default=0.65, help="standing height: pelvis at least this high (m)")
ap.add_argument("--float_speed", type=float, default=0.10, help="nearly stationary: pelvis speed at most this (m/s)")

_SIM = None


def _sim():
    """The G1 MuJoCo model with a floor plane added: ``(model, data, floor geom, [left foot geoms, right foot geoms])``."""
    global _SIM
    if _SIM is None:
        import mujoco

        from supermimic.robot import G1_JOINT_NAMES, MJCF_PATH

        spec = mujoco.MjSpec.from_file(MJCF_PATH)
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
        m = spec.compile()
        assert m.nq == 36 and [m.joint(j).name for j in range(1, m.njnt)] == G1_JOINT_NAMES  # cache rows are qpos
        feet = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(n).id and m.geom_contype[g] != 0]
                for n in ("left_ankle_roll_link", "right_ankle_roll_link")]  # fmt: skip
        assert all(len(f) > 0 for f in feet)
        _SIM = (m, mujoco.MjData(m), m.geom("floor").id, feet)
    return _SIM


def kinematics(job):
    """Rows ``a..b`` of the cache -> float16 (b-a, 3): lowest body-link height, left and right foot clearance."""
    import mujoco

    from supermimic.motion_cache import open_frames

    cache, a, b = job
    m, d, floor, feet = _sim()
    x = np.array(open_frames(cache)[a:b], np.float64)
    out = np.empty((b - a, 3), np.float32)
    for t in range(b - a):
        d.qpos[:] = x[t]  # root position, root quaternion wxyz, joints in MJCF order
        mujoco.mj_kinematics(m, d)
        out[t, 0] = d.xpos[1:, 2].min()
        out[t, 1] = min(mujoco.mj_geomDistance(m, d, g, floor, 10.0, None) for g in feet[0])
        out[t, 2] = min(mujoco.mj_geomDistance(m, d, g, floor, 10.0, None) for g in feet[1])
    return a, out.astype(np.float16)


def longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True."""
    if not mask.any():
        return 0
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def name_hits(names: list[str], keywords: list[str]) -> list[list[str]]:
    low = [n.lower() for n in names]
    return [[k for k in keywords if k.lower() in n] for n in low]


if __name__ == "__main__":
    args = ap.parse_args()
    from supermimic.motion_cache import FRAME_DIM, MANIFEST, open_frames, read_clips

    t0 = time.time()
    names, start, length = read_clips(args.cache)
    C, N = len(names), int(length.sum())
    mm = open_frames(args.cache)
    assert len(mm) == N and mm.shape[1] == FRAME_DIM, (mm.shape, N)

    # ---------------- pass 1: names
    keywords = [] if args.no_name_filter else SONIC_KEYWORDS + [k for k in args.extra if k not in SONIC_KEYWORDS]
    hits = name_hits(names, keywords)
    name_drop = np.array([len(h) > 0 for h in hits])
    sonic_drop = np.array([any(k in SONIC_KEYWORDS for k in h) for h in hits])
    kw_count = {k: [sum(k in h for h in hits), sum(h == [k] for h in hits)] for k in keywords}

    # ---------------- pass 2: kinematics of every frame, then the rules
    feas_path = os.path.join(args.cache, FEAS)
    if not (args.reuse and os.path.isfile(feas_path) and os.path.getsize(feas_path) == N * 3 * 2):
        out = np.lib.format.open_memmap(feas_path + ".tmp.npy", mode="w+", dtype=np.float16, shape=(N, 3))
        jobs = [(args.cache, a, min(a + 100_000, N)) for a in range(0, N, 100_000)]
        with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            for k, (a, z) in enumerate(pool.imap_unordered(kinematics, jobs)):
                out[a : a + len(z)] = z
                if (k + 1) % 50 == 0:
                    print(f"[filter] kinematics {k + 1}/{len(jobs)} blocks, {time.time() - t0:.0f} s", flush=True)
        np.asarray(out).tofile(feas_path)
        del out
        os.remove(feas_path + ".tmp.npy")
        print(f"[filter] kinematics of {N} frames in {time.time() - t0:.0f} s", flush=True)
    feas = np.memmap(feas_path, dtype=np.float16, mode="r", shape=(N, 3))

    cos_tilt = np.cos(np.deg2rad(args.float_tilt_deg))
    n_elev, n_float = int(np.ceil(args.elevated_s / DT - 1e-9)), int(np.ceil(args.float_s / DT - 1e-9))
    reasons: list[list[str]] = [[] for _ in range(C)]
    details: list[list[str]] = [[] for _ in range(C)]
    stance = []  # foot clearance on frames standing still on the floor
    for c in range(C):
        a, b = int(start[c]), int(start[c] + length[c])
        z = np.asarray(feas[a:b], np.float32)
        x = np.asarray(mm[a:b, :7])
        zmin, feet_min = z[:, 0], np.minimum(z[:, 1], z[:, 2])
        run = longest_run(zmin > args.elevated_z)
        if run >= n_elev:
            reasons[c].append("elevated_support"), details[c].append(f"z_min>{args.elevated_z} for {run * DT:.2f}s")
        frac = float((zmin < args.penetration_frac_z).mean())
        if zmin.min() < args.penetration_z or frac > args.penetration_frac:
            reasons[c].append("ground_penetration"), details[c].append(f"min z_min {zmin.min():.3f}; z_min<{args.penetration_frac_z} in {100 * frac:.0f}%")
        p, quat = x[:, :3], x[:, 3:7] / np.linalg.norm(x[:, 3:7], axis=-1, keepdims=True)
        up = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)  # R[2,2]: cosine of the pelvis tilt
        speed = np.linalg.norm(np.gradient(p, DT, axis=0), axis=-1) if len(p) > 1 else np.zeros(len(p))
        still = (up >= cos_tilt) & (p[:, 2] >= args.float_pelvis_z) & (speed <= args.float_speed)
        run = longest_run(still & (z[:, 1] > args.float_z) & (z[:, 2] > args.float_z))
        if run >= n_float:
            reasons[c].append("floating_feet"), details[c].append(f"both feet>{args.float_z} for {run * DT:.2f}s while standing still")
        if not name_drop[c] and not reasons[c]:
            g = feet_min[still & (np.maximum(z[:, 1], z[:, 2]) < args.float_z)]
            if len(g):
                stance.append(np.array([g.mean(), len(g)]))
    feas_drop = np.array([len(r) > 0 for r in reasons])
    keep = ~name_drop & ~feas_drop

    # ---------------- report
    n1, n2 = int((~name_drop).sum()), int(keep.sum())
    hours = float(length[keep].sum()) * DT / 3600
    by_rule = {r: int(sum(r in reasons[c] for c in range(C) if not name_drop[c])) for r in ("elevated_support", "ground_penetration", "floating_feet")}
    st = np.array(stance) if stance else np.zeros((0, 2))
    report = {
        "raw": C, "after_name": n1, "after_feasibility": n2, "hours": round(hours, 2), "paper": PAPER,
        "difference_to_paper": {"raw": C - PAPER["raw"], "after_name": n1 - PAPER["after_name"],
                                "after_feasibility": n2 - PAPER["after_feasibility"], "hours": round(hours - PAPER["hours"], 2)},
        "removed_by_name": C - n1, "removed_by_name_sonic_list_only": int(sonic_drop.sum()), "removed_by_feasibility": n1 - n2,
        "feasibility_by_rule_after_name_filter": by_rule,
        "feasibility_flags_among_name_removed": int((feas_drop & name_drop).sum()),
        "keywords": {k: {"clips": v[0], "only_this_keyword": v[1]} for k, v in kw_count.items()},
        "stance_clearance_m": {"clips": len(st), "mean": float(np.average(st[:, 0], weights=st[:, 1])) if len(st) else None,
                               "std_over_clips": float(st[:, 0].std()) if len(st) else None, "paper": "-0.004 +- 0.004"},
        "args": vars(args),
    }  # fmt: skip
    print(f"[filter] {'stage':<26s}{'ours':>10s}{'paper':>10s}{'diff':>8s}")
    for k, label in (("raw", "raw clips"), ("after_name", "after name filter"), ("after_feasibility", "after feasibility filter")):
        print(f"[filter] {label:<26s}{report[k]:>10d}{PAPER[k]:>10d}{report[k] - PAPER[k]:>+8d}")
    print(f"[filter] {'hours kept':<26s}{hours:>10.2f}{PAPER['hours']:>10.2f}{hours - PAPER['hours']:>+8.2f}")
    print(f"[filter] name filter removes {C - n1} (SONIC's list alone: {int(sonic_drop.sum())}; paper 12553); "
          f"feasibility removes {n1 - n2} more (paper 153): {by_rule}")  # fmt: skip
    if len(st):
        print(f"[filter] foot clearance when standing still on the floor: {report['stance_clearance_m']['mean']:+.4f} m "
              f"(std over {len(st)} clips {report['stance_clearance_m']['std_over_clips']:.4f}); paper: -0.004 +- 0.004 m")  # fmt: skip
    if not args.dry_run:
        with open(os.path.join(args.cache, MANIFEST + ".tmp"), "w") as fh:
            fh.write("".join(f"{names[c]}\n" for c in range(C) if keep[c]))
        os.replace(os.path.join(args.cache, MANIFEST + ".tmp"), os.path.join(args.cache, MANIFEST))
        with open(os.path.join(args.cache, "rejected.csv"), "w") as fh:
            fh.write("name,pass,reason,detail\n")
            for c in range(C):
                if name_drop[c]:
                    also = f"; also {'+'.join(reasons[c])}" if reasons[c] else ""
                    fh.write(f"{names[c]},name,keyword,{' '.join(hits[c])}{also}\n")
                elif feas_drop[c]:
                    fh.write(f"{names[c]},feasibility,{'+'.join(reasons[c])},{'; '.join(details[c])}\n")
        json.dump(report, open(os.path.join(args.cache, "filter_report.json"), "w"), indent=1)
        print(f"[filter] wrote {MANIFEST} ({n2} clips), rejected.csv ({C - n2}), filter_report.json in {args.cache} ({time.time() - t0:.0f} s)")
