"""Checkpoint evaluation (paper Appendix F.1) on all motions of the library.

Each motion plays once from its first frame with friction 1.0, no domain randomization, no
observation or action noise. A motion fails if the rollout triggers the training termination.
Normalized return is the undiscounted reward sum divided by its maximum attainable value.

Run as a watcher: ``python supermimic/evaluate.py --run_dir logs/<run> --watch``. With ``--video`` it also renders
a short video of a few clips per checkpoint into ``<run_dir>/videos/`` (tools/render.py in a subprocess; a failed
render is logged and skipped). The watcher exits once training is over (``<run_dir>/done`` or ``--train_pid``)
and the remaining checkpoints are processed.

Large libraries (BONES-SEED, ``--motion_dir`` of a cache written by tools/bones_seed_convert.py):
``--eval_clips N --eval_seed S`` scores a fixed subset of N clips, the same for every checkpoint and every run with
that seed; ``--eval_envs E`` plays the clips E at a time (longest first) instead of all at once, which is how the
whole benchmark fits on a GPU: ``--ckpt <file> --eval_envs 4096 --out <result.json>``.
"""

from __future__ import annotations

import argparse
import glob
import json
import atexit
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--record", type=str, default=None, help="save trajectories (npz) for rendering")
parser.add_argument("--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1"))
parser.add_argument("--eval_clips", type=int, default=0, help="score a fixed random subset of this many clips (0: all clips)")
parser.add_argument("--eval_seed", type=int, default=0, help="seed of the --eval_clips subset")
parser.add_argument("--eval_envs", type=int, default=0, help="environments; fewer than clips: the clips are played in chunks (0: one environment per clip)")
parser.add_argument("--out", type=str, default=None, help="--ckpt: also write the result, with the per-clip returns, to this json file")
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
parser.add_argument("--latest_only", action="store_true", help="with a backlog of checkpoints, evaluate only the newest")
parser.add_argument("--train_pid", type=int, default=None, help="--watch: also stop when this process is gone")
parser.add_argument("--video", action="store_true", help="render <run_dir>/videos/ckpt_<it>.mp4 for the newest checkpoint of each pass")
parser.add_argument("--video_clips", nargs="*", default=["walk1_subject1", "dance1_subject1", "jumps1_subject1", "fallAndGetUp1_subject1"])
parser.add_argument("--video_seconds", type=float, default=10.0, help="per clip")
parser.add_argument("--video_start", type=float, default=0.0, help="seconds into each clip")
parser.add_argument("--video_timeout", type=float, default=600.0, help="give up on a render after this many seconds")
add_launcher_args(parser)
args = parser.parse_args()

import numpy as np
import torch

import supermimic.networks as networks
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import ACTOR_OBS_DIMS, MAX_STEP_REWARD, MDP
from supermimic.motion import ACTOR_GOAL_DIM, OFFICIAL_ACTOR_GOAL_DIM, MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda:0"
kin = G1Kinematics(dev)
if args.eval_clips > 0:
    from supermimic.motion_cache import load_subset

    motion = load_subset(args.motion_dir, kin, dev, args.eval_clips, args.eval_seed)
else:
    motion = MotionLib.from_dir(args.motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
C = len(motion.names)
E = C if args.eval_envs <= 0 else min(args.eval_envs, C)  # environments
# CHUNKS: (clip of each environment, number of environments that count). One chunk with every clip in library order,
# unless --eval_envs is smaller than the library: then longest clips first, so that a chunk's clips end together.
if E == C:
    CHUNKS = [(torch.arange(C, device=dev), C)]
else:
    _order = torch.argsort(motion.clip_len, descending=True, stable=True)
    CHUNKS = [(_order[a : a + E], min(E, C - a)) for a in range(0, C, E)]
    CHUNKS[-1] = (torch.cat((CHUNKS[-1][0], CHUNKS[-1][0][-1:].expand(E - CHUNKS[-1][1]))), CHUNKS[-1][1])  # pad the last chunk
    print(f"[eval] {C} clips in {len(CHUNKS)} chunks of {E} environments, {sum(int(motion.clip_len[c[0]]) - 1 for c, _ in CHUNKS)} steps per checkpoint", flush=True)
FAILED_NAMES_IN_CSV = 100  # eval.csv lists at most this many failed clips per checkpoint (the json has all of them)


@torch.no_grad()
def evaluate(env: TrackingSim, ck: dict, record: str | None = None, rec_clips: list[int] | None = None, rec_steps: int | None = None) -> dict:
    """Score a checkpoint on all clips; ``record`` saves the trajectories (optionally only some clips / the first steps)."""
    a_ = ck.get("args") or json.load(open(os.path.join(args.run_dir, "args.json")))
    mode = a_.get("actor_obs", "proprio")  # checkpoints from before the switch are proprio
    Hh, official = (a_["hist"] if mode == "proprio" else 1), mode == "privileged_official"
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    pi = Policy(ACTOR_OBS_DIMS[mode], OFFICIAL_ACTOR_GOAL_DIM if official else ACTOR_GOAL_DIM, hist=Hh).to(dev)
    for k, w in (("dec.0.weight", pi.dec[0].weight), ("enc.0.weight", pi.enc[0].weight)):
        if ck["pi"][k].shape != w.shape:
            raise RuntimeError(f"checkpoint policy does not match its args (actor_obs={mode}, hist={Hh}): {k} {tuple(ck['pi'][k].shape)}, expected {tuple(w.shape)}")
    pi.load_state_dict(ck["pi"])
    pi.eval()
    tail = lambda d: os.path.normpath(d).split(os.sep)[-2:]  # noqa: E731  (the same data under another mount point is fine)
    if tail(a_.get("motion_dir") or args.motion_dir) != tail(args.motion_dir) and not getattr(evaluate, "warned", False):
        evaluate.warned = True
        print(f"[eval] WARNING: the checkpoint was trained on {a_['motion_dir']}, evaluating on {args.motion_dir}", flush=True)
    ids = torch.arange(E, device=dev)
    ret = torch.zeros(C, device=dev)
    failed = torch.zeros(C, dtype=torch.bool, device=dev)
    keep = ids if rec_clips is None else torch.tensor(rec_clips, device=dev)
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    for k, (clips, n) in enumerate(CHUNKS):
        t0 = time.time()
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(ids, motion.clip_start[clips])
        clip_end = motion.clip_end[clips]
        ret_env = torch.zeros(E, device=dev)
        hist = None
        while not bool(env.done_eval.all()):
            S = env.state()
            o = mdp.actor_obs(S, env.a_prev, env.xi, mode)
            hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            a = pi(hist, motion.goal_official(env.f, S[1]) if official else motion.goal(env.f, S[0], S[1])[0])
            alive = ~env.done_eval.clone()
            if record and (rec_steps is None or len(traj["f"]) < rec_steps):
                from supermimic.rotations import mat_to_quat

                traj["root_p"].append(S[0][keep].cpu()), traj["root_quat"].append(mat_to_quat(S[1])[keep].cpu())
                traj["q"].append(S[4][keep].cpu()), traj["f"].append(env.f[keep].cpu()), traj["alive"].append(alive[keep].cpu())
            out = env.step(a)
            ret_env += torch.where(alive, out["r"], 0.0)
            env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
        ret[clips[:n]], failed[clips[:n]] = ret_env[:n], env.failed_eval[:n]
        if len(CHUNKS) > 1 and (k % 4 == 3 or k + 1 == len(CHUNKS)):
            print(f"[eval] chunk {k + 1}/{len(CHUNKS)}: {int(motion.clip_len[clips[0]]) - 1} steps in {time.time() - t0:.0f}s, failed so far {int(failed.sum())}", flush=True)
    norm = ret / ((motion.clip_len - 1).float() * MAX_STEP_REWARD)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((motion.clip_len - 1).sum() * MAX_STEP_REWARD)).item(),
        "n_failed": int(failed.sum().item()),
        "failed": [motion.names[i] for i in failed.nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
    }
    if record:
        np.savez_compressed(record, names=np.array(motion.names)[keep.cpu().numpy()], **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start[keep].cpu().numpy(), clip_len=motion.clip_len[keep].cpu().numpy())  # fmt: skip
    return res


def training_over() -> bool:
    if os.path.exists(os.path.join(args.run_dir, "done")):  # written by train.py when it ends
        return True
    if args.train_pid is not None:
        try:
            os.kill(args.train_pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
    return False


def make_video(it: int, traj: str, clips: list[str]):
    """Render the recorded clips to videos/ckpt_<it>.mp4 in a subprocess; a failure is logged, never raised."""
    out = os.path.join(args.run_dir, "videos", f"ckpt_{it:06d}.mp4")
    tmp = out[:-4] + ".tmp.mp4"  # renamed when complete: train.py's W&B logger picks up ckpt_<it>.mp4
    cmd = [sys.executable, os.path.join(REPO, "tools", "render.py"), traj, tmp, "--clips", *clips, "--seconds", str(args.video_seconds),
           "--start", str(args.video_start), "--title", f"it {it}"]  # fmt: skip
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=args.video_timeout, stdin=subprocess.DEVNULL)
        if os.path.getsize(tmp) == 0:
            raise RuntimeError("empty video")
        os.replace(tmp, out)
        print(f"[video] it {it}: {out} ({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[video] it {it}: render failed after {time.time() - t0:.0f}s, skipped: {e!r}", flush=True)


sim_cfg = make_sim(dev, args)
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, E, motion, mdp, eval_mode=True)
    if len(CHUNKS) > 1 and (args.record or args.video):
        print("[eval] --record / --video need one environment per clip (no --eval_envs below the number of clips): ignored", flush=True)
        args.record, args.video = None, False
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev)
        t0 = time.time()
        res = evaluate(env, ck, args.record)
        if args.out:
            json.dump({**res, "num_clips": C, "ckpt": args.ckpt, "motion_dir": args.motion_dir, "eval_clips": args.eval_clips,
                       "eval_seed": args.eval_seed, "eval_seconds": round(time.time() - t0)}, open(args.out, "w"), indent=1)  # fmt: skip
            res["failed"] = res["failed"][:FAILED_NAMES_IN_CSV]
        print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
        sys.exit(0)
    os.makedirs(os.path.join(args.run_dir, "videos") if args.video else args.run_dir, exist_ok=True)
    vid_clips = [n for n in args.video_clips if n in motion.names]
    if args.video and len(vid_clips) < len(args.video_clips):
        print(f"[video] clips not in the library, dropped: {sorted(set(args.video_clips) - set(vid_clips))}", flush=True)
    vid_clips = vid_clips or [motion.names[i] for i in sorted({round(k * (C - 1) / 3) for k in range(4)})]
    vid_ids = [motion.names.index(n) for n in vid_clips]
    vid_steps = int((args.video_start + args.video_seconds) * 50)
    traj_path = os.path.join(args.run_dir, "videos", "traj_latest.npz")  # overwritten per checkpoint
    if args.watch:  # lets train.py's W&B logger wait for the last evaluation
        pid_file = os.path.join(args.run_dir, "eval_watch.pid")
        open(pid_file, "w").write(str(os.getpid()))
        atexit.register(lambda: os.path.exists(pid_file) and os.remove(pid_file))
    out_csv = os.path.join(args.run_dir, "eval.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
    else:
        open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
    while True:
        over = args.watch and training_over()  # read before listing, so the last checkpoint is not missed
        ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
        todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
        todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 2000) % args.every == 0]
        if args.latest_only and len(todo) > 1:
            print(f"[eval] backlog: skipping {len(todo) - 1} older checkpoints", flush=True)
            done.update(int(os.path.basename(c)[5:11]) for c in todo[:-1])
            todo = todo[-1:]
        for c in todo:
            it = int(os.path.basename(c)[5:11])
            ck = torch.load(c, map_location=dev)
            vid = args.video and c == todo[-1]  # videos never queue: only the newest pending checkpoint
            t0 = time.time()
            res = evaluate(env, ck, traj_path, vid_ids, vid_steps) if vid else evaluate(env, ck)
            shown = res["failed"][:FAILED_NAMES_IN_CSV] + ([f"+{res['n_failed'] - FAILED_NAMES_IN_CSV} more"] if res["n_failed"] > FAILED_NAMES_IN_CSV else [])
            line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(shown)}\n"
            open(out_csv, "a").write(line)
            json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
            print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)", flush=True)
            done.add(it)
            if vid:
                make_video(it, traj_path, vid_clips)
        if not args.watch or over:
            break
        time.sleep(60)
