"""Optional Weights & Biases logging for train.py (``--wandb_project``); the training process is the only writer.

Training rows go to global step = iteration. Evaluation rows and videos are produced later by a separate
process (``evaluate.py --watch --video``), so the trainer tails ``eval.csv`` and ``videos/`` and logs them
against the custom step ``eval/iteration`` — at most one per training row, so the global step never goes
backwards and nothing is dropped. A W&B failure after start-up is printed and never stops training.
"""

from __future__ import annotations

import csv
import glob
import io
import os
import re
import time

WATCH_PID = "eval_watch.pid"  # written by evaluate.py --watch
RUN_ID = "wandb_run_id.txt"


def pid_alive(pid_file: str) -> bool:
    try:
        os.kill(int(open(pid_file).read()), 0)
        return True
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False


class WandbLogger:
    def __init__(self, args, start_it: int = 0):
        import wandb  # only needed when enabled

        self.wandb, self.dir, self.start_it = wandb, args.log_dir, start_it
        id_file = os.path.join(self.dir, RUN_ID)
        rid = open(id_file).read().strip() if args.resume and os.path.exists(id_file) else None  # --resume continues the run
        self.run = wandb.init(project=args.wandb_project, name=args.wandb_name, group=args.wandb_group, config=vars(args),
                              dir=self.dir, id=rid, resume="allow" if rid else None)  # fmt: skip
        open(id_file, "w").write(self.run.id + "\n")
        wandb.define_metric("eval/iteration")
        wandb.define_metric("eval/*", step_metric="eval/iteration")
        self.floor = int(getattr(self.run, "step", 0) or 0)  # a resumed run's last step: W&B drops rows logged below it
        self.step = max(start_it, self.floor)
        self.seen: set[tuple[str, int]] = set()

    def _pending(self) -> list[tuple[tuple[str, int], dict]]:
        """Evaluation rows and videos not logged yet (those up to a resumed iteration count as logged)."""
        out = []
        try:
            txt = open(os.path.join(self.dir, "eval.csv")).read()
        except OSError:
            txt = ""
        for r in csv.DictReader(io.StringIO(txt[: txt.rfind("\n") + 1])):  # complete lines only
            try:
                d = {"eval/iteration": int(r["iteration"]), "eval/norm_return": float(r["norm_return"]),
                     "eval/norm_return_framew": float(r["norm_return_framew"]), "eval/n_failed": int(r["n_failed"]),
                     "eval/train_wall_h": float(r["train_wall_h"])}  # fmt: skip
            except (KeyError, TypeError, ValueError):
                continue
            out.append((("row", d["eval/iteration"]), d))
        for p in sorted(glob.glob(os.path.join(self.dir, "videos", "ckpt_*.mp4"))):
            m = re.fullmatch(r"ckpt_(\d+)\.mp4", os.path.basename(p))  # not the *.tmp.mp4 being rendered
            if m:
                out.append((("video", int(m.group(1))), {"eval/iteration": int(m.group(1)), "eval/video": p}))
        return [(k, d) for k, d in out if k not in self.seen and k[1] > self.start_it]

    def log(self, step: int, data: dict):
        """Log one row at global step ``step`` (must not decrease), plus at most one pending evaluation item."""
        try:
            pend = self._pending() if step >= self.floor else []
            if pend:
                key, extra = pend[0]
                self.seen.add(key)
                if key[0] == "video":
                    extra = {**extra, "eval/video": self.wandb.Video(extra["eval/video"])}
                data = {**data, **extra}
            self.step = max(self.step, step)
            self.wandb.log(data, step=step, commit=True)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] log failed at step {step}: {e!r}", flush=True)

    def finish(self, wait_min: float = 0.0):
        """Wait (bounded) for a live eval watcher to finish the last checkpoint, log what is left, close the run."""
        pid_file, t0 = os.path.join(self.dir, WATCH_PID), time.time()
        while pid_alive(pid_file) and time.time() - t0 < 60 * wait_min:
            time.sleep(5)
        while self._pending():  # steps past the last iteration; the eval panels use eval/iteration
            n = len(self.seen)
            self.log(self.step + 1, {})
            if len(self.seen) == n:
                break
        self.wandb.finish()
