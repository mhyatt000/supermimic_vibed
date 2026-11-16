"""Simulator-free check of supermimic/dist.py on a toy copy of train.py's update structure.

Mirrors what the real loop does to its networks: three MLPs with the real running normalizers, Adam or Muon+Adam,
bf16 autocast, torch.compile'd rollout steps, BPTT through a frozen "world model" (requires_grad toggled),
several critic steps per iteration, gradient-norm and value clipping, a finite check before each step.

    python tools/test_dist.py                                                   # single process, dist inactive
    python -m torch.distributed.run --standalone --nproc_per_node 1 tools/test_dist.py   # must print the same losses
    python -m torch.distributed.run --standalone --nproc_per_node 2 tools/test_dist.py   # spread must stay 0
    ... tools/test_dist.py --die_rank 1 --die_at 5                              # launcher must stop the other ranks

Each rank draws different data (per-rank seed); "spread" is the largest parameter/buffer difference between ranks.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import supermimic.dist as dist

dist.early_init()

import torch
import torch.nn as nn

import supermimic.networks as networks
from supermimic.networks import RunningNorm

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=30)
p.add_argument("--width", type=int, default=2048)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--horizon", type=int, default=8)
p.add_argument("--optimizer", default="adam", choices=["adam", "muon"])
p.add_argument("--amp", default="bf16", choices=["none", "bf16"])
p.add_argument("--no_compile", action="store_true")
p.add_argument("--die_rank", type=int, default=-1)
p.add_argument("--die_at", type=int, default=-1)
args = p.parse_args()

dev = "cuda:0"
assert torch.cuda.device_count() == 1 or not dist.ACTIVE, "rank is not masked to one GPU"
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
AMP = torch.bfloat16 if args.amp == "bf16" else None
D, A = 64, 16


def mlp(i, hidden, o):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class Net(nn.Module):
    def __init__(self, i, hidden, o):
        super().__init__()
        self.net, self.norm = mlp(i, hidden, o), RunningNorm(i)

    def forward(self, x):
        x = self.norm(x)
        if AMP is None:
            return self.net(x)
        with torch.autocast("cuda", dtype=AMP):
            return self.net(x).float()


W = args.width
wm, pi, critic = Net(D + A, [W, W, W], D).to(dev), Net(D, [W, W // 2], A).to(dev), Net(D, [256, 128], 1).to(dev)


def build_optimizers(module, betas=(0.9, 0.999)):
    if args.optimizer == "adam":
        return [torch.optim.Adam(module.parameters(), lr=2e-4, betas=betas)]
    hidden = [lin.weight for seq in module.modules() if isinstance(seq, nn.Sequential)
              for lin in [m for m in seq if isinstance(m, nn.Linear)][1:-1]]  # fmt: skip
    ids = {id(q) for q in hidden}
    rest = [q for q in module.parameters() if id(q) not in ids]
    return [torch.optim.Adam(rest, lr=2e-4, betas=betas), torch.optim.Muon(hidden, lr=2e-4, adjust_lr_fn="match_rms_adamw")]


def opt_step(opts, loss, params, max_norm, clip_value=0.0):
    params = list(params)
    for o in opts:
        o.zero_grad(set_to_none=True)
    loss.backward()
    dist.sync_grads(params)
    gn = nn.utils.clip_grad_norm_(params, max_norm)
    if clip_value > 0:
        nn.utils.clip_grad_value_(params, clip_value)
    if torch.isfinite(gn):
        for o in opts:
            o.step()
    return gn


opt_wm, opt_pi, opt_v = build_optimizers(wm), build_optimizers(pi, (0.9, 0.95)), build_optimizers(critic)
dist.init()
dist.broadcast_state(wm, pi, critic)
if dist.ACTIVE:
    networks.MOMENT_REDUCE = dist.reduce_moments
if dist.RANK:
    torch.manual_seed(dist.rank_seed(0))  # after the (identical) model init


def _wm_step(s, a):
    return s + 0.1 * wm(torch.cat((s, a), -1))


def _imag_step(s):
    a = pi(s)
    return _wm_step(s, a), a


maybe_compile = (lambda f: f) if args.no_compile else torch.compile
wm_step, imag_step = maybe_compile(_wm_step), maybe_compile(_imag_step)


def set_requires_grad(m, flag):
    for q in m.parameters():
        q.requires_grad_(flag)


def true_dyn(s, a):
    return s + 0.1 * torch.tanh(s.roll(1, -1) + a.repeat(1, D // A))


t_sync, n_sync = 0.0, 0
t0 = time.time()
for it in range(args.iters):
    if it == args.die_at and dist.RANK == args.die_rank:
        raise RuntimeError(f"rank {dist.RANK}: injected failure at iteration {it}")
    s, a = torch.randn(args.batch, D, device=dev), torch.randn(args.batch, A, device=dev)
    wm.norm.update(torch.cat((s, a), -1)), pi.norm.update(s), critic.norm.update(s)
    for _ in range(3):  # world-model fit
        wm_loss = ((wm_step(s, a) - true_dyn(s, a)) ** 2).mean()
        opt_step(opt_wm, wm_loss, wm.parameters(), 1.0)
    set_requires_grad(wm, False), set_requires_grad(critic, False)  # policy: BPTT through the frozen world model
    x, J = s, 0.0
    for t in range(args.horizon):
        x, act = imag_step(x)
        J = J - (x**2).mean(-1) - 0.01 * (act**2).mean(-1)
    pi_loss = -(J + critic(x).squeeze(-1)).mean()
    gn = opt_step(opt_pi, pi_loss, pi.parameters(), 1.0, clip_value=0.1)
    set_requires_grad(wm, True), set_requires_grad(critic, True)
    for _ in range(4):  # critic
        v_loss = ((critic(s).squeeze(-1) - J.detach()) ** 2).mean()
        opt_step(opt_v, v_loss, critic.parameters(), 1.0)
    if (it + 1) % 10 == 0 or it + 1 == args.iters:
        spread = dist.state_spread(wm, pi, critic)
        row = dist.mean_floats([float(wm_loss), float(pi_loss), float(v_loss), float(gn)])
        if dist.is_main():
            print(f"it {it + 1} wm {row[0]:.9g} pi {row[1]:.9g} v {row[2]:.9g} gn {row[3]:.9g} spread {spread:.3g} "
                  f"count {float(wm.norm.count):.0f}", flush=True)  # fmt: skip
        assert spread == 0.0, f"ranks diverged: {spread}"

if dist.ACTIVE:  # cost of one gradient all-reduce at the real world-model size
    params = list(wm.parameters())
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        dist.sync_grads(params)
    torch.cuda.synchronize()
    ms = 1e3 * (time.time() - t) / 20
    if dist.is_main():
        import torch.distributed as td

        print(f"world {dist.WORLD} backend {td.get_backend()} | grad all-reduce of {sum(q.numel() for q in params) / 1e6:.1f} M "
              f"params: {ms:.2f} ms", flush=True)  # fmt: skip
if dist.is_main():
    print(f"done in {time.time() - t0:.1f} s", flush=True)
dist.shutdown()
