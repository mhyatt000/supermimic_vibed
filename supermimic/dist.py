"""Single-node data-parallel training: N processes, one per GPU, each with its own simulator and replay buffer.

The ranks hold identical networks and optimizers and stay identical because every optimizer step uses the
rank-averaged gradient (:func:`sync_grads`) and every running-normalizer update uses the rank-pooled batch
moments (:func:`reduce_moments`). Nothing wraps the modules, so checkpoints keep the single-GPU format.

Inactive unless a launcher exports ``RANK`` and ``WORLD_SIZE`` (``torchrun`` does); every function is then
a no-op and the single-GPU path is untouched. ``WORLD_SIZE=1`` is active and must reproduce it exactly.

Device binding: :func:`early_init` masks ``CUDA_VISIBLE_DEVICES`` to this rank's GPU, so ``"cuda:0"`` is
the rank's own GPU for PyTorch, Warp and Isaac Lab alike. Call it before anything initializes CUDA.
"""

from __future__ import annotations

import os

RANK, WORLD, ACTIVE = 0, 1, False
_shared_gpu = False  # several ranks on one GPU (functional tests only): NCCL refuses that, gloo does not


def early_init() -> bool:
    """Read the rank layout and bind this process to its GPU. Uses no CUDA; returns whether distributed."""
    global RANK, WORLD, ACTIVE, _shared_gpu
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    RANK, WORLD, ACTIVE = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), True
    local = int(os.environ.get("LOCAL_RANK", RANK))
    n_local = int(os.environ.get("LOCAL_WORLD_SIZE", WORLD))
    if n_local != WORLD:
        raise RuntimeError("supermimic.dist supports one node only (ranks of several nodes were launched)")
    gpus = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    gpus = gpus or [str(i) for i in range(n_local)]  # no mask given: assume GPUs 0..N-1
    _shared_gpu = len(gpus) < n_local
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[local % len(gpus)]
    # One node: NCCL needs no network transport, and its InfiniBand/RoCE plugin segfaults in the first collective
    # on g4 (bnxt_re RoCE devices, NCCL 2.28.9): keep it off unless the launcher says otherwise.
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    # The compile caches (Warp kernels, torch.compile) are shared by the ranks, as they are by concurrent
    # single-GPU jobs: both write atomically. SM_DIST_RANK_CACHE=1 gives each rank its own torch.compile cache.
    if WORLD > 1 and os.environ.get("SM_DIST_RANK_CACHE") == "1":
        for var in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            if os.environ.get(var):
                os.environ[var] = os.path.join(os.environ[var], f"rank{RANK}")
    return True


def is_main() -> bool:
    """Rank 0 owns every file the run writes."""
    return RANK == 0


def rank_seed(seed: int) -> int:
    """Seed of this rank's sampling stream; rank 0 keeps ``seed`` so one rank reproduces the single-GPU run."""
    return seed + 1_000_003 * RANK


def init() -> None:
    """Join the process group. Call where every rank arrives quickly (before the simulator starts)."""
    if not ACTIVE:
        return
    from datetime import timedelta

    import torch.distributed as td

    backend = os.environ.get("SM_DIST_BACKEND") or ("gloo" if _shared_gpu else "nccl")
    td.init_process_group(backend=backend, timeout=timedelta(minutes=60))


def shutdown() -> None:
    if ACTIVE:
        import torch.distributed as td

        if td.is_initialized():
            td.destroy_process_group()


def _flat(tensors):
    import torch

    return torch.cat([t.reshape(-1) for t in tensors])


def _unflat(flat, tensors) -> None:
    off = 0
    for t in tensors:
        t.copy_(flat[off : off + t.numel()].view_as(t))
        off += t.numel()


def sync_grads(params) -> None:
    """Replace every gradient by its mean over the ranks: the gradient of the mean loss over all ranks' batches.

    Call between ``backward()`` and clipping, so clipping, the finite check and the optimizer see the same
    numbers on every rank. Parameters without a gradient (frozen modules) are skipped.
    """
    if not ACTIVE:
        return
    import torch.distributed as td

    grads = [p.grad for p in params if p.grad is not None]
    flat = _flat(grads)
    td.all_reduce(flat)
    flat /= WORLD
    _unflat(flat, grads)


def reduce_moments(n: int, mean, var):
    """Pool the batch moments of one normalizer update over the ranks (equal batch sizes on every rank).

    Written so that one rank returns its inputs bit for bit.
    """
    import torch
    import torch.distributed as td

    buf = torch.cat((mean, var, mean * mean))
    td.all_reduce(buf)
    buf /= WORLD
    m, v, m2 = buf.split(mean.shape[0])
    return n * WORLD, m, v + (m2 - m * m).clamp(min=0)  # within-rank variance + variance of the rank means


def mean_floats(xs: list[float]) -> list[float]:
    """Mean of each entry over the ranks (logging)."""
    if not ACTIVE:
        return xs
    import torch
    import torch.distributed as td

    t = torch.tensor(xs, dtype=torch.float64, device="cuda:0")
    td.all_reduce(t)
    return (t / WORLD).tolist()


def broadcast_state(*modules) -> None:
    """Copy rank 0's parameters and buffers to every rank (start of training)."""
    if not ACTIVE:
        return
    import torch
    import torch.distributed as td

    with torch.no_grad():
        for m in modules:
            tensors = list(m.state_dict().values())
            flat = _flat(tensors)
            td.broadcast(flat, src=0)
            _unflat(flat, tensors)


def state_spread(*modules) -> float:
    """Largest difference of any parameter or buffer entry between ranks; 0.0 while the ranks are in sync."""
    if not ACTIVE:
        return 0.0
    import torch
    import torch.distributed as td

    with torch.no_grad():
        lo = _flat([t for m in modules for t in m.state_dict().values()])
        hi = lo.clone()
        td.all_reduce(lo, op=td.ReduceOp.MIN)
        td.all_reduce(hi, op=td.ReduceOp.MAX)
        return float((hi - lo).max())


def check_sync(it: int, *modules) -> None:
    """Report (and repair from rank 0) any difference between the ranks' networks; there should be none."""
    spread = state_spread(*modules)
    if spread != 0.0:  # e.g. GPUs of different models rounding differently
        if is_main():
            print(f"[dist] it {it}: networks differ between ranks by up to {spread:.3g}; copying rank 0's", flush=True)
        broadcast_state(*modules)
