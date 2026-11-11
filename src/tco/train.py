import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tco.model import RegisterModel
from tco.tasks import make_batch

DEPTH_EDGES = [1, 2, 4, 8, 16, 32, 64]  # buckets: 1, 2, 3-4, 5-8, ..., 65+
AGE_EDGES = [8, 32, 128, 512, 2048]  # buckets: <=8, 9-32, ..., 2049+


def evaluate(model, cfg, T, n_batches, gen):
    model.eval()
    d_hit, d_tot = torch.zeros(len(DEPTH_EDGES) + 1), torch.zeros(len(DEPTH_EDGES) + 1)
    a_hit, a_tot = torch.zeros(len(AGE_EDGES) + 1), torch.zeros(len(AGE_EDGES) + 1)
    with torch.no_grad():
        for _ in range(n_batches):
            op, key, val, tgt, depth, age = make_batch(cfg.eval_batch, T, cfg.keys, cfg.vals, gen=gen)
            pred = model(op, key, val).argmax(-1)
            m = tgt >= 0
            ok = (pred == tgt)[m].float()
            for edges, x, hit, tot in ((DEPTH_EDGES, depth, d_hit, d_tot), (AGE_EDGES, age, a_hit, a_tot)):
                idx = torch.bucketize(x[m], torch.tensor(edges))
                hit.index_add_(0, idx, ok)
                tot.index_add_(0, idx, torch.ones_like(ok))
    model.train()

    def ratio(h, t):
        return [round((a / b).item(), 4) if b > 0 else None for a, b in zip(h, t)]

    return {
        "acc": round((d_hit.sum() / d_tot.sum()).item(), 4),
        "by_depth": ratio(d_hit, d_tot),
        "by_age": ratio(a_hit, a_tot),
        "reads": int(d_tot.sum().item()),
    }


def run(cfg):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = RegisterModel(
        cfg.keys, cfg.vals, dim=cfg.dim, layers=cfg.layers, heads=cfg.heads,
        gate=cfg.gate, rank=cfg.rank, pos=cfg.pos,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / cfg.steps))))
    )
    losses, t0 = [], time.time()
    for step in range(cfg.steps):
        T = int(torch.randint(cfg.train_len // 2, cfg.train_len + 1, (1,), generator=gen))
        op, key, val, tgt, _, _ = make_batch(cfg.batch, T, cfg.keys, cfg.vals, gen=gen)
        loss = F.cross_entropy(model(op, key, val).flatten(0, 1), tgt.flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == cfg.steps - 1:
            losses.append([step, round(loss.item(), 4)])
            print(f"step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)

    egen = torch.Generator().manual_seed(10_000 + cfg.seed)
    result = {
        "config": vars(cfg),
        "params": sum(p.numel() for p in model.parameters()),
        "train_seconds": round(time.time() - t0, 1),
        "loss": losses,
        "eval": {str(T): evaluate(model, cfg, T, cfg.eval_batches, egen) for T in cfg.test_lens},
        "depth_edges": DEPTH_EDGES,
        "age_edges": AGE_EDGES,
    }
    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(result, indent=1))
    return result


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--gate", default="none", choices=["none", "scalar", "typed"])
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--pos", default="rope", choices=["rope", "none"])
    p.add_argument("--keys", type=int, default=8)
    p.add_argument("--vals", type=int, default=16)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--train-len", type=int, default=128)
    p.add_argument("--test-lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    return p


if __name__ == "__main__":
    res = run(parser().parse_args())
    print(json.dumps(res["eval"], indent=1))
