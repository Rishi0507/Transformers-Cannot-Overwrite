"""Phase 0a sweep: K-register task, plain vs scalar vs typed gates. Skips finished runs."""
import argparse
import itertools
from pathlib import Path

from tco.train import parser, run

GATES = ["none", "scalar", "typed"]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/registers")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--keys", type=int, nargs="+", default=[2, 8, 32])
    sweep, extra = p.parse_known_args()
    for seed, keys, gate in itertools.product(sweep.seeds, sweep.keys, GATES):
        out = Path(sweep.out_dir) / f"L1H2_k{keys}_{gate}_s{seed}.json"
        if out.exists():
            continue
        args = ["--gate", gate, "--keys", str(keys), "--seed", str(seed), "--out", str(out)] + extra
        print("RUN", " ".join(args), flush=True)
        run(parser().parse_args(args))
