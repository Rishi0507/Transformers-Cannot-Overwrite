"""Phase 0a sweep: K-register task, plain vs scalar vs typed gates. Skips finished runs."""
import itertools
import sys
from pathlib import Path

from tco.train import parser, run

OUT = Path("results/registers")
GATES = ["none", "scalar", "typed"]
KEYS = [2, 8, 32]
SEEDS = [0, 1, 2]

if __name__ == "__main__":
    extra = sys.argv[1:]
    for seed, keys, gate in itertools.product(SEEDS, KEYS, GATES):
        out = OUT / f"L1H2_k{keys}_{gate}_s{seed}.json"
        if out.exists():
            continue
        args = ["--gate", gate, "--keys", str(keys), "--seed", str(seed), "--out", str(out)] + extra
        print("RUN", " ".join(args), flush=True)
        run(parser().parse_args(args))
