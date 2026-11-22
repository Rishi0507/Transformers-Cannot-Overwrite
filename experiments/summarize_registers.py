"""Print mean accuracy per (keys, gate) from results/registers, across seeds."""
import json
import sys
from collections import defaultdict
from pathlib import Path


def mean(rows):
    cols = list(zip(*rows))
    return [round(sum(x for x in c if x is not None) / max(1, sum(x is not None for x in c)), 3)
            if any(x is not None for x in c) else None for c in cols]


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results/registers")
    groups = defaultdict(list)
    for f in sorted(folder.glob("*.json")):
        r = json.loads(f.read_text())
        c = r["config"]
        groups[(c["layers"], c["heads"], c["keys"], c["gate"])].append(r)
    for (layers, heads, keys, gate), runs in sorted(groups.items()):
        lens = list(runs[0]["eval"])
        print(f"\nL{layers} H{heads} keys={keys} gate={gate} seeds={len(runs)}")
        print("  acc by test length", dict(zip(lens, mean([[r["eval"][T]["acc"] for T in lens] for r in runs]))))
        for T in (lens[0], lens[-1]):
            print(f"  T={T} by depth", mean([r["eval"][T]["by_depth"] for r in runs]))
            print(f"  T={T} by age  ", mean([r["eval"][T]["by_age"] for r in runs]))
