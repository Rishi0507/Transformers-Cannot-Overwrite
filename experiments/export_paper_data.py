"""Write figure data and the results table for paper/ from results/registers. Stdlib only."""
import json
import sys
from collections import defaultdict
from pathlib import Path

DEPTH_LABELS = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+"]
AGE_LABELS = [r"$\le 8$", "9-32", "33-128", "129-512", "513-2048", "2049+"]
GATE_NAME = {"none": "plain", "scalar": "scalar gate", "typed": "typed gate"}


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    return m, max(xs) - m, m - min(xs), len(xs)


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/registers")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "paper")
    (out / "data").mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for f in sorted(src.glob("*.json")):
        r = json.loads(f.read_text())
        groups[(r["config"]["keys"], r["config"]["gate"])].append(r)

    # accuracy by depth bucket at the longest test length
    for (keys, gate), runs in groups.items():
        T = runs[0]["config"]["test_lens"][-1]
        lines = ["x mean eplus eminus seeds"]
        for b in range(len(DEPTH_LABELS)):
            s = stats([r["eval"][str(T)]["by_depth"][b] for r in runs])
            if s:
                lines.append(f"{b} {s[0]:.4f} {s[1]:.4f} {s[2]:.4f} {s[3]}")
        (out / "data" / f"depth_k{keys}_{gate}.dat").write_text("\n".join(lines) + "\n")

    # overall accuracy by test length
    lens = groups[next(iter(groups))][0]["config"]["test_lens"]
    rows = []
    for (keys, gate) in sorted(groups, key=lambda kg: (kg[0], list(GATE_NAME).index(kg[1]))):
        runs = groups[(keys, gate)]
        cells = []
        for T in lens:
            m, ep, em, _ = stats([r["eval"][str(T)]["acc"] for r in runs])
            cells.append(f"{m:.3f} ({m - em:.3f}, {m + ep:.3f})")
        rows.append(f"{keys} & {GATE_NAME[gate]} & {len(runs)} & " + " & ".join(cells) + r" \\")
    head = "$K$ & head & seeds & " + " & ".join(f"$T={T}$" for T in lens) + r" \\"
    table = [r"\begin{tabular}{rlr" + "c" * len(lens) + "}", r"\toprule", head, r"\midrule", *rows,
             r"\bottomrule", r"\end{tabular}"]
    (out / "table_acc.tex").write_text("\n".join(table) + "\n")
    # accuracy by age of the latest write at the longest test length
    T = lens[-1]
    rows = []
    for (keys, gate) in sorted(groups, key=lambda kg: (kg[0], list(GATE_NAME).index(kg[1]))):
        runs = groups[(keys, gate)]
        cells = []
        for b in range(4):
            s = stats([r["eval"][str(T)]["by_age"][b] for r in runs])
            cells.append(f"{s[0]:.3f}" if s else "--")
        rows.append(f"{keys} & {GATE_NAME[gate]} & " + " & ".join(cells) + r" \\")
    table = [r"\begin{tabular}{rlcccc}", r"\toprule", "$K$ & head & " + " & ".join(AGE_LABELS[:4]) + r" \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (out / "table_age.tex").write_text("\n".join(table) + "\n")
    print(f"wrote {len(groups)} depth files and two tables from {sum(map(len, groups.values()))} runs")
