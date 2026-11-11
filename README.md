# Transformers Cannot Overwrite

Research code for typed forget gates: an attention mechanism in which a later token suppresses specific earlier tokens, so a context can hold registers that overwrite.

```
logit(i, j) = q_i . k_j - b_j . (S_i - S_j)      S = running sum of kill vectors
```

## Layout

| Path | Content |
|------|---------|
| `src/tco/attention.py` | Gated attention: plain, scalar gate, typed gate |
| `src/tco/model.py` | Small transformer for the register task |
| `src/tco/tasks.py` | K-register stream generator |
| `src/tco/train.py` | Training and evaluation entry point |
| `experiments/` | Sweep scripts |
| `results/` | Saved run logs (JSON) |
| `theory/` | Proof notes |
| `notes/` | Prior-art notes |
| `PROGRESS.md` | Project tracker |

## Setup

```
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
```

## Run

```
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe -m tco.train --gate typed --keys 8 --out results/registers/demo.json
.venv/Scripts/python.exe experiments/registers_sweep.py
```
