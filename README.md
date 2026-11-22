# Transformers Cannot Overwrite

Research code for typed forget gates in softmax attention. A later token suppresses only those earlier tokens whose type matches its kill vector, so one head can hold several registers that overwrite.

```
logit(i, j) = q_i . k_j - b_j . (S_i - S_j)      S_t = sum of kill vectors u_1, ..., u_t
```

Plain attention is the case `u = 0`. A scalar forget gate (FoX) is the case `r = 1`, `b = 1`.

## Quick Start (about five minutes on a CPU laptop)

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU needed.

1. Create the environment and install the package (PyTorch is the slow part):

   ```
   uv venv --python 3.12 .venv
   uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
   ```

   On Linux or macOS, use `.venv/bin/python` in place of `.venv/Scripts/python.exe` here and below. On Linux, add `--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple` to skip the large CUDA build of PyTorch.

2. Run the unit tests (about 10 seconds):

   ```
   .venv/Scripts/python.exe -m pytest
   ```

   Expected: `10 passed`.

3. Run a smoke version of the register sweep (about 2 minutes). It trains three tiny models (plain, scalar gate, typed gate) for 100 steps and writes to a scratch folder:

   ```
   .venv/Scripts/python.exe experiments/registers_sweep.py --out-dir scratch/smoke --seeds 0 --keys 8 --steps 100 --test-lens 128 256 --eval-batches 2
   .venv/Scripts/python.exe experiments/summarize_registers.py scratch/smoke
   ```

   100 steps is too few to learn the task, so accuracy stays near chance (1/16). The smoke run checks the pipeline, not the result.

## Full Sweep

The v1 sweep trains 27 one-layer models: three attention types, `K` in {2, 8, 32} keys, three seeds. Each run takes about 15 minutes on a laptop CPU, so the full sweep takes about 7 hours. The script skips runs whose output file exists.

```
.venv/Scripts/python.exe experiments/registers_sweep.py
.venv/Scripts/python.exe experiments/summarize_registers.py
.venv/Scripts/python.exe experiments/export_paper_data.py
```

`results/registers/` holds 17 finished runs. `export_paper_data.py` writes the figure data and tables in `paper/` from those logs.

One run with your own settings:

```
.venv/Scripts/python.exe -m tco.train --gate typed --keys 8 --out scratch/demo.json
```

`python -m tco.train --help` lists every option.

## Layout

| Path | Content |
|------|---------|
| `src/tco/attention.py` | Gated attention: plain, scalar gate, typed gate |
| `src/tco/model.py` | Small transformer for the register task |
| `src/tco/tasks.py` | K-register stream generator |
| `src/tco/train.py` | Training and evaluation entry point |
| `tests/` | Unit tests |
| `experiments/` | Sweep, summary, and paper-data scripts |
| `results/registers/` | Saved run logs (JSON) |
| `theory/proofs.md` | Proofs of Theorems 1 to 4, with assumptions and a review log |
| `paper/` | Draft paper (LaTeX) |
| `notes/` | Prior-art notes, partial |
| `PROGRESS.md` | Project tracker |

## Status

The theorems cover one attention layer and have not yet had an independent check. Deep-model versions are conjectures. See `theory/proofs.md`, Section 7.
