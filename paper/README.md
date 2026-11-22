# Draft Paper

`main.tex` is the draft. It needs no bibliography tool: references sit inside the file.

## Files

| Path | Content |
|------|---------|
| `main.tex` | Paper source |
| `table_acc.tex`, `table_age.tex` | Result tables, generated |
| `data/*.dat` | Figure data, generated |

Regenerate the tables and data from `results/registers/` with `python experiments/export_paper_data.py` from the repo root. Do not edit them by hand.

## Compile on Overleaf

1. On Overleaf, choose New Project, then Upload Project.
2. Zip the `paper/` folder and upload the zip. Keep `data/` as a subfolder.
3. Open Menu and set Compiler to pdfLaTeX and TeX Live version to 2023 or later.
4. Set the main document to `main.tex` and click Recompile.

## Compile locally

With a TeX distribution that includes pgfplots (TeX Live or MiKTeX), run from this folder:

```
pdflatex main.tex
pdflatex main.tex
```

The second pass resolves cross-references.
