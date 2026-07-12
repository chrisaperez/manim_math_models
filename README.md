# Manim Math Models

3Blue1Brown-style explanations of the hardest concepts from CS229 (Machine Learning),
CS246 (Mining Massive Datasets), and CS205L (Numerical Methods), extracted from my
problem sets. Each demo is one topic told twice from the same verified math:

- a cinematic **Manim video** (`scenes/` → `media/final/<topic>.mp4`)
- an **interactive web page** with live controls (`web/<topic>.html`)

## Layout

| Path | What |
|---|---|
| `core/sims/` | pure-numpy simulation engines — no manim imports, fully pytest-covered |
| `core/style.py` | shared 3b1b palette + manim helpers |
| `scenes/` | manim scene modules, one per demo |
| `storyboards/` | 3-part-arc plans (Intuition → Breakdown → Visual Proof) with beat tables |
| `scripts/` | read-along narration, one per video, timestamped to the animation |
| `web/` | self-contained interactive pages + shared design system (`assets/`) |
| `tests/` | the math, proven before it is drawn |

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# LaTeX (for MathTex): BasicTeX + packages, see storyboards/../plan; latex must be on PATH
```

## Commands

```sh
.venv/bin/python -m pytest tests/ -q            # verify the numerics
.venv/bin/manim render -ql --media_dir media scenes/gd_convergence.py EigenSplit   # draft
.venv/bin/manim render -qh --media_dir media scenes/gd_convergence.py ValleyOpener EigenSplit  # final
python3 serve.py 8322                           # browse the interactives (no-cache static server)
```

## Demos

Start at the gallery: `web/index.html` (videos are symlinked at `web/videos → media/final`).

| # | Topic | Source | Video | Interactive | Script |
|---|---|---|---|---|---|
| 1 | GD convergence & the curvature cliff | CS229 PS1.1 · CS205L HW9.5 | `media/final/gd_convergence.mp4` | `web/gd_convergence.html` | `scripts/01_gd_convergence.md` |
| 2 | Backprop as batched matrix flow | CS229 PS2.4 | `media/final/backprop_flow.mp4` | `web/backprop_flow.html` | `scripts/02_backprop_flow.md` |
| 3 | SVD vs PCA | CS229 PS4.1 · CS246 HW2.1 · CS205L HW3 | `media/final/svd_vs_pca.mp4` | `web/svd_vs_pca.html` | `scripts/03_svd_vs_pca.md` |
| 4 | GNN message passing & depth | CS246 HW4.1 | `media/final/gnn_depth.mp4` | `web/gnn_depth.html` | `scripts/04_gnn_depth.md` |
| 5 | Bloom filters & count-min sketch | CS246 HW4.4 | `media/final/bloom_cms.mp4` | `web/bloom_cms.html` | `scripts/05_bloom_cms.md` |
| 6 | PageRank random surfer | CS246 HW3.1–3.2 | `media/final/pagerank.mp4` | `web/pagerank.html` | `scripts/06_pagerank.md` |
| 7 | MinHash & LSH | CS246 HW1.3–1.4 | `media/final/minhash_lsh.mp4` | `web/minhash_lsh.html` | `scripts/07_minhash_lsh.md` |
| 8 | Newton vs GD & conditioning | CS205L HW7–9 | `media/final/newton_conditioning.mp4` | `web/newton_conditioning.html` | `scripts/08_newton_conditioning.md` |
| 9 | AdaBoost & the exponential-loss bound | CS229 PS3.3–3.4 | `media/final/adaboost.mp4` | `web/adaboost.html` | `scripts/09_adaboost.md` |
| 10 | KL divergence & maximum likelihood | CS229 PS4.3 | `media/final/kl_mle.mp4` | `web/kl_mle.html` | `scripts/10_kl_mle.md` |
| 11 | Ridge bias–variance | CS229 PS2.2 | `media/final/ridge.mp4` | `web/ridge.html` | `scripts/11_ridge.md` |
