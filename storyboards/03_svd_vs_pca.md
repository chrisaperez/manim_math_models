# Demo 3 — SVD vs PCA: One Decomposition, Two Stories

**Source:** CS229 PS4.1 (PCA + SVD relationship), CS246 HW2.1 (MᵀM eigenstructure), CS205L HW3.2–3.4 (SVD properties, pseudoinverse)
**Sim:** `core/sims/svd_pca.py` · **Scene:** `scenes/svd_vs_pca.py` · **Web:** `web/svd_vs_pca.html`

## Load-bearing math

- Projection onto unit u: `f_u(x) = (uᵀx)u`. Pythagoras per point: `‖x‖² = ‖f_u(x)‖² + ‖x − f_u(x)‖²`.
- Σ‖x⁽ⁱ⁾‖² is fixed ⇒ minimizing reconstruction error ≡ maximizing Σ(uᵀx⁽ⁱ⁾)² = uᵀ(Σᵢ x⁽ⁱ⁾x⁽ⁱ⁾ᵀ)u — the
  variance along u. Optimum: top eigenvector of the covariance `MᵀM` (rows of M = centered data).
- `MᵀM = (UΣVᵀ)ᵀ(UΣVᵀ) = VΣ²Vᵀ` ⇒ PCA directions = right singular vectors V; `λ_i = σ_i²`.
- `M = UΣVᵀ`: every matrix is rotation → axis-aligned scaling → rotation.

## Arc

### 1. Intuition (~0:00–0:40)
Two camps: "PCA finds the directions of most variance" and "SVD factors any matrix into
rotate–stretch–rotate." Same V, same math. Why?

### 2. Mathematical breakdown (~0:40–2:00)
The pset asks to minimize mean squared projection error. Each point makes a right triangle:
hypotenuse ‖x‖ fixed; one leg is what the projection keeps, the other what it loses.
Minimizing loss = maximizing keep. That maximization is a Rayleigh quotient on MᵀM.

### 3. Visual proof (~2:00–4:30)
Sweep the line u through the cloud: triangles reshape, keep/lose bars trade off with constant sum.
Lock onto the eigenvector. Then flip the story: unit circle pushed through M shows Vᵀ (rotate),
Σ (stretch), U (rotate) — and the stretch axes are exactly the variance axes. Close the loop with
`MᵀM = VΣ²Vᵀ` morph.

## Beat table

| # | t | On screen | Feeds from |
|---|-----|-----------|-----------|
| 1 | 0:00 | 2D correlated cloud fades in; a candidate line u through origin rotates slowly | `make_cloud(seed)` |
| 2 | 0:30 | One highlighted point: drop perpendicular to u; right triangle with legs labeled `‖f_u(x)‖` (keep) and `‖x−f_u(x)‖` (lose); `MathTex` Pythagoras | `project(cloud, u)` |
| 3 | 1:10 | All points project; two stacked bars: total keep vs total lose, sum pinned (constant bar) as u sweeps 180° | `sweep_variance(cloud)` |
| 4 | 1:50 | Bars extremize together at the same angle — "min error IS max variance"; u snaps to that direction, cloud breathes | `principal_dir(cloud)` |
| 5 | 2:20 | Rayleigh morph: `Σ(uᵀx)² = uᵀ(MᵀM)u` → "maximize over ‖u‖=1" → eigenvector statement | — |
| 6 | 2:50 | Act II: unit circle + grid; apply M in three stages — `Vᵀ` rotates grid, `Σ` stretches along axes (circle→ellipse), `U` rotates ellipse into place; stage badges | `svd_stages(M)` |
| 7 | 3:40 | Ellipse axes overlaid on the data cloud: singular directions = variance directions; σ₁, σ₂ arrows vs point spread | `svd_stages` + cloud |
| 8 | 4:10 | Algebra close: `MᵀM = (UΣVᵀ)ᵀ(UΣVᵀ)` → `VΣᵀΣVᵀ` term-cancel animation (UᵀU=I flashes) → `VΣ²Vᵀ`; badge `λᵢ = σᵢ²` (CS246 HW2 1(d)) | — |
| 9 | 4:30 | End card: pseudoinverse `M⁺ = VΣ⁺Uᵀ` — run the movie backwards (CS205L 3.2) | — |

## Web interactive

Panel A: draggable projection line through a 2D cloud (regenerate/seed button); live keep/lose
bars with pinned sum; ghost triangle on nearest point to cursor; "snap to PC1" button.
Panel B: editable 2×2 matrix M (number inputs + presets); unit circle animated through
Vᵀ → Σ → U with a stage scrubber; σ values and `λᵢ = σᵢ²` readout cross-checked live.
