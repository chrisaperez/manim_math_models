# Demo 1 — GD Convergence, Learning Rates & Hessian Curvature

**Source:** CS229 PS1.1 (convergence ⟺ α < 2/β_max), CS205L HW9.5 (GD vs condition number)
**Sim:** `core/sims/gradient_descent.py` · **Scene:** `scenes/gd_convergence.py` · **Web:** `web/gd_convergence.html`

## Load-bearing math

- Objective `J(θ) = ½ θᵀAθ`, A symmetric PSD with eigenvalues `0 < λ_min ≤ … ≤ λ_max = β_max`.
- GD iterate: `θ_{t+1} = θ_t − α∇J = (I − αA)θ_t`.
- Eigenbasis `A = QΛQᵀ`, coords `c = Qᵀθ`: each mode is independent, `c_i(t) = (1 − αλ_i)ᵗ c_i(0)`.
- Convergence ⟺ `|1 − αλ_i| < 1` for all i ⟺ `α < 2/λ_max`. Rate limited by slow mode `|1 − αλ_min|`.
- Optimal fixed step `α* = 2/(λ_min+λ_max)` gives rate `(κ−1)/(κ+1)`, κ = λ_max/λ_min.

## Arc

### 1. Intuition (~0:00–0:40)
A ball rolling downhill should just... arrive. Why does the same step size that glides down one
valley explode out of another? The answer is curvature — and it's directional.

### 2. Mathematical breakdown (~0:40–2:00)
Vectorized GD looks coupled and opaque; the update matrix `(I − αA)` mixes coordinates.
The pset asks: for what α does this converge? Scalar case is easy (`|1−αa|<1`); the
d-dimensional case seems hard — until you rotate into the eigenbasis and it becomes d scalar
problems running in parallel.

### 3. Visual proof (~2:00–4:00)
Split screen. Left: real coordinates, curved coupled trajectory. Right: eigen-coordinates, two
independent 1D bounces `(1−αλ_i)ᵗ`. Dial α upward; the mode with λ_max hits `|1−αλ_max| = 1`
first and slingshots — divergence is *always* the steep mode's fault. Threshold `2/β_max`.

## Beat table

| # | t | On screen | Feeds from |
|---|-----|-----------|-----------|
| 1 | 0:00 | 3D valley `J(θ)=½θᵀAθ` (ThreeDAxes surface), ball drops in, glides to min with small α | `gd_path(A, θ0, α_safe)` |
| 2 | 0:25 | Same valley, α slightly larger → ball ricochets out. Title: "same bowl, same start — why?" | `gd_path(A, θ0, α_diverge)` |
| 3 | 0:45 | Top-down contour view. `MathTex` morph: `θ_{t+1} = θ_t − α∇J(θ_t)` → `θ_{t+1} = (I−αA)θ_t` | — |
| 4 | 1:10 | Scalar warm-up inset: `c_{t+1} = (1−αλ)c_t`, geometric decay bars for `|1−αλ|<1`, flip sign past 1/λ, blow-up past 2/λ | `scalar_modes(λ, α)` |
| 5 | 1:40 | Rotated-A contours (ellipse tilted 30°): trajectory curves, coordinates coupled. `TransformMatchingTex`: `A = QΛQᵀ`, grid rotates by `Qᵀ` → ellipse axis-aligned | `eigen_frame(A)` |
| 6 | 2:10 | Split screen: left standard coords (curved path), right eigen-coords (two 1D bounces along axes). Same iterates, linked dots | `gd_path` + `to_eigen` |
| 7 | 2:50 | α-dial ValueTracker sweeps: per-mode factor bars `1−αλ_i` slide toward −1; at `α = 2/λ_max` steep-mode bar hits −1 → path slingshots out of valley (camera shake beat) | `mode_factors(A, α)` |
| 8 | 3:30 | Zoom on threshold: `MathTex` conclusion `α < 2/β_max`, β_max = largest Hessian eigenvalue; κ chip shows why the *slow* mode sets the speed even when safe | `convergence_threshold(A)` |
| 9 | 3:50 | End card: linear-regression tie-in `H = (1/n)XᵀX` (PS1(e)) — the data itself sets your learning-rate budget | — |

## Web interactive

Controls: α slider (log scale, `2/β_max` tick-marked in red), κ slider (reshapes A), rotation-angle
slider for Q, start-point drag. Views: contour + live trajectory (fading trail), per-mode error
bars `|c_i(t)|`, factor gauges `1−αλ_i` with the ±1 danger zone. Divergence flashes the border red.
