# Demo 11 — Ridge Regression: Buying Accuracy with Bias

**Source:** CS229 PS2.2 (bias and variance of the ridge and OLS estimators)
**Sim:** `core/sims/ridge.py` · **Scene:** `scenes/ridge_bias_variance.py` · **Web:** `web/ridge.html`

## Load-bearing math

- `ŵ_λ = (XᵀX + λI)⁻¹ Xᵀ y`, `y = Xw* + ε`, `ε ~ N(0, σ²I)`.
- In the eigenbasis of `XᵀX = Σ γᵢ vᵢvᵢᵀ`:
  `E[ŵ_λ] − w* = −λ Σ (vᵢᵀw*)/(γᵢ+λ) vᵢ` ⇒ `bias²(λ) = λ² Σ (vᵢᵀw*)²/(γᵢ+λ)²`
  `Var(λ) = σ² Σ γᵢ/(γᵢ+λ)²`.
- λ=0 (OLS): zero bias, variance `σ² Σ 1/γᵢ` — huge when some γᵢ is small.
- MSE(λ) = bias² + variance is U-shaped, and `MSE′(0) < 0` always (σ² > 0):
  **some ridge always helps** — the PS2.2(d) punchline.

## Arc

1. **Intuition:** OLS is a dart thrower with perfect aim and a shaky hand.
   Ridge steadies the hand by aiming slightly off-target on purpose.
2. **Breakdown:** the decomposition MSE = ‖bias‖² + tr Var; per-eigenmode shrinkage γ/(γ+λ).
3. **Visual proof:** clouds of estimates tightening and drifting as λ grows; the U-curve;
   the derivative at 0 is negative — λ=0 is never the bottom.

## Beats

| # | On screen | Feeds from |
|---|-----------|-----------|
| 1 | w-space bullseye at w*; 60 OLS estimates rain in — centered, wide | `estimate_cloud(0)` |
| 2 | λ dial: cloud tightens AND drifts; E[ŵ] cross drifts off the bullseye | clouds for λ grid |
| 3 | MathTex: MSE = ‖E ŵ − w*‖² + E‖ŵ − E ŵ‖² (brace: bias² + variance) | — |
| 4 | U-curve: bias² rising, variance falling, MSE U; λ* dot; zoom chip: slope at λ=0 is negative | `curves(lams)` |
| 5 | Eigen-shrinkage: two bars γᵢ/(γᵢ+λ) — the weak-signal direction is shrunk first and hardest (callback to Demo 1/3) | `shrinkage(lam)` |
| 6 | End card: PS2.2(d) — "for every σ² > 0 there is a λ > 0 strictly better than OLS" | — |

## Web interactive

λ slider (log): left panel w-space cloud (Monte Carlo, resampleable) with w* bullseye and
E[ŵ] cross; right panel U-curve (bias², variance, MSE) with current-λ marker and λ* flag;
σ and conditioning sliders reshape everything; readouts cross-checked against closed form.
