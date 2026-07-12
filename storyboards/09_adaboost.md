# Demo 9 — AdaBoost: Failing Better Every Round

**Source:** CS229 PS3.3 (weights, stump selection, coefficients), PS3.4 (exponential-loss bound)
**Sim:** `core/sims/adaboost.py` · **Scene:** `scenes/adaboost.py` · **Web:** `web/adaboost.html`

## Load-bearing math

- Round t: pick stump `h_t` minimizing weighted error `ε_t = Σ w_i 𝟙[h_t(x⁽ⁱ⁾) ≠ y⁽ⁱ⁾]`;
  coefficient `α_t = ½ ln((1−ε_t)/ε_t)`.
- Reweight `w_i ← w_i·exp(−α_t y⁽ⁱ⁾h_t(x⁽ⁱ⁾)) / Z_t`, `Z_t = 2√(ε_t(1−ε_t))`.
- The self-erasing property: under the NEW weights, `h_t` has weighted error exactly ½ —
  yesterday's stump is worthless tomorrow, forcing diversity.
- The bound chain (PS3.4): `𝟙[H(x)≠y] ≤ e^{−y f(x)}` ⇒ train error ≤ `Π_t Z_t`
  = `Π_t 2√(ε_t(1−ε_t))` ≤ `exp(−2 Σ_t γ_t²)`, `γ_t = ½ − ε_t`.

## Arc

1. **Intuition:** one stump is barely better than a coin. Boosting: after each round,
   reweight so your mistakes matter more — then hire another specialist.
2. **Breakdown:** why ½ln((1−ε)/ε)? why do the weights make h_t exactly 50-50 afterwards?
3. **Visual proof:** dots grow when misclassified; stumps stack into a jagged committee
   boundary; the train-error staircase hugs down under Π Z_t under exp(−2Σγ²).

## Beats

| # | On screen | Feeds from |
|---|-----------|-----------|
| 1 | 2D +/− dataset, equal dot sizes (weights); first stump line + shaded halves; misclassified flash | `best_stump`, `boost_round` |
| 2 | Weights morph: wrong dots inflate, right dots deflate; α₁, ε₁ chips; gauge: h₁'s weighted error under new weights = exactly 0.500 | round data |
| 3 | Rounds 2–3 fast: new stumps chase the heavy dots | `run(T=3)` |
| 4 | Committee: decision-region heat of sign(Σα_t h_t); staircase of train error over T | `predict_grid` |
| 5 | Bound chain morph: 𝟙 ≤ e^{−yf} → ΠZ_t → exp(−2Σγ²); chart: error ≤ ΠZ ≤ exp curve | `bound_history` |
| 6 | End card: boosting = greedy coordinate descent on E[e^{−yf}] | — |

## Web interactive

Click-to-add ± points (or preset); "boost one round" button: shows chosen stump, α, ε,
dot sizes = weights; committee heat backdrop; chart with train-error staircase, ΠZ_t and
exp(−2Σγ²) curves; hover a dot for its weight; reset/preset buttons.
