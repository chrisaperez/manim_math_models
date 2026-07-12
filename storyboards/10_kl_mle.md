# Demo 10 — KL Divergence & Maximum Likelihood

**Source:** CS229 PS4.3 (nonnegativity, chain rule, KL ↔ MLE)
**Sim:** `core/sims/kl_mle.py` · **Scene:** `scenes/kl_mle.py` · **Web:** `web/kl_mle.html`

## Load-bearing math

- `KL(P‖Q) = Σ P(x) log(P(x)/Q(x)) ≥ 0`, with equality iff P = Q (Jensen on the concave log).
- Not a distance: `KL(P‖Q) ≠ KL(Q‖P)` in general.
- Chain rule: `KL(P(X,Y)‖Q(X,Y)) = KL(P(X)‖Q(X)) + KL(P(Y|X)‖Q(Y|X))`.
- The bridge (PS4.3(c)): with empirical distribution P̂,
  `argmin_θ KL(P̂‖P_θ) = argmax_θ (1/n)Σ log P_θ(x⁽ⁱ⁾)` — the two objectives differ by the
  constant −H(P̂), so their curves are mirror images and extremize at the same θ.

## Arc

1. **Intuition:** how wrong is it to believe Q when the world runs on P? Pay
   log(P/Q) per observation, on average.
2. **Breakdown:** why ≥ 0 (Jensen); why the direction matters; how a joint splits.
3. **Visual proof:** the KL(θ) curve and the log-likelihood(θ) curve are the same
   curve upside down — fitting by ML *is* minimizing KL from the data.

## Beats

| # | On screen | Feeds from |
|---|-----------|-----------|
| 1 | Two bar distributions P (teal), Q (yellow); KL(P‖Q) number; swap → different number (asymmetry) | `kl(p, q)` |
| 2 | Jensen: log curve, chord below; E[log Z] ≤ log E[Z] ⇒ −KL ≤ 0 morph | — |
| 3 | Chain rule: joint heat table splits into marginal bars × conditional rows; three KL numbers add exactly | `chain_rule_parts` |
| 4 | Samples rain into a histogram P̂; family P_θ = Binomial(5, θ) sweeps θ; two meters lockstep: KL(P̂‖P_θ) falls as loglik rises; both extremize at θ = mean/5 | `kl_curve`, `loglik_curve` |
| 5 | Mirror shot: both curves on shared axes (loglik flipped), identical shape, same argmin; `KL(P̂‖P_θ) = −H(P̂) − avg loglik` badge | identity check |
| 6 | End card: "training on cross-entropy = this picture" | — |

## Web interactive

Panel A: drag bars of P and Q (6 outcomes, auto-renormalized); live KL both directions;
equality detector. Panel B: hidden true θ*; draw n samples → histogram P̂; θ slider with
KL(P̂‖P_θ) and avg-loglik curves sharing an axis (mirrored), argmin/argmax marker = sample
mean/5; "more samples" button shows the estimate tightening toward θ*.
