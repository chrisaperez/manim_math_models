# AdaBoost — Failing Better Every Round — read-along script

**Video:** `media/final/adaboost.mp4` · **Length:** 0:35
**Source:** CS229 PS3.3–3.4

---

### One weak stump (0:00–0:10)

**[0:00]** Here's a dataset no single straight cut can separate — pluses and minuses in
opposite corners. Every point's size is its *weight*; they all start equal. We pick the
best decision stump: one axis-aligned split, and it's barely better than a coin flip.
Its weighted error is `ε₁`, about 0.29, and its vote in the final committee is
`α₁ = ½ ln((1−ε₁)/ε₁)` — bigger votes for smaller errors.

---

### The reweighting trick (0:10–0:18)

**[0:10]** Now the magic. We reweight the data: multiply each point's weight by
`exp(−α·y·h(x))`, so the points this stump got *wrong* swell up, and the ones it got
right shrink. Next round's learner will be forced to focus on today's mistakes.

**[0:15]** And here's the property that makes boosting tick: under the *new* weights, the
stump we just used has weighted error exactly one-half. It's become a pure coin flip — no
longer worth anything. Yesterday's expert is today's random guesser, which guarantees the
next stump does something genuinely different.

---

### Building the committee (0:18–0:27)

**[0:18]** Rounds two and three: each new stump chases the now-heavier points, cutting
the plane from a fresh angle. Combine them by weighted vote —
`H(x) = sign(Σ α_t h_t(x))` — and the decision region is no longer a single line but a
jagged, capable boundary. The training error is already dropping.

---

### The guarantee (0:27–0:35)

**[0:27]** Why does this always work? The problem-set chain. The zero-one error is bounded
by the exponential loss, `𝟙[H(x)≠y] ≤ e^(−yf(x))`; that loss telescopes exactly to the
product `Π Z_t = Π 2√(ε_t(1−ε_t))`; and that product is at most `exp(−2 Σ γ_t²)`, where
`γ_t = ½ − ε_t` is each round's edge over chance.

**[0:31]** On the chart, the training error sits below the product-of-Z curve, which sits
below the exponential bound — each squeezed under the one above it, driven to zero
geometrically. AdaBoost is, underneath, greedy coordinate descent on that exponential loss.

---

**Recap.** AdaBoost combines weak learners by reweighting the data so each new learner
attacks the previous one's mistakes — and the reweighting makes the last learner exactly
chance, forcing diversity. The training error is provably trapped under `Π Z_t ≤
exp(−2Σγ²)`, so as long as every round beats a coin flip, the ensemble error vanishes.
