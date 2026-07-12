# KL Divergence & Maximum Likelihood — read-along script

**Video:** `media/final/kl_mle.mp4` · **Length:** 0:42
**Source:** CS229 PS4.3

---

### The price of a wrong belief (0:00–0:12)

**[0:00]** Suppose the world really runs on distribution `P`, but you believe `Q`. How
wrong is that? On average, every observation costs you `log P(x)/Q(x)` — and summing that
up is the KL divergence, `KL(P‖Q) = Σ P log(P/Q)`. Here are two distributions as bars, and
their KL is a single number.

**[0:07]** But swap the roles — `KL(Q‖P)` — and you get a *different* number. KL is not
symmetric, and it's not a distance. The direction matters: being surprised by the truth
is not the same cost as being surprised by your model.

---

### Why it's never negative (0:12–0:22)

**[0:12]** KL is always at least zero, and Jensen's inequality is the reason. The log is
concave, so a chord always lies below the curve — `E[log Z] ≤ log E[Z]`. Apply that to
`−KL`: it equals `E_P[log(Q/P)]`, which is at most `log E_P[Q/P] = log 1 = 0`. So `KL ≥ 0`,
with equality only when `P` and `Q` are identical. You can never be *helped*, on average,
by believing the wrong distribution.

---

### The chain rule (0:22–0:28)

**[0:22]** KL also decomposes cleanly over joints. The divergence between two joint
distributions splits into the divergence of the marginals plus the divergence of the
conditionals — and the three numbers add exactly. Surprise about `(X, Y)` is surprise
about `X`, plus the surprise `Y` adds once you know `X`.

---

### The bridge to maximum likelihood (0:28–0:42)

**[0:28]** Now the payoff. Draw 400 samples from a binomial and form the empirical
distribution `P̂`. Sweep a model parameter `θ` and watch two meters. On the left, the
model bars chase the data histogram. On the right, two curves: the KL from the data to the
model, and the average log-likelihood.

**[0:36]** They are the *same curve, flipped* — one dips exactly where the other peaks —
because they differ only by the constant entropy of the data:
`KL(P̂‖P_θ) = −H(P̂) − avg log-likelihood`. So minimizing KL and maximizing likelihood are
the identical optimization, and both land on the same `θ` — here, exactly the sample mean
over five, the closed-form MLE. Training a model on cross-entropy *is* this picture.

---

**Recap.** KL divergence measures the average excess cost of encoding data from `P` using
a model `Q` — nonnegative by Jensen, asymmetric, and additive over joints. Fitting a model
by maximum likelihood is exactly minimizing KL from the empirical distribution, because the
two objectives differ only by the data's entropy, a constant in `θ`.
