# Gradient Descent & the Curvature Cliff — read-along script

**Video:** `media/final/gd_convergence.mp4` · **Length:** 1:09
**Source:** CS229 PS1.1, CS205L HW9.5

---

### Part 1 — The mystery (0:00–0:11)

**[0:00]** Here's a bowl-shaped loss surface, `J(θ) = ½ θᵀAθ`. A ball starts up on
the rim, and we run gradient descent — step downhill, over and over. With a learning
rate of 0.42, watch it glide gently to the bottom. This is what we expect optimization
to feel like.

**[0:04]** Now the *same bowl*, the *same starting point* — but we nudge the learning
rate up to 0.62. Suddenly the ball doesn't settle. It ricochets off the walls and
slings itself clean out of the valley. Same landscape, same start; one number changed,
and descent became divergence.

**[0:09]** The answer to *why* is curvature — and, crucially, curvature has a direction.

---

### Part 2 — Making it precise (0:11–0:38)

**[0:11]** Write one gradient-descent step out. `θ` becomes `θ − α∇J`, and because the
gradient of this quadratic is `Aθ`, that's the same as multiplying by the fixed matrix
`(I − αA)`. Every step is that one matrix applied again. So the whole question — does GD
converge? — is really: does repeatedly applying this matrix shrink the vector or grow it?

**[0:21]** The scalar version already tells the whole story. If the update is
`c ← (1 − αλ)c`, then the fate of `c` depends entirely on the size of `1 − αλ`. When
`αλ` is small, you get smooth monotone decay. Push it past 1 and the sign flips —
decay, but now oscillating. Push it past 2 and the factor's magnitude exceeds one, and
`c` blows up. Convergence in the scalar case is simply `|1 − αλ| < 1`, i.e. `α < 2/λ`.

**[0:34]** The full multidimensional problem *looks* coupled — the matrix `(I − αA)`
mixes all the coordinates together. The trick is to stop using the coordinates you were
handed.

---

### Part 3 — The eigenbasis reveals everything (0:40–1:00)

**[0:40]** Diagonalize: `A = QΛQᵀ`. Rotate the whole picture into the eigenbasis, and
the tilted elliptical contours snap axis-aligned. In these coordinates the coupling
vanishes — the d-dimensional walk becomes d independent scalar problems running side
by side, each one exactly the `(1 − αλᵢ)ᵗ` decay we just understood.

**[0:52]** So watch the split screen. On the left, the real trajectory curving through
the valley; on the right, two independent 1-D bounces along the eigen-axes. Same
iterates — just a change of basis. Now sweep the learning rate up. Each mode has a
gauge showing its factor `1 − αλᵢ`. The steep mode — the one with the *largest*
eigenvalue — is the first to reach −1. The instant its gauge crosses into the red zone,
that single direction explodes, and it drags the whole trajectory out with it.

---

### Part 4 — The theorem (1:00–1:09)

**[1:00]** That's the punchline: gradient descent converges if and only if
`α < 2/β_max`, where `β_max` is the largest eigenvalue of the Hessian. Divergence is
*always* the steepest direction's fault — it hits the cliff first. And even when you're
safely under the cliff, the *slowest* mode sets your speed, with a rate governed by the
condition number: `(κ−1)/(κ+1)`.

**[1:05]** One last connection: in linear regression the Hessian is `H = (1/n)XᵀX`. So
the eigenvalues that decide your learning-rate budget aren't abstract — they're a
property of your data.

---

**Recap.** Gradient descent is secretly a set of independent 1-D decays, one per
Hessian eigenvalue. The largest eigenvalue controls stability (the `2/β_max` cliff);
the smallest controls speed (the condition number). Coupling was an illusion of the
coordinate system.
