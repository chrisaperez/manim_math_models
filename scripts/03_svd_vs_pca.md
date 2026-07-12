# SVD vs PCA — read-along script

**Video:** `media/final/svd_vs_pca.mp4` · **Length:** 0:56
**Source:** CS229 PS4.1, CS246 HW2.1, CS205L HW3

---

### Act I — The Pythagorean trade-off (0:00–0:30)

**[0:00]** Two camps describe the same math. One says PCA finds the direction of
greatest variance; the other says SVD factors any matrix into rotate–stretch–rotate.
We'll show they're the same statement. Start with a cloud of 2-D data and a candidate
projection line `u` through the origin.

**[0:08]** Pick one point and drop it onto the line. You get a right triangle. The
hypotenuse is the fixed length `‖x‖`. One leg is what the projection *keeps*; the other
leg is what it *loses*. By Pythagoras, `‖x‖² = ‖f_u(x)‖² + ‖x − f_u(x)‖²` — and since
the hypotenuse never changes, keep and lose are locked in a see-saw.

**[0:16]** Now sweep the line through every angle. Two bars track the totals: variance
kept in green, error lost in red. Their sum stays pinned, because it's just the total
energy of the cloud. So the angle that *maximizes* kept variance is exactly the angle
that *minimizes* reconstruction error. They're not two goals — they're one goal, seen
from two sides. Lock onto that direction: it's the first principal component, `v₁`.

**[0:26]** Written as an optimization, maximizing `Σ(uᵀx)²` is maximizing
`uᵀ(MᵀM)u` — a Rayleigh quotient — whose maximizer is the top eigenvector of `MᵀM`.
That's the bridge to SVD.

---

### Act II — What a matrix does (0:30–0:50)

**[0:30]** Second story. Take a unit circle with spokes and push it through the matrix
`M = UΣVᵀ`, one factor at a time. First `Vᵀ`: a rigid rotation, spinning the circle in
place. Then `Σ`: a stretch along the axes — the circle becomes an ellipse. Then `U`:
one final rotation, turning the ellipse into its resting orientation. Rotate, stretch,
rotate — that's every matrix, always.

**[0:44]** And here's the payoff — overlay the ellipse's stretch axes on the original
data cloud. They line up exactly with the variance directions from Act I. The singular
vectors *are* the principal components. The stretch amounts, the singular values, are
the spread of the data.

---

### Closing the loop (0:50–0:56)

**[0:50]** Finally, the algebra that ties it shut. `MᵀM = (UΣVᵀ)ᵀ(UΣVᵀ)`. The `UᵀU` in
the middle collapses to the identity, and you're left with `VΣ²Vᵀ`. So PCA's directions
are `V`, and the covariance eigenvalues are the squared singular values, `λᵢ = σᵢ²`.
Two stories, one decomposition. (And running the movie backwards — `M⁺ = VΣ⁺Uᵀ` — is
the pseudoinverse.)

---

**Recap.** PCA and SVD are the same object. "Maximize variance" and "minimize
reconstruction error" are one problem by Pythagoras; its solution is the top eigenvector
of `MᵀM`, which is a right singular vector of `M`, whose singular value squared is the
variance captured. `MᵀM = VΣ²Vᵀ` says it in one line.
