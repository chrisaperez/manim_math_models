# Ridge Regression — Buying Accuracy with Bias — read-along script

**Video:** `media/final/ridge.mp4` · **Length:** 0:42
**Source:** CS229 PS2.2

---

### OLS: perfect aim, shaky hand (0:00–0:12)

**[0:00]** Think of an estimator as a dart thrower. The bullseye is the true weight `w*`.
Each dart is the estimate you'd get from one noisy sample of data. Ordinary least
squares throws a whole cloud of darts — and its center sits *right on* the bullseye. OLS
is unbiased; on average it's exactly right.

**[0:08]** But look how *wide* the cloud is. Perfect aim, terribly shaky hand. Any single
estimate can land far from the truth, because the variance is large — especially along
directions where the data barely constrains the fit.

---

### Turning the ridge dial (0:12–0:22)

**[0:12]** Now add a ridge penalty `λ` and turn it up. Two things happen at once. The
cloud *tightens* — variance falls. But its center *drifts off* the bullseye — we've
introduced bias. We're deliberately aiming a little wrong in exchange for a much steadier
hand.

**[0:18]** That's the decomposition on screen: mean squared error splits into bias
squared plus variance. Bias is the gap from the average estimate to the truth; variance
is the scatter around that average.

---

### The U-curve (0:22–0:34)

**[0:22]** Plot both against `λ`. Bias squared climbs from zero; variance falls from its
enormous OLS value, `σ²Σ1/γᵢ`. Their sum — the MSE — is a U. There's a sweet spot `λ*`
strictly better than OLS.

**[0:28]** And here's the problem set's punchline, the reason this always matters: the
slope of the MSE curve at `λ = 0` is *strictly negative* for any noise `σ² > 0`. The
curve always dips before it climbs. So OLS is *never* the bottom of the U — a little
ridge always helps.

---

### Where the shrinkage lands (0:34–0:42)

**[0:34]** Why does biasing help so much? In the eigenbasis, ridge shrinks each direction
by `γ/(γ+λ)`. The weak-signal direction — the one with the small eigenvalue — is shrunk
first and hardest. And that's precisely the direction where OLS variance blows up, since
it scales as `σ²/γ`. Ridge spends its bias exactly where variance is most out of control.

**[0:40]** So the theorem, stated plainly: for every positive noise level there is a
positive `λ` that beats ordinary least squares.

---

**Recap.** Ridge regression trades a little bias for a large reduction in variance by
shrinking each eigen-direction by `γ/(γ+λ)`, hitting the weak, high-variance directions
hardest. The MSE is a U-shaped curve whose slope at `λ = 0` is always negative — so some
positive ridge penalty always outperforms OLS.
