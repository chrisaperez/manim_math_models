# Demo 8 — Newton vs Gradient Descent: Why Curvature Information Wins

**Source:** CS205L HW7.4 (Newton initial-guess sensitivity), HW7.5 (logistic regression via Newton),
HW8.1/8.4 (Broyden, BFGS), HW9.5 (GD vs condition number)
**Sim:** `core/sims/newton.py` · **Scene:** `scenes/newton_conditioning.py` · **Web:** `web/newton_conditioning.html`

## Load-bearing math

- GD on `J(θ)=½θᵀAθ`: steps ⊥ contours, zig-zags; iterations to ε-accuracy ~ `κ·log(1/ε)`, κ=λ_max/λ_min.
- Newton step: `θ⁺ = θ − H⁻¹∇J`. On a quadratic, `H⁻¹∇J = A⁻¹Aθ = θ` ⇒ exact minimum in ONE step,
  any κ, any rotation — Newton is affine-invariant: it optimizes in a space where κ=1.
- Non-quadratic: Newton = repeatedly minimizing the local quadratic model; diverges from bad
  starts (HW7.4) without damping/line search.
- Quasi-Newton: approximate H from gradient differences. Secant condition `B_{t+1}s_t = y_t`
  (`s_t = θ_{t+1}−θ_t`, `y_t = ∇J_{t+1}−∇J_t`); Broyden = least-change rank-1, BFGS = rank-2,
  SPD-preserving.

## Arc

### 1. Intuition (~0:00–0:35)
Demo 1 ended in bad news: GD's speed is hostage to the condition number. If the valley is a
1000:1 canyon, GD takes thousands of tiny zig-zags. But the canyon's shape is *knowable* — it's
the Hessian. What if we used it?

### 2. Mathematical breakdown (~0:35–1:50)
GD follows −∇J: perpendicular to contours, which is the WRONG direction in a stretched space.
Newton multiplies by H⁻¹ first — undoing the stretch. On a quadratic that's not an improvement,
it's the answer: one step, exactly.

### 3. Visual proof (~1:50–4:10)
Race on the same canyon: GD zig-zags 200 steps; Newton lands in 1. Show WHY as a change of
coordinates: H⁻¹ un-squashes the ellipses into circles where −∇J points straight at the min.
Then honesty beats: Rosenbrock (Newton needs damping), and BFGS learning curvature on the fly.

## Beat table

| # | t | On screen | Feeds from |
|---|-----|-----------|-----------|
| 1 | 0:00 | Callback: κ=50 canyon contours; GD trajectory zig-zagging, step counter spinning to ~200; energy decay curve crawling | `gd_run(A, θ0, α*)` |
| 2 | 0:40 | Newton overlay: single golden arrow from θ0 directly to minimum, counter "1". Silence beat. | `newton_run(A, θ0)` |
| 3 | 1:00 | `MathTex` morph: `θ − α∇J` → `θ − H⁻¹∇J`; on quadratic: `H⁻¹∇J = A⁻¹(Aθ) = θ` cancel animation ⇒ `θ⁺ = 0` exactly | — |
| 4 | 1:40 | The un-squash: grid transform by `A^{−1/2}` — ellipses → circles; in this space the SAME GD arrow points dead at the center; caption "Newton = GD in the metric of the Hessian; κ→1" | `whiten(A)` |
| 5 | 2:20 | Affine invariance: rotate/stretch the problem arbitrarily (three random As); Newton's one-step behavior identical each time; GD step counts vary wildly (table fills in) | `race_table(As)` |
| 6 | 2:50 | Honesty beat: Rosenbrock valley; full Newton from a bad start overshoots into the wrong basin (HW7.4); damped Newton (line search) snakes safely along the banana | `rosenbrock_runs` |
| 7 | 3:30 | Quasi-Newton: BFGS's ellipse (its current B model) morphs step by step toward the true Hessian ellipse; secant condition `B s = y` badge; race chips: GD 200 / BFGS 12 / Newton 5 (Rosenbrock) | `bfgs_run` w/ model snapshots |
| 8 | 4:00 | End card: cost ledger — ∇ costs O(d), H costs O(d²)+solve O(d³); BFGS the practical middle; "this is why deep learning uses neither H nor pure GD — but that's another video" | — |

## Web interactive

Contour playground: objective picker (quadratic with κ + rotation sliders / Rosenbrock);
click to set start point; race GD (α slider) vs Newton (damping toggle) vs BFGS — animated
trails, step counters, loss-vs-iteration semilog chart. "Whiten space" toggle re-renders the
whole scene in `A^{−1/2}` coordinates to show Newton's view of the world.
