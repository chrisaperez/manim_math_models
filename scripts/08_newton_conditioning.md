# Newton vs Gradient Descent — read-along script

**Video:** `media/final/newton_conditioning.mp4` · **Length:** 0:43
**Source:** CS205L HW7.4–7.5, HW8.1/8.4, HW9.5

---

### The bad news, revisited (0:00–0:14)

**[0:00]** Back to a canyon-shaped quadratic — this time with condition number 50, a
1000-to-1 stretch. Gradient descent, even with the optimal fixed step, zig-zags across
the narrow valley: the step counter spins up past 400 before it's satisfied. GD's speed
is hostage to the conditioning.

**[0:08]** Now overlay Newton's method. From the same start, a single golden arrow points
straight at the minimum. One step. Not a hundred — one.

---

### Why one step (0:14–0:26)

**[0:14]** The difference is a single matrix. Gradient descent moves along `−α∇J`.
Newton multiplies by the inverse Hessian first: `θ⁺ = θ − H⁻¹∇J`. On a quadratic, `H`
*is* `A`, so `H⁻¹∇J = A⁻¹(Aθ) = θ` — the step is the entire vector back to the origin.
Newton doesn't approximate the answer; on a quadratic it *is* the answer.

**[0:20]** Geometrically, multiplying by `H⁻¹` un-squashes the space. Apply that change
of coordinates and the stretched ellipses become perfect circles — condition number one.
In that whitened space, the ordinary gradient points straight at the center. Newton is
just gradient descent in the metric of the Hessian, and it's affine-invariant: rotate or
stretch the problem however you like, and Newton's one-step behavior never changes.

---

### The honest part (0:26–0:38)

**[0:26]** Real problems aren't quadratic. Here's Rosenbrock's banana valley. Full Newton
from a bad start would overshoot into the wrong basin, so we *damp* it with a line search —
and damped Newton snakes safely down the curve in 22 steps.

**[0:32]** And when computing the Hessian is too expensive, BFGS *learns* it. It refines
an estimate of the curvature from successive gradients, enforcing the secant condition
`B·s = y` at every step, and finishes in 34 — while plain gradient descent would need
over twenty thousand. Newton 22, BFGS 34, GD twenty-thousand-plus: curvature information
is worth a great deal.

---

### The cost ledger (0:38–0:43)

**[0:38]** The trade-off in one line. A gradient costs `O(d)` but you need about `κ`
steps. A Newton step costs `O(d³)` to solve the system, but you need `O(1)` of them.
BFGS is the practical middle at `O(d²)` per step. Which is why deep learning uses neither
pure gradient descent nor the exact Hessian — but that's another video.

---

**Recap.** Gradient descent pays for bad conditioning with a step count proportional to
`κ`. Newton multiplies by `H⁻¹` to optimize in a space where `κ = 1`, solving quadratics
in a single step, at the cost of forming and inverting the Hessian. Quasi-Newton methods
like BFGS learn the curvature cheaply and sit in between.
