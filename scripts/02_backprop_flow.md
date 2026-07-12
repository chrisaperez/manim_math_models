# Backpropagation, Batched — read-along script

**Video:** `media/final/backprop_flow.mp4` · **Length:** 0:51
**Source:** CS229 PS2.4

---

### Part 1 — The forward pass as geometry (0:00–0:18)

**[0:00]** A batch isn't a mysterious tensor — it's a cloud of points. Here are eight
data rows scattered on the plane, the rows of the matrix `X`. A layer applies one linear
map `W` to all of them at once, sweeping the whole cloud to new positions. That's
`Z = XW + b`.

**[0:12]** Then the activation `σ` squashes the plane — a nonlinear warp that pulls
everything toward the unit square. Notice one point flung far out before the squash:
it lands deep in saturation, where `σ` is nearly flat. Remember that point; it matters
in a moment.

---

### Part 2 — Bookkeeping forces the rules (0:18–0:30)

**[0:18]** Keep the shapes on screen: `X` is n×d, `Z` and `A` are n×k, and the loss `L`
is a single scalar. The entire trick of batched backprop is that these dimensions leave
you no choice — each gradient formula is the *only* way to make the shapes fit.

---

### Part 3 — Error flows backward (0:30–0:42)

**[0:30]** The loss sees only the output `A`, and it hands back one red arrow per data
point — the upstream gradient `G = ∂L/∂A`. Read each arrow as "the direction the loss
wants this output to move."

**[0:34]** First those arrows pass through the activation's gate. We multiply by
`σ′(Z)` element-wise — a Hadamard product. And there's our saturated point: its `σ′` is
essentially zero, so its arrow shrinks to nothing. That's the dead-gradient problem, in
one frame — a saturated unit learns nothing, because its gate is closed.

**[0:38]** Now the key move. The surviving error flows *backward* through the layer, and
the grid transforms by `Wᵀ`. Going backward through `W` is the same as going forward
through `W`-transpose — row space to column space. In symbols: `∂L/∂X = (∂L/∂Z)Wᵀ`, and
the shapes check out, (n×k)(k×d) → (n×d).

---

### Part 4 — The weight gradient is a vote (0:42–0:51)

**[0:42]** What about the weights? `∂L/∂W = Xᵀ(∂L/∂Z)`, and that product is a *sum of
outer products* — one rank-1 sheet per example. Each data point stamps its own little
vote onto the weight update, and they pile up into the final gradient matrix.

**[0:48]** And to prove none of this is hand-waving: the analytic gradient and a
finite-difference estimate of the same number agree to eight decimal places. Stack this
one layer a few times and you have the MNIST network from the problem set.

---

**Recap.** Backprop through a batch never differentiates a matrix by a matrix. It follows
error arrows backward: through the activation they're gated by `σ′` (Hadamard), through
the weights they map by `Wᵀ`, and the weight update is a sum of rank-1 votes, one per
example. Shapes dictate every formula.
