# Demo 2 — Batched Neural-Network Gradients (Backprop as Matrix Flow)

**Source:** CS229 PS2.4 (batched gradients), PS2.5 (MNIST net)
**Sim:** `core/sims/backprop.py` · **Scene:** `scenes/backprop_flow.py` · **Web:** `web/backprop_flow.html`

## Load-bearing math

- Layer: `Z = XW + b` with `X ∈ ℝ^{n×d}`, `W ∈ ℝ^{d×k}`; activation `A = σ(Z)` elementwise.
- Given upstream `G = ∂L/∂A ∈ ℝ^{n×k}`:
  - through σ: `∂L/∂Z = G ⊙ σ′(Z)` (Hadamard mask)
  - `∂L/∂X = (∂L/∂Z) Wᵀ` — shapes: (n×k)(k×d) = n×d
  - `∂L/∂W = Xᵀ (∂L/∂Z)` — shapes: (d×n)(n×k) = d×k
  - `∂L/∂b = 𝟙ᵀ(∂L/∂Z)` — sum over the batch
- Every rule is forced by shape bookkeeping + linearity; no tensor gymnastics needed.

## Arc

### 1. Intuition (~0:00–0:40)
One training example: chain rule on scalars, fine. A batch of n examples through a weight
*matrix*: suddenly it's ∂(matrix)/∂(matrix) and notation panic. The trick: never differentiate
matrices by matrices — follow where each scalar *flows*.

### 2. Mathematical breakdown (~0:40–2:10)
Forward pass as geometry: rows of X are n points; W is one linear map applied to all of them.
The loss sees only A; the upstream gradient G is "one error vector per data point."
Question: how does an error at the *output* become blame at the *input* and at the *weights*?

### 3. Visual proof (~2:10–4:30)
The error cloud G flows backward through the same wires the data flowed forward. Backward
through W is forward through Wᵀ (row-space ↔ column-space). ∂L/∂W = Xᵀ(∂L/∂Z) appears as a sum
of outer products — each data point stamps its own rank-1 vote on the weight update.

## Beat table

| # | t | On screen | Feeds from |
|---|-----|-----------|-----------|
| 1 | 0:00 | 2D plane, n=8 colored points (rows of X). W applied: grid transform sweeps points to Z positions | `forward(X, W, b)` |
| 2 | 0:30 | σ squashes plane (nonlinear warp), saturated regions dim. Loss meter appears | `forward` acts |
| 3 | 1:00 | Shape ribbon `MathTex`: X: n×d → Z: n×k → A: n×k → L: scalar; dims highlighted per hop | — |
| 4 | 1:30 | Upstream G drawn as red arrows attached to each output point ("how L wants each output to move") | `backward(...)['G']` |
| 5 | 2:10 | Hadamard gate: σ′(Z) heatmap multiplies G; arrows in saturated zones shrink to zero ("dead gradient") | `backward(...)['dZ']` |
| 6 | 2:40 | THE beat: red arrows flow backward through the layer; grid morphs by Wᵀ; `TransformMatchingTex`: `∂L/∂X = (∂L/∂Z)Wᵀ` with (n×k)(k×d) → (n×d) badge | `backward(...)['dX']` |
| 7 | 3:20 | Weight blame: outer-product stack — each point i contributes `x⁽ⁱ⁾ᵀ δ⁽ⁱ⁾`; rank-1 sheets pile into ∂L/∂W matrix heatmap; `∂L/∂W = Xᵀ(∂L/∂Z)` | `outer_product_decomp` |
| 8 | 4:00 | Sanity ritual: side panel checks every formula's shapes; finite-difference needle agrees with analytic gradient to 1e-6 (numbers on screen from the actual test) | `fd_check` |
| 9 | 4:20 | End card: this one layer, composed, is the whole MNIST net of PS2.5 | — |

## Web interactive

Tiny 2-layer net (2→3→1) on a 2D toy dataset. Heatmap matrices for X, W₁, Z, A, W₂, ŷ; hover a
weight → its forward influence path and backward gradient path light up. Slider steps through:
forward → loss → G → mask → dX/dW. "Nudge" button perturbs a weight and shows loss change vs
predicted `∂L/∂w · Δw`.
