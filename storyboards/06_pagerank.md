# Demo 6 — PageRank: Random Surfer & Power Iteration

**Source:** CS246 HW3.1 (mass conservation, teleportation, leaked rank), HW3.2 (power iteration in practice)
**Sim:** `core/sims/pagerank.py` · **Scene:** `scenes/pagerank_surfer.py` · **Web:** `web/pagerank.html`

## Load-bearing math

- Column-stochastic link matrix M (`M_ij = 1/d_j` if j→i): `r′ = Mr` preserves `Σr` because every
  column sums to 1 (Chris's HW3 1(a) swap-the-summations proof).
- Dead end ⇒ a zero column ⇒ mass leaks: `Σr′ = Σ_{j live} r_j < Σr`.
- Spider trap ⇒ mass conserved but *captured*.
- Google matrix: `r′ = βMr + (1−β)𝟙/n` — conserves mass (1(b)) and escapes traps; stationary r is
  the principal eigenvector; power iteration converges at rate ~β.

## Arc

### 1. Intuition (~0:00–0:35)
Importance is circular: a page matters if important pages link to it. Break the circularity with
a random surfer — importance = fraction of time the surfer spends on you.

### 2. Mathematical breakdown (~0:35–1:50)
One hop of the surfer = one multiply by M. Why does total rank stay 1? The column-sum argument,
animated. Then break it twice: a dead end drains mass out of the system; a spider trap keeps
mass but hoards it.

### 3. Visual proof (~1:50–3:50)
Teleportation: with prob 1−β jump anywhere. Watch the trap drain back out, the mass meter pin
to 1, and the rank vector converge — bars settling geometrically like βᵗ.

## Beat table

| # | t | On screen | Feeds from |
|---|-----|-----------|-----------|
| 1 | 0:00 | 6-node directed graph; rank as liquid fill level in each node (all 1/6); surfer dot hops randomly | `demo_graph()` |
| 2 | 0:30 | One synchronized flow step: liquid splits along out-edges (animated flow along arrows); `MathTex` `r′ = Mr`; M shown with column highlighted, column-sum = 1 badge | `flow_step(M, r)` |
| 3 | 1:00 | Mass meter (Σr = 1.000) pinned while steps repeat; swap-summation morph of the 1(a) proof | `mass(r)` |
| 4 | 1:30 | Add dead end node: its column in M turns all-zero; each step the meter visibly drops 1.0 → 0.94 → ... ; liquid drains into a "void" below | `add_dead_end` |
| 5 | 2:05 | Replace with spider trap (2-cycle): meter stays 1 but all liquid pools in the trap; other nodes dry up | `add_trap` |
| 6 | 2:40 | Teleport fix: dashed arcs from trap to every node; `TransformMatchingTex` `r′ = Mr` → `r′ = βMr + (1−β)𝟙/n`; β dial shown | `google_step(M, r, β)` |
| 7 | 3:10 | Convergence race: bar chart of r per iteration, deltas shrinking ~βᵗ; final bars = eigenvector computed independently (overlaid ticks) | `power_iterate` vs `eig_rank` |
| 8 | 3:40 | End card: HW3.2's 1000-node graph — same math, top-5 nodes flash (his reported IDs) | precomputed |

## Web interactive

Editable graph (add/remove nodes and edges); β slider; step / run buttons; liquid-fill nodes +
rank bar chart; total-mass meter that visibly leaks with a dead end and pins at 1 with teleport;
"converged after N steps" readout with tolerance control.
