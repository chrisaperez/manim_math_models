# Demo 4 — GNN Message Passing & the Depth Barrier

**Source:** CS246 HW4.1 (sum-aggregation GNN, distinguishing graphs, BFS simulation)
**Sim:** `core/sims/gnn.py` · **Scene:** `scenes/gnn_depth.py` · **Web:** `web/gnn_depth.html`

## Load-bearing math

- Sum aggregation: `h_v^{(k+1)} = h_v^{(k)} + Σ_{u∈N(v)} h_u^{(k)}`, `h_v^{(0)} = 1` for all v.
- After k layers, `h_v^{(k)}` is a function of v's k-hop neighborhood ONLY (receptive field).
- Two non-isomorphic graphs whose red nodes have identical k-hop views for k < k* produce
  identical embeddings until layer k*; they can only be told apart at depth ≥ k*.
  (Chris's pset trace: values 1 → 2 → 6 → 18 match through k=3, diverge at k=4.)
- Related: a GNN layer with max-aggregation + threshold can simulate one BFS frontier step
  (HW4 1(c)) — depth = how far information travels.

## Arc

### 1. Intuition (~0:00–0:35)
A GNN node is a gossip listener: each layer, it hears only its immediate neighbors. After k
rounds, news from k hops away arrives. Anything farther might as well not exist.

### 2. Mathematical breakdown (~0:35–1:40)
Two different graphs — same degrees near the red node, different structure farther out. The
update rule is deterministic arithmetic on neighborhoods. If the k-hop unrolled trees are
identical, the numbers MUST be identical. Show the receptive-field tree unroll.

### 3. Visual proof (~1:40–3:40)
Run the layers live on both graphs. Numbers stay locked in sync (checkmarks) at k=1,2,3 —
then at k=4 the structural difference finally enters the receptive field and the embeddings
split. Freeze, rewind, highlight the cycle-closure edge that did it.

## Beat table

| # | t | On screen | Feeds from |
|---|-----|-----------|-----------|
| 1 | 0:00 | Graph A (6-cycle) and Graph B (two triangles sharing structure / branched path) side by side; red target node each; all nodes labeled 1 | `demo_graphs()` |
| 2 | 0:30 | Layer-1 pulse: dots travel every edge simultaneously; each node's value updates to `own + Σ neighbors`; red nodes both read 3 (or pset values), big "=" + checkmark between panels | `run_layers(G, k)` |
| 3 | 1:10 | Receptive-field unroll: red node's 1-hop tree drawn below each graph — identical trees overlay perfectly | `unroll_tree(G, v, k)` |
| 4 | 1:40 | Fast-forward k=2, 3: pulses, updated numbers, growing identical trees, checkmarks stack. Values echo the pset trace | `run_layers` |
| 5 | 2:30 | k=4: pulse crosses the structurally different edge; numbers diverge — panels flash, "≠" replaces checkmark; counter: "depth needed: 4" | `divergence_layer(GA, GB, vA, vB)` |
| 6 | 3:00 | Rewind + zoom: the distinguishing edge is exactly 4 hops from red; rings radii 1..4 drawn; caption "k layers see k hops. No more." | `hop_rings` |
| 7 | 3:25 | BFS coda: binary features, max-aggregation layer flips a node to 1 iff a neighbor is 1 — one BFS frontier per layer (HW4 1(c)) | `bfs_layer(G, seed)` |
| 8 | 3:40 | End card: WL-test remark — sum aggregation ≈ Weisfeiler-Leman colors | — |

## Web interactive

Both graphs rendered with editable topology (click two nodes to toggle an edge). Step button
advances k; node values shown as numbers + heat colors; equality banner per step. Hover any
node → its k-hop receptive-field tree unfolds in a side panel. "Find divergence depth" button
runs layers until embeddings split and reports k*.
