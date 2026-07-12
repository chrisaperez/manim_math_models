# GNN Message Passing & the Depth Barrier — read-along script

**Video:** `media/final/gnn_depth.mp4` · **Length:** 0:27
**Source:** CS246 HW4.1

---

### Setup (0:00–0:07)

**[0:00]** Two graphs side by side — a cycle on the left, a branched path on the right —
each with one red target node. We run a graph neural network with the simplest possible
rule: a node's new value is itself plus the sum of its neighbors' values. Every node
starts at 1. The question: how many layers does it take to tell the two red nodes apart?

---

### The layers march (0:07–0:18)

**[0:07]** Layer 1. Values pulse along every edge at once, and each node updates. Both
red nodes read the same number. Checkmark — still identical.

**[0:11]** Layers 2 and 3 fly by. The red nodes compute 2, then 6, then 18 — matching
your problem-set trace exactly — and at every step the two panels stay in lockstep. The
reason is that after `k` rounds, a node's value depends only on what it can *see* within
`k` hops. As long as those `k`-hop neighborhoods look identical, the numbers *must* be
identical. It's not luck; it's forced.

---

### The barrier breaks (0:18–0:27)

**[0:18]** Layer 4. Now the pulse finally reaches the edge where the two graphs actually
differ — the cycle closes on one side, the path dangles on the other. The numbers split:
56 on the cycle, 55 on the branched path. The checkmark becomes a "not-equal," and both
red nodes flash. Four layers of message passing were needed — no fewer.

**[0:23]** Zoom out and it's clear why: the distinguishing structure sits exactly four
hops from the red node. Rings of radius 1, 2, 3, 4 show the receptive field growing one
hop per layer. A GNN sees exactly `k` hops after `k` layers — no more. This is, in
essence, the Weisfeiler–Leman test, and it's the fundamental limit on what message
passing can distinguish.

---

**Recap.** A `k`-layer GNN can only "see" a node's `k`-hop neighborhood. Two graphs
whose neighborhoods match out to distance `k−1` are indistinguishable until layer `k`.
Here the structural difference is 4 hops away, so exactly 4 layers are required — the
depth barrier is a statement about receptive fields, not about training.
