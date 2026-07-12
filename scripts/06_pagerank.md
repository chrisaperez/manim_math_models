# PageRank — Random Surfer & Power Iteration — read-along script

**Video:** `media/final/pagerank.mp4` · **Length:** 0:37
**Source:** CS246 HW3.1–3.2

---

### The circular problem (0:00–0:08)

**[0:00]** Importance on the web is circular: a page matters if important pages link to
it — but their importance depends on *their* backlinks, and so on. PageRank breaks the
circle with a story: imagine a surfer clicking links at random forever. A page's
importance is simply the fraction of time the surfer spends there.

---

### One step conserves rank (0:08–0:18)

**[0:08]** Model rank as liquid sitting in each node. One step of the surfer is one
multiply by the link matrix `M`: every node pours its rank out along its outgoing edges,
`r' = Mr`. Watch the mass meter at the top — it stays pinned at 1.000.

**[0:14]** Why does it hold? Because every column of `M` sums to one — a node gives away
exactly all of its rank, no more, no less. Swap the order of summation, as in the
problem set, and total rank is conserved by construction.

---

### Two ways it breaks (0:18–0:26)

**[0:18]** But the web isn't so tidy. Add a dead end — a node with no outgoing links —
and its column is all zeros. Now each step the meter *drops*: rank pours into the void
and leaks out of the system entirely.

**[0:22]** Or add a spider trap — a little cycle that links only to itself. Here the mass
is conserved, but it all pools inside the trap. Every other node dries up. The surfer
walks in and never leaves.

---

### Teleportation and convergence (0:26–0:37)

**[0:26]** The fix is teleportation. With probability `1 − β` the surfer jumps to a
random page instead of following a link: `r' = βMr + (1 − β)·𝟙/n`. Those dashed jump
arrows drain the trap back out, the meter snaps back to one, and no structure can hold
the surfer captive.

**[0:32]** Repeat this step and the rank vector converges — the bars settle geometrically,
at a rate governed by `β`. And the values it settles on are exactly the principal
eigenvector of the Google matrix, which we compute independently and overlay as ticks.
Power iteration and the eigenvector agree.

---

**Recap.** PageRank is the stationary distribution of a random surfer. Pure link-following
`r' = Mr` conserves rank only when there are no dead ends; teleportation
`r' = βMr + (1−β)𝟙/n` restores conservation and escapes spider traps, and power iteration
converges to the principal eigenvector at rate `β`.
