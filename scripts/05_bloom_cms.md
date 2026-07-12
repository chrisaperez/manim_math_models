# Bloom Filters & Count-Min Sketch — read-along script

**Video:** `media/final/bloom_cms.mp4` · **Length:** 0:49
**Source:** CS246 HW4.4 + streaming lectures

---

### Part 1 — Inserting into a Bloom filter (0:00–0:14)

**[0:00]** You can't fit a billion URLs in memory — but you can store their *shadows*. A
Bloom filter is just a row of bits. To insert an item, hash it with `k` different
functions and switch on the `k` bits they point to. Watch "cat" light three bits, then
"dog," then "svd." The array fills up with ones.

---

### Part 2 — Querying, and the one-sided lie (0:14–0:28)

**[0:14]** To query, hash the item and check its `k` bits. Query "cat": all three bits
are lit, so the answer is "maybe" — it's probably in the set. Query "gnn": one of its
bits is still zero, and that's a *definite* no. This is the whole personality of a Bloom
filter — a zero never lies. There are no false negatives, ever.

**[0:22]** But "maybe" can betray you. Query "ost" — a word we never inserted — and all
three of its bits happen to be lit by *other* items. The filter says "maybe," and it's
wrong. That's a false positive: not a bug, but the price of the compression, and it only
ever errs in the one safe direction.

---

### Part 3 — Calibrating the error (0:28–0:36)

**[0:28]** How likely is that lie? Balls in bins: the chance a given bit is still zero
after `n` inserts is about `e^(−kn/m)`, so the false-positive rate is
`(1 − e^(−kn/m))^k`. Trace the curve as the array fills — the fuller the shadow, the
more likely the lie. There's even an optimal number of hashes, `k = (m/n)ln2`, and the
measured rate from a live simulation lands right on the theoretical curve.

---

### Part 4 — Count-Min: from membership to counts (0:36–0:49)

**[0:36]** Upgrade the bits to counters and you can estimate *how many times* you've seen
each item. That's the count-min sketch: `d` rows of counters, each with its own hash.
Stream items in and the heavy hitters pile up.

**[0:42]** To query an item, look up its counter in every row. Each is an over-estimate —
collisions can only add — but the *noise* differs per row. So take the minimum across
rows, and it crushes the collision noise. The estimate is never below the truth.

**[0:46]** And it comes with a guarantee, straight from the problem set: with width
`⌈e/ε⌉` each row's expected over-count is at most `εt/e`, Markov bounds one row's failure
at `1/e`, and with `⌈ln(1/δ)⌉` independent rows the min fails with probability at most
`δ`. Same trick as the Bloom filter — hash once, and lie predictably.

---

**Recap.** Both structures trade exactness for tiny memory by hashing into shared cells.
The Bloom filter answers membership with no false negatives and a tunable false-positive
rate; the count-min sketch answers counts with a one-sided over-estimate that the
minimum-over-rows squeezes down to a provable `ε`–`δ` guarantee.
