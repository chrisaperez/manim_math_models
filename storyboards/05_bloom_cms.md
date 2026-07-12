# Demo 5 — Bloom Filters & Count-Min Sketch

**Source:** CS246 HW4.4 (count-min sketch ε–δ analysis) + Bloom filters (CS246 streaming lectures; explicit request)
**Sim:** `core/sims/sketches.py` · **Scene:** `scenes/bloom_cms.py` · **Web:** `web/bloom_cms.html`

## Load-bearing math

- Bloom filter: m bits, k hash functions, n inserted items.
  P[a bit still 0] = `(1−1/m)^{kn} ≈ e^{−kn/m}`; false-positive rate `p ≈ (1−e^{−kn/m})^k`;
  optimal `k = (m/n)ln2` gives `p = 2^{−k} ≈ 0.6185^{m/n}`. No false negatives, ever.
- Count-min sketch: d rows × w counters, row j uses hash `h_j`; estimate `ĉ(i) = min_j c_{j,h_j(i)}`.
  With `w = ⌈e/ε⌉`: `E[c_{j,h_j(i)}] ≤ F[i] + (t − F[i])·(e/ε)⁻¹·... ` → per HW4:
  `E[excess] ≤ ε·t/e`, Markov ⇒ `P[c_{j,h_j(i)} > F[i] + εt] ≤ 1/e`; independence over
  `d = ⌈ln(1/δ)⌉` rows ⇒ `P[ĉ(i) > F[i] + εt] ≤ e^{−d} ≤ δ`. Never underestimates.

## Arc

### 1. Intuition (~0:00–0:40)
You can't store a billion URLs in RAM — but you can store their *shadows*. Hash each item to a
few positions in a bit array. The shadow can lie ("maybe seen"), but only in one direction.

### 2. Mathematical breakdown (~0:40–2:00)
When does the shadow lie? A query is a false positive iff ALL k of its bits were set by others —
collisions compound. Compute P[bit=0] with the balls-in-bins argument; watch the FP curve rise
as the filter fills. There's an optimal k: too few bits checked → collisions cheap; too many →
array fills up.

### 3. Visual proof (~2:00–4:20)
Live filter fills; a never-inserted item hits k lit bits → the false-positive moment. Trace
empirical FP rate against the closed-form curve. Then upgrade bits → counters: count-min, where
the min over d independent rows crushes collision noise, with Markov + independence giving the
ε–δ certificate from the pset.

## Beat table

| # | t | On screen | Feeds from |
|---|-----|-----------|-----------|
| 1 | 0:00 | Bit array (m=32 cells); word "cat" hashes along k=3 arcs, bits light up; then "dog", "svd" | `bloom_insert(word)` |
| 2 | 0:40 | Query "cat": all 3 arcs land on lit bits → MAYBE (green). Query "gnn": one arc hits a 0 → definitely-NO (red). Caption: zeros never lie | `bloom_query` |
| 3 | 1:20 | Query "ost": never inserted, but all 3 bits lit by others → MAYBE. Freeze; collision arcs re-drawn in yellow — the false positive anatomy | crafted collision |
| 4 | 2:00 | Balls-in-bins `MathTex`: P[bit 0] = `(1−1/m)^{kn}` → `e^{−kn/m}` → FP `(1−e^{−kn/m})^k`; axes plot FP vs n, curve traced while the array visibly fills | `fp_theoretical(m,k,n)` |
| 5 | 2:40 | k-dial: family of FP curves for k=1..8 at fixed m/n; minimum highlighted at `k=(m/n)ln2`; empirical dots from Monte Carlo land on the curve | `fp_empirical` (test-verified) |
| 6 | 3:10 | Bits→counters morph: array becomes d=3 rows × w counters; stream of items increments cells; heavy-hitter "dai" piles up | `cms_update(stream)` |
| 7 | 3:40 | Query "dai": three row-estimates appear (each ≥ truth, noise varies); MIN operator sweeps and picks the smallest; true count marked — always ≤ estimate | `cms_query` |
| 8 | 4:00 | ε–δ certificate morph (the pset chain): `E[excess] ≤ εt/e` → Markov `P[row bad] ≤ 1/e` → rows independent `P[all bad] ≤ e^{−d} ≤ δ`; badge `w=⌈e/ε⌉, d=⌈ln(1/δ)⌉` | — |
| 9 | 4:15 | End card: same trick family — MinHash, LSH, Flajolet-Martin — hash once, lie predictably | — |

## Web interactive

Type-to-insert Bloom filter with visible bit array; m and k sliders; query box with MAYBE/NO
verdicts and collision arcs; live empirical-vs-theoretical FP chart (background Monte Carlo).
Count-min panel: preset zipfian stream, pick an item → row estimates with min highlighted,
ε/δ sliders resizing the sketch.
