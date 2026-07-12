# Demo 7 — MinHash & LSH: Similarity at Scale

**Source:** CS246 HW1.3 (MinHash "don't know" bound, cyclic-permutation pitfall), HW1.4 (LSH nearest-neighbor theory + image experiment)
**Sim:** `core/sims/minhash_lsh.py` · **Scene:** `scenes/minhash_lsh_scene.py` · **Web:** `web/minhash_lsh.html`

## Load-bearing math

- MinHash miracle: for a uniform random permutation π, `P[min π(A) = min π(B)] = |A∩B|/|A∪B| = J(A,B)`
  — the first row (under π) that is 1 in either column decides, and it's uniform over the union.
- k independent hashes → signature; agreement fraction is an unbiased estimator of J with
  variance J(1−J)/k.
- "Don't know" (all k rows are 0 in the column, HW1 3(a)): `P = ((n−m)/n)^k ≤ e^{−km/n}`;
  choosing `k = 10n/m` drives it under `e^{−10}` (3(b)).
- Cyclic permutations are NOT enough (3(c)) — dependence breaks uniformity over the union.
- LSH banding: b bands × r rows; `P[candidate] = 1 − (1−sʳ)ᵇ` — an S-curve with threshold
  `s* ≈ (1/b)^{1/r}`; HW1 4(a–c): union/intersection bounds give constant-probability NN guarantees.

## Arc

### 1. Intuition (~0:00–0:40)
Comparing all pairs of a million documents is 5·10¹¹ comparisons. We need similar items to
*collide* — turn similarity into a hash function's collision probability.

### 2. Mathematical breakdown (~0:40–2:10)
Characteristic matrix; permute rows; keep only the first 1 per column. Why is signature
agreement EXACTLY Jaccard? Walk the union rows: types x (both 1), y (one 1); the first union
row under π is uniform; agreement ⟺ it's type x ⇒ P = |x|/(|x|+|y|) = J.

### 3. Visual proof (~2:10–4:20)
Signatures agree at rate J (law of large numbers bars). Then banding: hash bands into buckets,
collide if any band matches. The S-curve `1−(1−sʳ)ᵇ` sharpens into a step function as r,b grow —
a tunable similarity threshold.

## Beat table

| # | t | On screen | Feeds from |
|---|-----|-----------|-----------|
| 1 | 0:00 | Two document blobs → shingle sets A, B as Venn diagram; `J = |A∩B|/|A∪B|` with region highlights | `make_sets` |
| 2 | 0:35 | Characteristic matrix (rows = shingles, 2 columns); rows shuffle (permutation π animated as row reorder); first 1 per column boxed → signature entries | `permute_and_min` |
| 3 | 1:20 | The proof beat: rows outside A∪B fade out ("spectators"); union rows colored x-type (purple, both) / y-type (gray, one); the topmost union row after shuffle blinks — agreement ⟺ purple; `P = x/(x+y) = J` morph | `union_row_types` |
| 4 | 2:10 | k=100 permutations run fast; agreement fraction ticker converges to J (live number vs true J); variance shrinks as 1/k badge | `minhash_signature(k)` |
| 5 | 2:45 | "Don't know" aside (HW1 3a): sparse column, all k sampled rows are 0 → shrug icon; bound `((n−m)/n)^k ≤ e^{−km/n}` traced | `dont_know_prob` |
| 6 | 3:10 | Banding: signature splits into b bands of r rows; bands hash to buckets (arcs); any shared bucket → candidate pair | `band_collide(sig, r, b)` |
| 7 | 3:35 | S-curve `1−(1−sʳ)ᵇ` drawn over s∈[0,1]; (r,b) morph (16,8)→(8,16)→(4,32): curve steepens/shifts; threshold `s*=(1/b)^{1/r}` dot slides; FP/FN regions shaded | `s_curve(r, b)` |
| 8 | 4:05 | End card: HW1 4(d) — LSH vs linear scan on image patches, near-identical neighbors, fraction of the time | — |

## Web interactive

Two editable shingle sets (chips add/remove) → live Venn, true J; k slider runs MinHash and shows
estimate ± error trace. S-curve panel: r, b sliders, threshold marker, shaded FP/FN mass for a
chosen similarity target; "sharpen" animation button sweeping (r,b) at constant rb.
