# MinHash & LSH — read-along script

**Video:** `media/final/minhash_lsh.mp4` · **Length:** 0:38
**Source:** CS246 HW1.3–1.4

---

### The scaling problem (0:00–0:08)

**[0:00]** Comparing a million documents pairwise is half a trillion comparisons — hopeless.
We need similar items to *collide* on their own. The idea: turn similarity into the
collision probability of a hash function. Start with two documents as sets of shingles,
drawn as a Venn diagram. Their Jaccard similarity is the overlap over the union — here,
two out of six, one-third.

---

### The MinHash miracle (0:08–0:26)

**[0:08]** Build the characteristic matrix: rows are shingles, columns are the two sets,
ones where a shingle is present. Now shuffle the rows with a random permutation, and for
each column keep the *first* row that holds a one. That single value is the column's
MinHash signature.

**[0:16]** Here's the beautiful part — why the two signatures agree with probability
*exactly* Jaccard. Ignore rows outside the union; they're spectators. Among the union
rows, some are "type x" — a one in both columns — and some "type y" — a one in just one.
After shuffling, the topmost union row is uniformly random. The signatures agree exactly
when that top row is type x. So the agreement probability is `|x| / (|x| + |y|)` — which
is the intersection over the union, the Jaccard similarity itself.

**[0:24]** Run a hundred independent permutations and the fraction that agree converges
right onto the true Jaccard, with variance shrinking like `1/k`. We've turned a similarity
into an unbiased estimator you get for free from hashing.

---

### LSH banding — sharpening the cut (0:26–0:38)

**[0:26]** Now amplify. Split each signature into `b` bands of `r` rows, and call two
items a candidate pair if *any* whole band matches. The probability of being a candidate
is `1 − (1 − s^r)^b`, an S-curve in the similarity `s`.

**[0:32]** And you can *tune* that curve. As you trade rows for bands — keeping the total
signature length fixed — the S-curve steepens and shifts, its threshold sitting near
`(1/b)^(1/r)`. Below the threshold, pairs almost never collide; above it, they almost
always do. That's a tunable similarity cutoff, built entirely from hashing.

---

**Recap.** MinHash makes the probability that two sets share a minimum-hash equal to their
Jaccard similarity — an unbiased estimator from random permutations. LSH banding then bends
that linear relationship into a near-step function, giving a sharp, tunable "similar or not"
threshold without comparing all pairs.
