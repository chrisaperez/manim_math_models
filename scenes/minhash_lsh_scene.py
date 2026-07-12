"""Demo 7 — MinHash & LSH: turning similarity into collision probability.

Act I: why P[minhash agree] = Jaccard, exactly — the first union row under a
random permutation is uniform. Act II: banding bends that linear estimator
into a tunable S-curve step function.

Render:
    manim render -qh scenes/minhash_lsh_scene.py MinhashLsh
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from manim import (  # noqa: E402
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Axes,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Indicate,
    ReplacementTransform,
    Scene,
    SurroundingRectangle,
    Text,
    VGroup,
    Write,
)

from core import style  # noqa: E402
from core.sims import minhash_lsh as mh  # noqa: E402

style.apply_house_style()

# shingle sets: J = 2/6 = 1/3
ELEMENTS = ["a", "b", "c", "d", "e", "f"]
IN_A = {"a", "b", "c", "d"}
IN_B = {"c", "d", "e", "f"}
J = mh.jaccard(IN_A, IN_B)
RNG = np.random.default_rng(46)


class MinhashLsh(Scene):
    def construct(self):
        self.beat_jaccard()
        self.beat_matrix_shuffle()
        self.beat_proof()
        self.beat_convergence()
        self.beat_banding()
        self.beat_s_curve()

    # -- Act I ----------------------------------------------------------
    def beat_jaccard(self):
        title = style.title_card(
            "MinHash — similarity as a collision",
            "compare a million documents without comparing a million documents",
        ).to_edge(UP)
        self.play(FadeIn(title))
        self.wait(0.6)

        circA = Circle(radius=1.5, color=style.TEAL).shift(LEFT * 0.9)
        circB = Circle(radius=1.5, color=style.YELLOW).shift(RIGHT * 0.9)
        posts = {
            "a": LEFT * 1.9 + UP * 0.5, "b": LEFT * 1.7 + DOWN * 0.6,
            "c": UP * 0.45, "d": DOWN * 0.5,
            "e": RIGHT * 1.9 + UP * 0.5, "f": RIGHT * 1.7 + DOWN * 0.6,
        }
        dots = VGroup(*[
            VGroup(Dot(posts[e], radius=0.05, color=style.WHITE),
                   Text(e, font_size=26, font=style.SERIF,
                        color=style.WHITE).next_to(posts[e], UP, buff=0.08))
            for e in ELEMENTS
        ])
        tagA = style.eq("A", size=34, color=style.TEAL).next_to(circA, LEFT)
        tagB = style.eq("B", size=34, color=style.YELLOW).next_to(circB, RIGHT)
        jac = style.eq(
            r"J(A,B) = \frac{|A\cap B|}{|A\cup B|} = \frac{2}{6} = \frac{1}{3}",
            size=36,
        ).to_edge(DOWN, buff=0.6)
        self.play(Create(circA), Create(circB), FadeIn(dots),
                  FadeIn(tagA), FadeIn(tagB))
        self.play(Write(jac))
        self.wait(1.2)
        self.play(*map(FadeOut, [circA, circB, dots, tagA, tagB, jac, title]))

    def matrix_group(self, order):
        """Characteristic matrix rows in the given element order."""
        rows = VGroup()
        for e in order:
            a, b = int(e in IN_A), int(e in IN_B)
            in_union = (a or b)
            row = VGroup(
                Text(e, font_size=24, font=style.SERIF, color=style.GRAY),
                style.eq(str(a), size=28,
                         color=style.TEAL if a else style.GRAY),
                style.eq(str(b), size=28,
                         color=style.YELLOW if b else style.GRAY),
            )
            row[0].move_to(LEFT * 1.2)
            row[1].move_to(LEFT * 0.0)
            row[2].move_to(RIGHT * 1.2)
            row.in_union = in_union
            row.element = e
            rows.add(row)
        for i, row in enumerate(rows):
            row.shift(UP * (1.8 - 0.62 * i))
        return rows

    def beat_matrix_shuffle(self):
        header = VGroup(
            Text("π", font_size=30, font=style.SERIF, color=style.GRAY),
            style.eq("A", size=30, color=style.TEAL),
            style.eq("B", size=30, color=style.YELLOW),
        )
        header[0].move_to(LEFT * 1.2 + UP * 2.45)
        header[1].move_to(UP * 2.45)
        header[2].move_to(RIGHT * 1.2 + UP * 2.45)

        self.order = list(ELEMENTS)
        rows = self.matrix_group(self.order)
        note = style.label(
            "characteristic matrix — shuffle the rows with a random permutation",
            size=26).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(header), FadeIn(rows), FadeIn(note))
        self.wait(0.6)

        # shuffle: permute row targets
        perm = list(RNG.permutation(len(self.order)))
        new_order = [self.order[i] for i in perm]
        anims = []
        for i, row in enumerate(rows):
            target = new_order.index(self.order[i])
            anims.append(row.animate.shift(UP * (0.62 * (i - target))))
        self.play(*anims, run_time=1.4)
        self.order = new_order
        self.rows = rows
        self.header = header

        # box the first 1 in each column
        first_a = next(e for e in new_order if e in IN_A)
        first_b = next(e for e in new_order if e in IN_B)
        boxes = VGroup()
        for e, col, color in ((first_a, 1, style.TEAL), (first_b, 2, style.YELLOW)):
            row = rows[[r.element for r in rows].index(e)]
            boxes.add(SurroundingRectangle(row[col], color=color, buff=0.12))
        sig = style.eq(
            rf"h_{{\min}}(A) = \text{{{first_a}}},\quad h_{{\min}}(B) = \text{{{first_b}}}"
            + (r"\quad \Rightarrow\ \text{agree!}" if first_a == first_b
               else r"\quad \Rightarrow\ \text{differ}"),
            size=30,
            color=style.GREEN if first_a == first_b else style.RED,
        ).to_edge(DOWN, buff=0.5)
        self.play(Create(boxes), ReplacementTransform(note, sig))
        self.wait(1.2)
        self.boxes = boxes
        self.sig = sig

    def beat_proof(self):
        # fade non-union rows (none here — all 6 in union), color x/y types
        x_rows, y_rows = VGroup(), VGroup()
        for row in self.rows:
            e = row.element
            if e in IN_A and e in IN_B:
                x_rows.add(row)
            else:
                y_rows.add(row)
        proof = style.eq(
            r"\text{first union row under } \pi \text{ is uniform}"
            r"\ \Rightarrow\ P[\text{agree}] = \frac{|x\text{-rows}|}{|x|+|y|}"
            r" = \frac{|A\cap B|}{|A\cup B|} = J",
            size=30, color=style.WHITE,
        ).to_edge(DOWN, buff=0.5)
        self.play(
            *[row.animate.set_opacity(1.0) for row in x_rows],
            *[Indicate(row, color=style.PURPLE, scale_factor=1.15)
              for row in x_rows],
            ReplacementTransform(self.sig, proof),
        )
        xlab = style.label("x-type: 1 in both (agree)", size=22, color=style.PURPLE)
        ylab = style.label("y-type: 1 in one (differ)", size=22, color=style.GRAY)
        VGroup(xlab, ylab).arrange(DOWN, buff=0.2).shift(RIGHT * 4.3 + UP * 1.4)
        self.play(FadeIn(xlab), FadeIn(ylab))
        self.wait(1.6)
        self.play(*map(FadeOut, [self.rows, self.header, self.boxes,
                                 proof, xlab, ylab]))

    def beat_convergence(self):
        ests = [mh.minhash_agreement(IN_A, IN_B, k=k, seed=7)
                for k in (10, 100, 1000, 10000)]
        lines = VGroup(*[
            style.eq(
                rf"k = {k}:\quad \hat{{J}} = {est:.4f}", size=32,
                color=style.WHITE if i < 3 else style.GREEN,
            )
            for i, (k, est) in enumerate(zip((10, 100, 1000, 10000), ests))
        ]).arrange(DOWN, buff=0.4, aligned_edge=LEFT).shift(UP * 0.6)
        truth = style.eq(r"J = 1/3 = 0.3333\ldots", size=34, color=style.YELLOW)
        truth.next_to(lines, DOWN, buff=0.5)
        var = style.eq(r"\operatorname{Var}(\hat{J}) = J(1-J)/k", size=28,
                       color=style.GRAY).next_to(truth, DOWN, buff=0.35)
        for line in lines:
            self.play(Write(line), run_time=0.5)
        self.play(Write(truth), FadeIn(var))
        self.wait(1.4)
        self.play(*map(FadeOut, [lines, truth, var]))

    # -- Act II ---------------------------------------------------------
    def beat_banding(self):
        header = style.title_card(
            "LSH banding — bending the estimator into a step",
            "split the signature into b bands of r rows; candidates = any band identical",
        ).to_edge(UP)
        self.play(FadeIn(header))
        self.header2 = header

        b, r = 4, 3
        cells = VGroup()
        rng = np.random.default_rng(2)
        for i in range(12):
            agree = rng.random() < 0.75
            cell = style.eq(f"{rng.integers(10, 99)}", size=24,
                            color=style.GREEN if agree else style.RED)
            cell.move_to(LEFT * 4.8 + UP * (1.6 - i * 0.42))
            cells.add(cell)
        bands = VGroup()
        for j in range(b):
            box = SurroundingRectangle(
                VGroup(*cells[j * r:(j + 1) * r]),
                color=style.BLUE, buff=0.12)
            bands.add(box)
        caption = style.label(
            "a band matches only if ALL r of its rows agree — then it hashes to the same bucket",
            size=24).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(cells), Create(bands), FadeIn(caption))
        self.wait(1.2)
        self.cells, self.bands, self.caption = cells, bands, caption

    def beat_s_curve(self):
        axes = Axes(
            x_range=[0, 1, 0.25], y_range=[0, 1, 0.25],
            x_length=6.4, y_length=4.0, tips=False,
            axis_config={"stroke_color": style.GRAY},
        ).shift(RIGHT * 1.6 + DOWN * 0.5)
        xlab = style.eq(r"s = \text{true similarity}", size=26, color=style.GRAY)
        xlab.next_to(axes, DOWN, buff=0.2)
        ylab = style.eq(r"P[\text{candidate}]", size=26, color=style.GRAY)
        ylab.next_to(axes, LEFT, buff=0.15).rotate(np.pi / 2)

        self.play(Create(axes), FadeIn(xlab), FadeIn(ylab))

        configs = [(3, 4, style.TEAL), (6, 16, style.YELLOW), (10, 40, style.RED)]
        formula = style.eq(r"P = 1 - (1 - s^{r})^{b}", size=36,
                           color=style.WHITE)
        formula.move_to(RIGHT * 4.6 + UP * 2.2)
        self.play(Write(formula))
        curve_prev, tag_prev, dot_prev = None, None, None
        for r, b, color in configs:
            curve = axes.plot(lambda s: float(mh.s_curve(s, r, b)),
                              x_range=[0.001, 0.999], color=color)
            t = mh.s_curve_threshold(r, b)
            dot = Dot(axes.coords_to_point(t, float(mh.s_curve(t, r, b))),
                      color=color, radius=0.06)
            tag = style.eq(rf"r={r},\ b={b}", size=26, color=color)
            tag.next_to(formula, DOWN, buff=0.3)
            if curve_prev is None:
                self.play(Create(curve), FadeIn(dot), FadeIn(tag))
            else:
                self.play(ReplacementTransform(curve_prev, curve),
                          ReplacementTransform(dot_prev, dot),
                          ReplacementTransform(tag_prev, tag))
            curve_prev, tag_prev, dot_prev = curve, tag, dot
            self.wait(0.8)

        thresh = style.eq(r"s^{*} \approx (1/b)^{1/r}", size=30,
                          color=style.GRAY).next_to(tag_prev, DOWN, buff=0.3)
        self.play(Write(thresh))
        end = style.label(
            "sharper curve = cleaner cut: below s* almost never collides, above almost always",
            size=24).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(self.caption, end))
        self.wait(2.2)
