"""Demo 5 — Bloom Filters & Count-Min Sketch: shadows that only lie one way.

A bit array + k hashes answers membership with no false negatives and a
calibrated false-positive rate; upgrade bits to counters and take a min over
independent rows and you get the count-min eps-delta guarantee from HW4.

Render:
    manim render -qh scenes/bloom_cms.py BloomCms
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
    Create,
    CubicBezier,
    FadeIn,
    FadeOut,
    Flash,
    ReplacementTransform,
    Scene,
    Square,
    Text,
    TransformMatchingTex,
    VGroup,
    Write,
)

from core import style  # noqa: E402
from core.sims import sketches as sk  # noqa: E402

style.apply_house_style()

M, K = 24, 3
BF = sk.BloomFilter(M, K)
WORDS = ["cat", "dog", "svd"]
for w in WORDS:
    BF.insert(w)
# a genuinely-negative word and a genuine false positive, found by search
NEG = next(w for w in ["gnn", "lsh", "map", "pca"] if not BF.query(w))
FP = sk.find_false_positive(
    BF, [f"q{i}" for i in range(500)])
CELL = 0.46


class BloomCms(Scene):
    def construct(self):
        self.build_array()
        self.beat_inserts()
        self.beat_queries()
        self.beat_theory()
        self.beat_count_min()
        self.beat_certificate()

    def cell_pos(self, i: int) -> np.ndarray:
        return np.array([(i - (M - 1) / 2) * (CELL + 0.06), -0.4, 0])

    def build_array(self):
        title = style.title_card(
            "Bloom filters — membership in one bit per item-ish",
            "hash each item to k positions; zeros never lie",
        ).to_edge(UP)
        self.play(FadeIn(title))
        self.title = title

        self.cells = VGroup(*[
            Square(CELL).set_stroke(style.GRAY, 1.5)
            .set_fill(style.BACKGROUND, 1.0).move_to(self.cell_pos(i))
            for i in range(M)
        ])
        self.zeros = VGroup(*[
            style.eq("0", size=22, color=style.GRAY).move_to(self.cell_pos(i))
            for i in range(M)
        ])
        self.play(FadeIn(self.cells), FadeIn(self.zeros), run_time=1.0)
        self.lit = set()

    def arcs_for(self, word: str, y0: float, color) -> tuple[VGroup, list[int]]:
        pos = BF.positions(word)
        arcs = VGroup()
        for p in pos:
            start = np.array([0, y0 - 0.35, 0])
            end = self.cell_pos(p) + np.array([0, CELL / 2, 0])
            ctrl1 = start + np.array([0, -0.8, 0])
            ctrl2 = end + np.array([0, 1.0, 0])
            arcs.add(CubicBezier(start, ctrl1, ctrl2, end,
                                 stroke_color=color, stroke_width=3))
        return arcs, pos

    def light(self, positions, color=style.YELLOW):
        anims = []
        for p in positions:
            if p not in self.lit:
                self.lit.add(p)
                one = style.eq("1", size=22, color=style.BACKGROUND)
                one.move_to(self.cell_pos(p))
                anims.append(self.cells[p].animate.set_fill(color, 0.95))
                anims.append(ReplacementTransform(self.zeros[p], one))
                self.zeros[p] = one
        return anims

    def beat_inserts(self):
        for word, color in zip(WORDS, [style.BLUE, style.TEAL, style.PURPLE]):
            tag = Text(word, font_size=34, color=color, font=style.SERIF)
            tag.move_to([0, 1.7, 0])
            arcs, pos = self.arcs_for(word, 1.7, color)
            self.play(FadeIn(tag), run_time=0.35)
            self.play(Create(arcs), run_time=0.8)
            self.play(*self.light(pos), run_time=0.5)
            self.play(FadeOut(arcs), FadeOut(tag), run_time=0.35)

    def verdict_label(self, text, color):
        v = Text(text, font_size=30, color=color, font=style.SERIF)
        return v.move_to([4.6, 1.7, 0])

    def beat_queries(self):
        cases = [
            ("cat", style.BLUE, "MAYBE — all k bits lit", style.GREEN),
            (NEG, style.GRAY, "NO — a zero never lies", style.RED),
            (FP, style.YELLOW, "MAYBE?!  false positive", style.YELLOW),
        ]
        for word, wcolor, verdict, vcolor in cases:
            tag = Text(f"query: {word}", font_size=32, color=wcolor,
                       font=style.SERIF).move_to([-4.6, 1.7, 0])
            arcs, pos = self.arcs_for(word, 1.7, wcolor)
            for a, p in zip(arcs, pos):
                a.put_start_and_end_on(
                    np.array([-4.6, 1.35, 0]),
                    self.cell_pos(p) + np.array([0, CELL / 2, 0]))
            v = self.verdict_label(verdict, vcolor)
            self.play(FadeIn(tag), Create(arcs), run_time=0.9)
            self.play(FadeIn(v), run_time=0.4)
            if word == FP:
                self.play(*[Flash(self.cells[p], color=style.YELLOW,
                                  flash_radius=0.45) for p in pos])
                anatomy = style.label(
                    "never inserted — but every bit was set by someone else",
                    size=26, color=style.YELLOW).to_edge(DOWN, buff=0.5)
                self.play(FadeIn(anatomy))
                self.wait(1.2)
                self.play(FadeOut(anatomy))
            else:
                self.wait(0.8)
            self.play(FadeOut(arcs), FadeOut(tag), FadeOut(v), run_time=0.4)

    def beat_theory(self):
        self.play(FadeOut(self.title))
        chain = style.eq(
            r"P[\text{bit still }0] = \left(1-\tfrac{1}{m}\right)^{kn}"
            r"\ \approx\ e^{-kn/m}",
            size=34,
        ).to_edge(UP, buff=0.5)
        fp = style.eq(
            r"P[\text{false positive}] \approx \left(1 - e^{-kn/m}\right)^{k},"
            r"\qquad k^{*} = \tfrac{m}{n}\ln 2",
            size=34, color=style.YELLOW,
        ).next_to(chain, DOWN, buff=0.3)
        self.play(Write(chain))
        self.play(Write(fp))

        # curve: FP vs n for our m,k — with the empirical dot from the sim
        axes = Axes(
            x_range=[0, 30, 10], y_range=[0, 1, 0.25],
            x_length=6.4, y_length=2.4, tips=False,
            axis_config={"stroke_color": style.GRAY, "include_ticks": True},
        ).shift(DOWN * 2.0 + LEFT * 2.6)
        curve = axes.plot(lambda n: sk.fp_theoretical(M, K, max(n, 0.01)),
                          x_range=[0, 30], color=style.YELLOW)
        xlab = style.label("items inserted (m = 24, k = 3)", size=22)
        xlab.next_to(axes, DOWN, buff=0.2)
        emp = sk.fp_empirical(M, K, len(WORDS), probes=2000, seed=5)
        dot = axes.coords_to_point(len(WORDS), emp)
        from manim import Dot  # local import to keep header tidy
        emp_dot = Dot(dot, color=style.TEAL, radius=0.07)
        emp_tag = style.label(f"measured now: {emp:.2f}", size=22,
                              color=style.TEAL)
        emp_tag.next_to(emp_dot, UP + RIGHT, buff=0.1)
        note = style.label(
            "the shadow fills up —\nlies get more likely",
            size=24, color=style.GRAY).shift(DOWN * 2.2 + RIGHT * 3.6)

        self.play(Create(axes), Create(curve), FadeIn(xlab))
        self.play(FadeIn(emp_dot), FadeIn(emp_tag), FadeIn(note))
        self.wait(1.6)
        self.play(*map(FadeOut, [chain, fp, axes, curve, xlab, emp_dot,
                                 emp_tag, note, self.cells, self.zeros]))

    def beat_count_min(self):
        header = style.title_card(
            "Counts, not just membership: count-min sketch",
            "bits become counters; d independent rows; answer = the minimum",
        ).to_edge(UP)
        self.play(FadeIn(header))
        self.header = header

        w, d = 9, 3
        cms = sk.CountMinSketch(w, d)
        stream = (["dai"] * 6 + ["fro"] * 3 + ["ele"] * 2
                  + ["sna", "gro", "cof"] * 1)
        for it in stream:
            cms.update(it)

        grid = VGroup()
        self.cms_labels = {}
        for j in range(d):
            for i in range(w):
                sq = Square(0.62).set_stroke(style.GRAY, 1.5)
                sq.move_to([(i - (w - 1) / 2) * 0.68, 0.9 - j * 0.72, 0])
                val = int(cms.table[j, i])
                lab = style.eq(str(val), size=24,
                               color=style.WHITE if val else style.GRAY)
                lab.move_to(sq)
                if val:
                    sq.set_fill(style.BLUE, 0.12 + 0.10 * min(val, 6))
                grid.add(VGroup(sq, lab))
                self.cms_labels[(j, i)] = sq
        row_tags = VGroup(*[
            style.eq(rf"h_{{{j+1}}}", size=26, color=style.TEAL)
            .move_to([-(w - 1) / 2 * 0.68 - 0.8, 0.9 - j * 0.72, 0])
            for j in range(d)
        ])
        self.play(FadeIn(grid, lag_ratio=0.02), FadeIn(row_tags), run_time=1.6)

        # query the heavy hitter
        target = "dai"
        true_count = stream.count(target)
        rows = cms.row_estimates(target)
        pos = cms.positions(target)
        q = Text(f'query: "{target}"  (true count {true_count})',
                 font_size=30, color=style.YELLOW, font=style.SERIF)
        q.move_to([0, 2.05, 0])
        self.play(FadeIn(q))
        ests = VGroup()
        for j, (r, p) in enumerate(zip(rows, pos)):
            cell = self.cms_labels[(j, p)]
            self.play(cell.animate.set_stroke(style.YELLOW, 4), run_time=0.35)
            e = style.eq(str(int(r)), size=30,
                         color=style.GREEN if r == rows.min() else style.RED)
            e.move_to([(w - 1) / 2 * 0.68 + 1.1, 0.9 - j * 0.72, 0])
            ests.add(e)
            self.play(FadeIn(e), run_time=0.25)
        answer = style.eq(
            rf"\hat{{c}} = \min_j = {int(rows.min())} \;\ge\; {true_count}"
            r"\ \text{(never under)}",
            size=32, color=style.GREEN,
        ).to_edge(DOWN, buff=0.9)
        self.play(Write(answer))
        self.wait(1.5)
        self.play(*map(FadeOut, [grid, row_tags, q, ests, answer,
                                 self.header]))

    def beat_certificate(self):
        lines = VGroup(
            style.eq(r"\mathbb{E}[\text{row excess}] \;\le\; \frac{\varepsilon t}{e}"
                     r"\qquad (w = \lceil e/\varepsilon\rceil)", size=36),
            style.eq(r"\text{Markov:}\quad P\big[\text{row error} > \varepsilon t\big]"
                     r" \;\le\; \tfrac{1}{e}", size=36),
            style.eq(r"d\text{ independent rows:}\quad"
                     r" P\big[\hat{c} > c + \varepsilon t\big] \le e^{-d} \le \delta"
                     r"\qquad (d = \lceil \ln(1/\delta)\rceil)",
                     size=36, color=style.YELLOW),
        ).arrange(DOWN, buff=0.55)
        for line in lines:
            self.play(Write(line))
            self.wait(0.6)
        coda = style.label(
            "same family, same trick: MinHash, LSH, Flajolet–Martin — hash once, lie predictably",
            size=26,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(coda))
        self.wait(2.2)
