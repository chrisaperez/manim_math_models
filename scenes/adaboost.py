"""Demo 9 — AdaBoost: failing better every round.

Dots grow when misclassified; each stump's alpha comes from its weighted
error; the new weights make yesterday's stump exactly worthless (eps = 1/2);
and the training error is trapped under prod Z_t <= exp(-2 sum gamma^2).

Render:
    manim render -qh scenes/adaboost.py AdaBoost
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
    DashedVMobject,
    Dot,
    FadeIn,
    FadeOut,
    Flash,
    Line,
    Rectangle,
    ReplacementTransform,
    Scene,
    Square,
    Text,
    VGroup,
    VMobject,
    Write,
)

from core import style  # noqa: E402
from core.sims import adaboost as ab  # noqa: E402

style.apply_house_style()

X, Y = ab.demo_data()
ROUNDS = ab.run(X, Y, T=12)
HIST = ab.bound_history(X, Y, ROUNDS)
S = 0.82
CENTER = LEFT * 2.6 + DOWN * 0.35


def pt(v):
    return np.array([v[0] * S, v[1] * S, 0.0]) + CENTER


def weight_radius(w: float) -> float:
    return float(0.05 + 1.35 * np.sqrt(w) * 0.55)


class AdaBoost(Scene):
    def construct(self):
        self.beat_setup()
        self.beat_first_round()
        self.beat_more_rounds()
        self.beat_committee()
        self.beat_bound()

    def make_marks(self, weights):
        marks = VGroup()
        for x, y, w in zip(X, Y, weights):
            r = weight_radius(w)
            if y > 0:
                m = Square(2 * r).set_stroke(style.YELLOW, 2.5)
                m.set_fill(style.YELLOW, 0.35)
            else:
                m = Circle(radius=r).set_stroke(style.BLUE, 2.5)
                m.set_fill(style.BLUE, 0.35)
            m.move_to(pt(x))
            marks.add(m)
        return marks

    def stump_gfx(self, stump, alpha_op=0.10):
        R = 2.75
        if stump.axis == 0:
            line = Line(pt([stump.thresh, -R]), pt([stump.thresh, R]),
                        stroke_color=style.WHITE, stroke_width=3)
            pos_side = Rectangle(width=(R - stump.thresh) * S, height=2 * R * S)
            pos_side.move_to(pt([(stump.thresh + R) / 2, 0]))
            neg_side = Rectangle(width=(stump.thresh + R) * S, height=2 * R * S)
            neg_side.move_to(pt([(stump.thresh - R) / 2, 0]))
        else:
            line = Line(pt([-R, stump.thresh]), pt([R, stump.thresh]),
                        stroke_color=style.WHITE, stroke_width=3)
            pos_side = Rectangle(height=(R - stump.thresh) * S, width=2 * R * S)
            pos_side.move_to(pt([0, (stump.thresh + R) / 2]))
            neg_side = Rectangle(height=(stump.thresh + R) * S, width=2 * R * S)
            neg_side.move_to(pt([0, (stump.thresh - R) / 2]))
        plus, minus = (pos_side, neg_side) if stump.sign > 0 else (neg_side, pos_side)
        plus.set_fill(style.YELLOW, alpha_op).set_stroke(width=0)
        minus.set_fill(style.BLUE, alpha_op).set_stroke(width=0)
        return VGroup(line, plus, minus)

    def beat_setup(self):
        title = style.title_card(
            "AdaBoost — failing better every round",
            "a stump is barely better than a coin; a committee of them is not",
        ).to_edge(UP)
        self.play(FadeIn(title))
        self.title = title
        self.marks = self.make_marks(ROUNDS[0].weights_before)
        legend = VGroup(
            style.label("■ y = +1", size=22, color=style.YELLOW),
            style.label("● y = −1", size=22, color=style.BLUE),
            style.label("area = weight", size=22, color=style.GRAY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
        legend.to_corner(DOWN + LEFT, buff=0.4)
        self.legend = legend
        self.play(FadeIn(self.marks, lag_ratio=0.03), FadeIn(legend))
        self.wait(0.5)

    def beat_first_round(self):
        rnd = ROUNDS[0]
        gfx = self.stump_gfx(rnd.stump)
        chips = VGroup(
            style.eq(rf"\varepsilon_1 = {rnd.eps:.3f}", size=30,
                     color=style.RED),
            style.eq(rf"\alpha_1 = \tfrac12\ln\tfrac{{1-\varepsilon_1}}"
                     rf"{{\varepsilon_1}} = {rnd.alpha:.3f}", size=30,
                     color=style.GREEN),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        chips.shift(RIGHT * 4.0 + UP * 1.7)
        self.play(Create(gfx), Write(chips[0]))
        wrong = rnd.stump.predict(X) != Y
        flashes = [Flash(self.marks[i], color=style.RED, flash_radius=0.35)
                   for i in np.nonzero(wrong)[0]]
        self.play(*flashes)
        self.play(Write(chips[1]))

        # reweight: wrong grows, right shrinks
        new_marks = self.make_marks(rnd.weights_after)
        rule = style.eq(
            r"w_i \leftarrow \frac{w_i\, e^{-\alpha_t\, y_i h_t(x_i)}}{Z_t}",
            size=32, color=style.WHITE,
        ).next_to(chips, DOWN, buff=0.5).align_to(chips, LEFT)
        self.play(ReplacementTransform(self.marks, new_marks), Write(rule),
                  run_time=1.6)
        self.marks = new_marks

        # the self-erasing property
        gauge = style.eq(
            r"\text{weighted error of } h_1 \text{ under the new } w:"
            r"\quad 0.500",
            size=28, color=style.TEAL,
        ).to_edge(DOWN, buff=0.4)
        gauge.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)
        note = style.label("yesterday's stump is now a coin flip — diversity is forced",
                           size=24, color=style.TEAL)
        note.next_to(gauge, UP, buff=0.15)
        self.play(Write(gauge), FadeIn(note))
        self.wait(1.4)
        self.play(FadeOut(gauge), FadeOut(note), FadeOut(rule),
                  FadeOut(chips), gfx.animate.set_opacity(0.25))
        self.old_gfx = VGroup(gfx)

    def beat_more_rounds(self):
        for t in (1, 2):
            rnd = ROUNDS[t]
            gfx = self.stump_gfx(rnd.stump)
            chip = style.eq(
                rf"t={t + 1}:\ \varepsilon = {rnd.eps:.2f},\ "
                rf"\alpha = {rnd.alpha:.2f}",
                size=28, color=style.WHITE,
            ).shift(RIGHT * 4.0 + UP * 1.9)
            new_marks = self.make_marks(rnd.weights_after)
            self.play(Create(gfx), FadeIn(chip), run_time=0.9)
            self.play(ReplacementTransform(self.marks, new_marks), run_time=0.9)
            self.marks = new_marks
            self.play(FadeOut(chip), gfx.animate.set_opacity(0.25),
                      run_time=0.5)
            self.old_gfx.add(gfx)

    def beat_committee(self):
        # decision-region heat for the 3-round committee
        G = 26
        R = 2.75
        gs = np.linspace(-R, R, G)
        cells = VGroup()
        pts = np.array([[a, b] for a in gs for b in gs])
        f = ab.committee_scores(pts, ROUNDS[:3])
        for p, v in zip(pts, f):
            c = Square(2 * R * S / G)
            c.set_stroke(width=0)
            c.set_fill(style.YELLOW if v > 0 else style.BLUE,
                       0.25 * min(abs(v) / 1.2, 1.0))
            c.move_to(pt(p))
            cells.add(c)
        eq = style.eq(
            r"H(x) = \operatorname{sign}\!\Big(\sum_t \alpha_t h_t(x)\Big)",
            size=34, color=style.WHITE,
        ).shift(RIGHT * 4.0 + UP * 1.9)
        err = style.eq(
            rf"\text{{train error after 3 rounds: }} {HIST['train_error'][2]:.3f}",
            size=28, color=style.GRAY,
        ).next_to(eq, DOWN, buff=0.4).align_to(eq, LEFT)
        self.play(FadeOut(self.old_gfx), FadeIn(cells), Write(eq))
        self.play(Write(err))
        self.wait(1.4)
        self.play(FadeOut(cells), FadeOut(eq), FadeOut(err),
                  FadeOut(self.marks), FadeOut(self.title),
                  FadeOut(self.legend))
        self.cells = None

    def beat_bound(self):
        chain = style.eq(
            r"\mathbb{1}[H(x)\neq y] \le e^{-y f(x)}",
            r"\ \Rightarrow\ \text{err} \le \prod_t Z_t",
            r"= \prod_t 2\sqrt{\varepsilon_t(1-\varepsilon_t)}",
            r"\le e^{-2\sum_t \gamma_t^2}",
            size=32,
        ).to_edge(UP, buff=0.5)
        self.play(Write(chain), run_time=2.0)

        T = len(ROUNDS)
        axes = Axes(
            x_range=[1, T, 1], y_range=[0, 1, 0.25],
            x_length=8.6, y_length=3.9, tips=False,
            axis_config={"stroke_color": style.GRAY},
        ).shift(DOWN * 0.9 + LEFT * 1.2)
        xlab = style.label("boosting rounds t", size=24).next_to(axes, DOWN, buff=0.2)

        def series_line(vals, color, dashed=False):
            line = VMobject(stroke_color=color, stroke_width=3)
            line.set_points_as_corners(
                [axes.coords_to_point(t + 1, v) for t, v in enumerate(vals)])
            return DashedVMobject(line, num_dashes=40) if dashed else line

        err_line = series_line(HIST["train_error"], style.RED)
        pz_line = series_line(HIST["prod_z"], style.YELLOW)
        eb_line = series_line(HIST["exp_bound"], style.TEAL, dashed=True)
        tags = VGroup(
            style.eq(r"\text{train error}", size=26, color=style.RED),
            style.eq(r"\textstyle\prod_t Z_t\ \text{(exp. loss)}", size=26,
                     color=style.YELLOW),
            style.eq(r"e^{-2\sum\gamma_t^2}", size=26, color=style.TEAL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        tags.shift(RIGHT * 4.9 + UP * 0.6)

        self.play(Create(axes), FadeIn(xlab))
        self.play(Create(eb_line), FadeIn(tags[2]))
        self.play(Create(pz_line), FadeIn(tags[1]))
        self.play(Create(err_line), FadeIn(tags[0]), run_time=1.4)
        note = style.label(
            "the error never touches the curves above it — squeezed to zero geometrically",
            size=24,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(note))
        self.wait(1.2)
        coda = style.label(
            "boosting = greedy coordinate descent on the exponential loss  E[e^{−yf(x)}]",
            size=24, color=style.GRAY,
        ).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(note, coda))
        self.wait(2.2)
