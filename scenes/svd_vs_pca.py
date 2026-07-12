"""Demo 3 — SVD vs PCA: one decomposition, two stories.

Act I  (PCA): the Pythagorean trade-off — sweeping a projection line through a
data cloud, keep + lose is pinned, so max-variance IS min-error.
Act II (SVD): any matrix is rotate → stretch → rotate; the stretch axes are
exactly the variance axes, and M^T M = V S^2 V^T closes the loop.

Render:
    manim render -qh scenes/svd_vs_pca.py SvdVsPca
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from manim import (  # noqa: E402
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    ApplyMatrix,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    NumberPlane,
    Polygon,
    Rectangle,
    ReplacementTransform,
    Scene,
    TransformMatchingTex,
    ValueTracker,
    VGroup,
    VMobject,
    Write,
    always_redraw,
    rate_functions,
)

from core import style  # noqa: E402
from core.sims import svd_pca as sp  # noqa: E402

style.apply_house_style()

CLOUD = sp.make_cloud(n=42, angle=0.6, sig1=1.9, sig2=0.55, seed=11)
VSTAR = sp.principal_dir(CLOUD)
THETA_STAR = float(np.arctan2(VSTAR[1], VSTAR[0]))
M = sp.rotation(0.9) @ np.diag([2.2, 0.8]) @ sp.rotation(-0.35)
STAGES = sp.svd_stages(M)
S = 0.62  # world scale


def pt(v) -> np.ndarray:
    return np.array([v[0] * S, v[1] * S, 0.0])


class SvdVsPca(Scene):
    def construct(self):
        self.act1_pythagoras()
        self.act1_sweep()
        self.act1_rayleigh()
        self.act2_stages()
        self.act2_close_the_loop()

    # ---- Act I: one point, one triangle -------------------------------
    def act1_pythagoras(self):
        title = style.title_card(
            "SVD vs PCA — one decomposition, two stories",
            "Act I: what does “direction of most variance” even minimize?",
        ).to_edge(UP)
        self.play(FadeIn(title))
        self.wait(0.8)
        self.play(FadeOut(title))

        dots = VGroup(*[Dot(pt(p), radius=0.05, color=style.BLUE,
                            fill_opacity=0.8) for p in CLOUD])
        self.play(FadeIn(dots, lag_ratio=0.02), run_time=1.4)
        self.dots = dots

        self.theta = ValueTracker(0.25)

        def line_builder():
            u = sp.unit(self.theta.get_value())
            return Line(pt(-4.6 * u), pt(4.6 * u), stroke_color=style.YELLOW,
                        stroke_width=3)
        self.u_line = always_redraw(line_builder)
        self.play(Create(self.u_line))

        # one highlighted point makes the right triangle
        x = CLOUD[7]
        def tri_builder():
            u = sp.unit(self.theta.get_value())
            proj = (x @ u) * u
            return VGroup(
                Polygon(pt([0, 0]), pt(x), pt(proj),
                        stroke_width=0, fill_color=style.TEAL, fill_opacity=0.18),
                Line(pt([0, 0]), pt(x), stroke_color=style.WHITE, stroke_width=2.5),
                Line(pt(proj), pt(x), stroke_color=style.RED, stroke_width=3),
                Line(pt([0, 0]), pt(proj), stroke_color=style.GREEN, stroke_width=4),
                Dot(pt(x), radius=0.07, color=style.WHITE),
            )
        tri = always_redraw(tri_builder)
        self.add(tri)

        pyth = style.eq(
            r"\|x\|^2", "=", r"\underbrace{\|f_u(x)\|^2}_{\text{kept}}",
            "+", r"\underbrace{\|x - f_u(x)\|^2}_{\text{lost}}",
            size=38,
        ).to_edge(UP, buff=0.4)
        pyth.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)
        self.play(Write(pyth))
        self.play(self.theta.animate.set_value(0.9), run_time=2.4,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(self.theta.animate.set_value(0.35), run_time=2.0,
                  rate_func=rate_functions.ease_in_out_sine)
        self.wait(0.5)
        tri.clear_updaters()
        self.play(FadeOut(tri))
        self.pyth = pyth

    # ---- Act I: the sweep with pinned total ----------------------------
    def act1_sweep(self):
        BARX = 5.05
        total = float(np.sum(CLOUD**2))
        BARH = 4.6  # full bar height for `total`

        def bars_builder():
            u = sp.unit(self.theta.get_value())
            p = sp.project(CLOUD, u)
            keep_h = BARH * p["keep"] / total
            lose_h = BARH * p["lose"] / total
            keep = Rectangle(width=0.5, height=max(keep_h, 1e-3),
                             fill_color=style.GREEN, fill_opacity=0.9,
                             stroke_width=0)
            lose = Rectangle(width=0.5, height=max(lose_h, 1e-3),
                             fill_color=style.RED, fill_opacity=0.9,
                             stroke_width=0)
            keep.move_to([BARX, -2.15 + keep_h / 2, 0])
            lose.move_to([BARX, -2.15 + keep_h + lose_h / 2, 0])
            frame = Rectangle(width=0.56, height=BARH).set_stroke(style.GRAY, 1.5)
            frame.move_to([BARX, -2.15 + BARH / 2, 0])
            return VGroup(keep, lose, frame)

        bars = always_redraw(bars_builder)
        keep_tag = style.eq(r"\text{kept}", size=26, color=style.GREEN)
        lose_tag = style.eq(r"\text{lost}", size=26, color=style.RED)
        pin_tag = style.eq(r"\text{sum pinned} = \textstyle\sum_i \|x^{(i)}\|^2",
                           size=24, color=style.GRAY)
        keep_tag.move_to([BARX + 1.0, -1.2, 0])
        lose_tag.move_to([BARX + 1.0, 1.2, 0])
        pin_tag.move_to([BARX - 0.1, 2.75, 0])

        # per-point projection whiskers
        def whiskers_builder():
            u = sp.unit(self.theta.get_value())
            p = sp.project(CLOUD, u)
            return VGroup(*[
                Line(pt(pr), pt(x), stroke_color=style.RED,
                     stroke_width=1.2, stroke_opacity=0.5)
                for pr, x in zip(p["proj"], CLOUD)
            ])
        whiskers = always_redraw(whiskers_builder)

        self.add(whiskers)
        self.play(FadeIn(bars), FadeIn(keep_tag), FadeIn(lose_tag),
                  FadeIn(pin_tag))
        self.play(self.theta.animate.set_value(np.pi * 0.97), run_time=5.0,
                  rate_func=rate_functions.ease_in_out_sine)
        self.wait(0.3)
        # snap to the principal direction
        snap = style.eq(
            r"\text{max kept} \iff \text{min lost} \iff u = v_1",
            size=30, color=style.YELLOW,
        ).to_edge(DOWN, buff=0.4)
        snap.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)
        self.play(self.theta.animate.set_value(THETA_STAR), Write(snap),
                  run_time=1.6)
        self.wait(1.2)

        whiskers.clear_updaters()
        bars.clear_updaters()
        self.play(FadeOut(VGroup(whiskers, bars, keep_tag, lose_tag, pin_tag,
                                 snap)))

    # ---- Act I -> bridge: the Rayleigh quotient -------------------------
    def act1_rayleigh(self):
        chain = style.eq(
            r"\max_{\|u\|=1}\ \sum_i (u^{\top}x^{(i)})^2",
            "=", r"\max_{\|u\|=1}\ u^{\top}\!\big(M^{\top}M\big)\,u",
            size=36,
        ).to_edge(UP, buff=0.4)
        chain.add_background_rectangle(color=style.BACKGROUND, opacity=0.92)
        verdict = style.eq(
            r"\Rightarrow\ u = \text{top eigenvector of } M^{\top}M",
            size=32, color=style.TEAL,
        ).next_to(chain, DOWN, buff=0.3)
        verdict.add_background_rectangle(color=style.BACKGROUND, opacity=0.92)
        self.play(ReplacementTransform(self.pyth, chain))
        self.play(Write(verdict))
        self.wait(1.4)
        self.u_line.clear_updaters()
        self.play(FadeOut(VGroup(chain, verdict, self.u_line, self.dots)))

    # ---- Act II: rotate, stretch, rotate --------------------------------
    def act2_stages(self):
        header = style.title_card(
            "Act II: what does a matrix do?",
            "M = U Σ Vᵀ — every matrix is rotate → stretch → rotate",
        ).to_edge(UP)
        self.play(FadeIn(header))
        self.wait(0.8)
        self.play(header.animate.scale(0.62).to_corner(UP + LEFT, buff=0.3))

        plane = NumberPlane(
            x_range=[-6, 6, 1], y_range=[-4, 4, 1],
            background_line_style={"stroke_color": "#242424", "stroke_width": 1},
        )
        circle = Circle(radius=1.7, color=style.BLUE, stroke_width=3)
        spokes = VGroup(*[
            Line(ORIGIN, 1.7 * np.array([np.cos(a), np.sin(a), 0]),
                 stroke_color=style.BLUE, stroke_width=1.2, stroke_opacity=0.5)
            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ])
        shape = VGroup(circle, spokes)
        self.play(FadeIn(plane), Create(circle), FadeIn(spokes))

        eq = style.eq("M", "=", "U", r"\Sigma", "V^{\\top}", size=44)
        eq.to_corner(UP + RIGHT, buff=0.5)
        eq.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)
        self.play(Write(eq))

        Vt, Sig, U = STAGES["Vt"], np.diag(STAGES["S"]), STAGES["U"]
        stages = [
            ("1 — Vᵀ: a rigid rotation", Vt, style.TEAL),
            ("2 — Σ: stretch along the axes  (σ₁ = %.1f, σ₂ = %.1f)"
             % (STAGES["S"][0], STAGES["S"][1]), Sig, style.YELLOW),
            ("3 — U: a final rotation", U, style.GREEN),
        ]
        current_tag = None
        for text, mat, color in stages:
            tag = style.label(text, size=28, color=color).to_edge(DOWN, buff=0.5)
            anims = [ApplyMatrix(mat, shape, run_time=1.8)]
            if current_tag is None:
                anims.append(FadeIn(tag))
            else:
                anims.append(ReplacementTransform(current_tag, tag))
            self.play(*anims)
            current_tag = tag
            self.wait(0.5)

        # overlay singular axes on the resulting ellipse
        ax = VGroup()
        for i, color in [(0, style.YELLOW), (1, style.TEAL)]:
            direction = U[:, i] * STAGES["S"][i] * 1.7
            ax.add(Line(pt([0, 0]), np.array([direction[0], direction[1], 0]),
                        stroke_color=color, stroke_width=5))
        note = style.label(
            "the stretch axes are exactly the variance axes of Act I",
            size=26, color=style.WHITE,
        ).to_edge(DOWN, buff=0.5)
        self.play(Create(ax), ReplacementTransform(current_tag, note))
        self.wait(1.4)
        self.play(FadeOut(VGroup(plane, shape, ax, note, eq, header)))

    # ---- The loop closes: M^T M = V S^2 V^T -----------------------------
    def act2_close_the_loop(self):
        e1 = style.eq(r"M^{\top}M", "=", r"(U\Sigma V^{\top})^{\top}",
                      r"(U\Sigma V^{\top})", size=44)
        e2 = style.eq(r"M^{\top}M", "=", r"V\Sigma^{\top}",
                      r"\underbrace{U^{\top}U}_{=\,I}", r"\Sigma V^{\top}",
                      size=44)
        e3 = style.eq(r"M^{\top}M", "=", r"V\,\Sigma^2\,V^{\top}", size=48)
        for e in (e1, e2, e3):
            e.move_to(UP * 1.2)

        self.play(Write(e1))
        self.wait(0.8)
        self.play(TransformMatchingTex(e1, e2))
        self.wait(1.0)
        self.play(TransformMatchingTex(e2, e3))
        self.wait(0.6)

        badge = style.eq(
            r"\text{PCA directions} = V, \qquad \lambda_i = \sigma_i^2",
            size=38, color=style.YELLOW,
        ).next_to(e3, DOWN, buff=0.7)
        self.play(Write(badge))
        self.wait(1.0)

        coda = style.eq(
            r"\text{coda: } M^{+} = V\,\Sigma^{+}U^{\top}"
            r"\ \text{— run the movie backwards (CS205L 3.2)}",
            size=28, color=style.GRAY,
        ).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(coda))
        self.wait(2.2)
