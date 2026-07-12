"""Demo 11 — Ridge Regression: buying accuracy with bias.

OLS throws darts with perfect aim and a shaky hand. Ridge steadies the hand
by aiming slightly off-target on purpose — and the MSE slope at lambda = 0 is
always negative, so some ridge always helps (CS229 PS2.2(d)).

Render:
    manim render -qh scenes/ridge_bias_variance.py RidgeBiasVariance
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
    Cross,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    ReplacementTransform,
    Scene,
    ValueTracker,
    VGroup,
    Write,
    always_redraw,
    rate_functions,
)

from core import style  # noqa: E402
from core.sims import ridge as rg  # noqa: E402

style.apply_house_style()

X = rg.make_design(n=40, gammas=(0.4, 8.0), seed=9)
W_STAR = np.array([1.1, 0.8])
SIGMA = 0.8
LAM_GRID = np.concatenate([[0.0], np.geomspace(0.02, 18, 60)])
CLOUDS = {i: rg.estimate_cloud(X, W_STAR, SIGMA, l, trials=70, seed=4)
          for i, l in enumerate(LAM_GRID)}
CURVES = rg.curves(X, W_STAR, SIGMA, np.linspace(0, 12, 300))
LSTAR = float(CURVES["lams"][np.argmin(CURVES["mse"])])
S = 1.15
CENTER = LEFT * 3.1 + DOWN * 0.4


def pt(w):
    return np.array([(w[0] - W_STAR[0]) * S, (w[1] - W_STAR[1]) * S, 0.0]) + CENTER


class RidgeBiasVariance(Scene):
    def construct(self):
        self.beat_darts()
        self.beat_dial()
        self.beat_decomposition()
        self.beat_ucurve()
        self.beat_shrinkage()

    def beat_darts(self):
        title = style.title_card(
            "Ridge — buying accuracy with bias",
            "OLS: perfect aim, shaky hand. Each dot is the estimate from one fresh noise draw.",
        ).to_edge(UP)
        self.play(FadeIn(title))
        self.title = title

        rings = VGroup(*[
            Dot(pt(W_STAR), radius=0.001).set_opacity(0)
        ])
        bullseye = VGroup(
            *[Line(pt(W_STAR) + off * 0.18, pt(W_STAR) - off * 0.18,
                   stroke_color=style.GREEN, stroke_width=3)
              for off in (RIGHT, UP)],
        )
        wtag = style.eq(r"w^{*}", size=30, color=style.GREEN)
        wtag.next_to(bullseye, UP + RIGHT, buff=0.08)
        self.play(FadeIn(bullseye), FadeIn(wtag))

        cloud = CLOUDS[0]
        dots = VGroup(*[Dot(pt(w), radius=0.045, color=style.BLUE,
                            fill_opacity=0.6) for w in cloud])
        mean = cloud.mean(axis=0)
        cross = Cross(stroke_color=style.YELLOW, stroke_width=4,
                      scale_factor=0.12).move_to(pt(mean))
        tag = style.eq(r"\lambda = 0\ \text{(OLS)}", size=30,
                       color=style.WHITE).to_corner(DOWN + LEFT, buff=0.5)
        note = style.label("unbiased — the cross sits on the bullseye — but scattered",
                           size=24, color=style.GRAY).to_edge(DOWN, buff=0.4)
        note.shift(RIGHT * 1.5)
        self.play(FadeIn(dots, lag_ratio=0.02), run_time=2.0)
        self.play(FadeIn(cross), FadeIn(tag), FadeIn(note))
        self.wait(1.0)
        self.dots, self.cross, self.tag, self.note = dots, cross, tag, note
        self.bullseye = VGroup(bullseye, wtag)

    def beat_dial(self):
        for idx, label, blurb in [
            (24, r"\lambda = %.2f" % LAM_GRID[24],
             "tighter…  and starting to drift"),
            (45, r"\lambda = %.2f" % LAM_GRID[45],
             "very steady — aimed at the wrong spot"),
        ]:
            cloud = CLOUDS[idx]
            new_dots = VGroup(*[Dot(pt(w), radius=0.045, color=style.BLUE,
                                    fill_opacity=0.6) for w in cloud])
            mean = cloud.mean(axis=0)
            new_cross = Cross(stroke_color=style.YELLOW, stroke_width=4,
                              scale_factor=0.12).move_to(pt(mean))
            new_tag = style.eq(label, size=30, color=style.WHITE)
            new_tag.to_corner(DOWN + LEFT, buff=0.5)
            new_note = style.label(blurb, size=24, color=style.GRAY)
            new_note.to_edge(DOWN, buff=0.4).shift(RIGHT * 1.5)
            self.play(
                ReplacementTransform(self.dots, new_dots),
                ReplacementTransform(self.cross, new_cross),
                ReplacementTransform(self.tag, new_tag),
                ReplacementTransform(self.note, new_note),
                run_time=1.8,
            )
            self.dots, self.cross = new_dots, new_cross
            self.tag, self.note = new_tag, new_note
            self.wait(0.8)

    def beat_decomposition(self):
        decomp = style.eq(
            r"\underbrace{\mathbb{E}\|\hat{w}-w^{*}\|^{2}}_{\text{MSE}}",
            "=",
            r"\underbrace{\|\mathbb{E}\hat{w}-w^{*}\|^{2}}_{\text{bias}^2}",
            "+",
            r"\underbrace{\mathbb{E}\|\hat{w}-\mathbb{E}\hat{w}\|^{2}}_{\text{variance}}",
            size=34,
        ).shift(RIGHT * 3.2 + UP * 1.4)
        legend = VGroup(
            style.label("bias: cross → bullseye", size=24, color=style.YELLOW),
            style.label("variance: dots → cross", size=24, color=style.BLUE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.next_to(decomp, DOWN, buff=0.5)
        self.play(Write(decomp))
        self.play(FadeIn(legend))
        self.wait(1.6)
        self.play(*map(FadeOut, [decomp, legend, self.dots, self.cross,
                                 self.tag, self.note, self.bullseye,
                                 self.title]))

    def beat_ucurve(self):
        head = style.title_card(
            "The U-curve — and why λ = 0 is never the bottom",
            "bias² climbs from zero; variance falls from σ²Σ1/γᵢ; their sum dips first",
        ).to_edge(UP)
        self.play(FadeIn(head))
        self.head = head

        ymax = float(CURVES["mse"].max()) * 1.08
        axes = Axes(
            x_range=[0, 12, 2], y_range=[0, ymax, 0.5],
            x_length=8.4, y_length=4.1, tips=False,
            axis_config={"stroke_color": style.GRAY},
        ).shift(DOWN * 0.85 + LEFT * 1.3)
        xlab = style.eq(r"\lambda", size=30, color=style.GRAY)
        xlab.next_to(axes, DOWN, buff=0.15)

        lines = {}
        for key, color in [("bias2", style.YELLOW), ("var", style.BLUE),
                           ("mse", style.RED)]:
            lines[key] = axes.plot_line_graph(
                CURVES["lams"], CURVES[key], line_color=color,
                add_vertex_dots=False, stroke_width=3)
        tags = VGroup(
            style.eq(r"\text{bias}^2", size=26, color=style.YELLOW),
            style.eq(r"\text{variance}", size=26, color=style.BLUE),
            style.eq(r"\text{MSE}", size=28, color=style.RED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        tags.shift(RIGHT * 5.0 + UP * 0.5)

        star_i = int(np.argmin(CURVES["mse"]))
        star = Dot(axes.coords_to_point(CURVES["lams"][star_i],
                                        CURVES["mse"][star_i]),
                   color=style.GREEN, radius=0.08)
        star_tag = style.eq(rf"\lambda^{{*}} = {LSTAR:.2f}", size=28,
                            color=style.GREEN)
        star_tag.next_to(star, DOWN, buff=0.2)
        ols = Dot(axes.coords_to_point(0, CURVES["mse"][0]), color=style.RED,
                  radius=0.07)
        ols_tag = style.label("OLS", size=22, color=style.RED)
        ols_tag.next_to(ols, RIGHT, buff=0.12)

        slope = rg.mse_slope_at_zero(X, W_STAR, SIGMA)
        slope_tag = style.eq(
            rf"\left.\tfrac{{d\,\text{{MSE}}}}{{d\lambda}}\right|_{{0}}"
            rf" = {slope:.2f} < 0\ \text{{— always}}",
            size=28, color=style.GREEN,
        ).to_edge(DOWN, buff=0.35)
        slope_tag.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)

        self.play(Create(axes), FadeIn(xlab))
        self.play(Create(lines["bias2"]), FadeIn(tags[0]))
        self.play(Create(lines["var"]), FadeIn(tags[1]))
        self.play(Create(lines["mse"]), FadeIn(tags[2]), run_time=1.4)
        self.play(FadeIn(ols), FadeIn(ols_tag))
        self.play(FadeIn(star), Write(star_tag))
        self.play(Write(slope_tag))
        self.wait(1.8)
        self.play(*map(FadeOut, [axes, xlab, *lines.values(), tags, star,
                                 star_tag, ols, ols_tag, slope_tag,
                                 self.head]))

    def beat_shrinkage(self):
        head = style.eq(
            r"\hat{w}_\lambda\ \text{shrinks each eigen-direction by}\ "
            r"\frac{\gamma_i}{\gamma_i+\lambda}",
            size=36,
        ).to_edge(UP, buff=0.7)
        self.play(Write(head))

        lam = ValueTracker(0.0)
        gammas = np.linalg.eigvalsh(X.T @ X)

        def bars():
            f = gammas / (gammas + lam.get_value())
            group = VGroup()
            for i, (g, fac) in enumerate(zip(gammas, f)):
                rail = Rectangle(width=1.1, height=3.0)
                rail.set_stroke(style.GRAY, 1.5).set_fill(opacity=0)
                rail.move_to([i * 2.6 - 1.3, -0.4, 0])
                fill = Rectangle(width=1.1, height=max(3.0 * fac, 0.02))
                fill.set_fill(style.TEAL if i else style.YELLOW, 0.85)
                fill.set_stroke(width=0)
                fill.move_to([i * 2.6 - 1.3, -1.9 + 3.0 * fac / 2, 0])
                lab = style.eq(rf"\gamma = {g:.1f}", size=26,
                               color=style.TEAL if i else style.YELLOW)
                lab.next_to(rail, DOWN, buff=0.2)
                val = style.eq(rf"{fac:.2f}", size=26, color=style.WHITE)
                val.next_to(rail, UP, buff=0.15)
                group.add(rail, fill, lab, val)
            return group

        graph = always_redraw(bars)
        lam_read = always_redraw(lambda: style.eq(
            rf"\lambda = {lam.get_value():.1f}", size=32, color=style.WHITE,
        ).shift(RIGHT * 4.4 + UP * 0.6))
        self.add(graph, lam_read)
        self.play(lam.animate.set_value(8.0), run_time=4.0,
                  rate_func=rate_functions.ease_in_out_sine)
        note = style.label(
            "the weak-signal direction (small γ) is shrunk first and hardest —",
            size=26,
        ).to_edge(DOWN, buff=0.75)
        note2 = style.label(
            "exactly the direction where OLS variance explodes (σ²/γ)",
            size=26, color=style.GRAY,
        ).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(note), FadeIn(note2))
        self.wait(1.2)
        coda = style.eq(
            r"\text{PS2.2(d): for every } \sigma^2 > 0 \text{ there is a }"
            r"\lambda > 0 \text{ strictly better than OLS}",
            size=30, color=style.GREEN,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeOut(note), FadeOut(note2))
        self.play(Write(coda))
        self.wait(2.2)
