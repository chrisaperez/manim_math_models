"""Demo 10 — KL Divergence & Maximum Likelihood.

How wrong is it to believe Q when the world runs on P? KL(P||Q) — nonnegative
by Jensen, direction-sensitive, additive over joints. And fitting a model by
maximum likelihood is exactly minimizing KL from the empirical distribution:
the two curves are mirror images.

Render:
    manim render -qh scenes/kl_mle.py KlMle
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
from core.sims import kl_mle as km  # noqa: E402

style.apply_house_style()

P = np.array([0.9, 0.05, 0.05, 0.0])[:3]
P = np.array([0.55, 0.30, 0.15])
Q = np.array([0.20, 0.35, 0.45])

COUNTS = km.sample_counts(theta_true=0.62, n_samples=400, seed=5)
P_HAT = km.empirical_dist(COUNTS)
THETAS = np.linspace(0.05, 0.95, 181)
KL_C = km.kl_curve(P_HAT, THETAS)
LL_C = km.loglik_curve(P_HAT, THETAS)
THETA_MLE = km.mle_theta(COUNTS)


def bar_chart(dist, color, x0, width=0.75, hmax=2.2, labels=None):
    bars = VGroup()
    for i, p in enumerate(dist):
        b = Rectangle(width=width, height=max(hmax * p, 0.02))
        b.set_fill(color, 0.8).set_stroke(color, 1)
        b.move_to([x0 + i * (width + 0.25), -1.3 + hmax * p / 2, 0])
        bars.add(b)
        if labels:
            lab = style.eq(labels[i], size=22, color=style.GRAY)
            lab.move_to([x0 + i * (width + 0.25), -1.62, 0])
            bars.add(lab)
    return bars


class KlMle(Scene):
    def construct(self):
        self.beat_two_beliefs()
        self.beat_jensen()
        self.beat_chain()
        self.beat_mle_bridge()

    def beat_two_beliefs(self):
        title = style.title_card(
            "KL divergence — the price of believing Q",
            "the world runs on P; every observation costs you log P(x)/Q(x)",
        ).to_edge(UP)
        self.play(FadeIn(title))

        p_bars = bar_chart(P, style.TEAL, -4.6, labels=["a", "b", "c"])
        q_bars = bar_chart(Q, style.YELLOW, 1.4, labels=["a", "b", "c"])
        p_tag = style.eq("P", size=36, color=style.TEAL).move_to([-3.6, 1.4, 0])
        q_tag = style.eq("Q", size=36, color=style.YELLOW).move_to([2.4, 1.4, 0])
        self.play(FadeIn(p_bars), FadeIn(q_bars), FadeIn(p_tag), FadeIn(q_tag))

        fwd = style.eq(
            rf"KL(P\|Q) = \sum_x P(x)\log\tfrac{{P(x)}}{{Q(x)}} = {km.kl(P, Q):.3f}",
            size=32, color=style.WHITE,
        ).to_edge(DOWN, buff=0.9)
        rev = style.eq(
            rf"KL(Q\|P) = {km.kl(Q, P):.3f} \;\neq\; KL(P\|Q)"
            r"\quad\text{— not a distance}",
            size=32, color=style.RED,
        ).to_edge(DOWN, buff=0.9)
        self.play(Write(fwd))
        self.wait(1.2)
        self.play(ReplacementTransform(fwd, rev))
        self.wait(1.2)
        self.play(*map(FadeOut, [p_bars, q_bars, p_tag, q_tag, rev, title]))

    def beat_jensen(self):
        axes = Axes(
            x_range=[0.05, 3.2, 1], y_range=[-2.4, 1.4, 1],
            x_length=6.4, y_length=3.8, tips=False,
            axis_config={"stroke_color": style.GRAY},
        ).shift(LEFT * 2.4 + DOWN * 0.5)
        curve = axes.plot(np.log, x_range=[0.09, 3.2], color=style.BLUE)
        clabel = style.eq(r"\log z", size=30, color=style.BLUE)
        clabel.next_to(axes.coords_to_point(2.9, np.log(2.9)), UP, buff=0.15)

        # chord between two points sits below the concave curve
        z1, z2 = 0.35, 2.6
        chord = Line(axes.coords_to_point(z1, np.log(z1)),
                     axes.coords_to_point(z2, np.log(z2)),
                     stroke_color=style.RED, stroke_width=3)
        mid = 0.5 * (z1 + z2)
        d_curve = Dot(axes.coords_to_point(mid, np.log(mid)),
                      color=style.BLUE, radius=0.06)
        d_chord = Dot(axes.coords_to_point(mid, 0.5 * (np.log(z1) + np.log(z2))),
                      color=style.RED, radius=0.06)
        gap = DashedLine(d_chord.get_center(), d_curve.get_center(),
                         stroke_color=style.GRAY)

        steps = VGroup(
            style.eq(r"\mathbb{E}[\log Z] \;\le\; \log \mathbb{E}[Z]",
                     size=34, color=style.WHITE),
            style.eq(r"-KL(P\|Q) = \mathbb{E}_P\Big[\log\tfrac{Q}{P}\Big]"
                     r"\le \log \mathbb{E}_P\Big[\tfrac{Q}{P}\Big] = \log 1 = 0",
                     size=30, color=style.WHITE),
            style.eq(r"\Rightarrow\ KL \ge 0,\ \text{ zero iff } P = Q",
                     size=32, color=style.GREEN),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        steps.shift(RIGHT * 2.7 + UP * 1.9)
        steps[1].scale(0.78).align_to(steps[0], LEFT)

        head = style.label("why it can never be negative: Jensen", size=26,
                           color=style.WHITE).to_edge(UP, buff=0.5)
        self.play(FadeIn(head), Create(axes), Create(curve), FadeIn(clabel))
        self.play(Create(chord), FadeIn(d_curve), FadeIn(d_chord), Create(gap))
        self.play(Write(steps[0]))
        self.play(Write(steps[1]))
        self.play(Write(steps[2]))
        self.wait(1.6)
        self.play(*map(FadeOut, [axes, curve, clabel, chord, d_curve, d_chord,
                                 gap, steps, head]))

    def beat_chain(self):
        rng = np.random.default_rng(2)
        jp = rng.uniform(0.05, 1.0, size=(3, 3)); jp /= jp.sum()
        jq = rng.uniform(0.05, 1.0, size=(3, 3)); jq /= jq.sum()
        parts = km.chain_rule_parts(jp, jq)

        chain = style.eq(
            r"KL\big(P(X,Y)\,\|\,Q(X,Y)\big)", "=",
            r"KL\big(P(X)\|Q(X)\big)", "+",
            r"KL\big(P(Y|X)\,\|\,Q(Y|X)\big)",
            size=32,
        ).to_edge(UP, buff=0.6)
        nums = style.eq(
            rf"{parts['total']:.4f}", "=", rf"{parts['marginal']:.4f}", "+",
            rf"{parts['conditional']:.4f}",
            size=40, color=style.YELLOW,
        ).next_to(chain, DOWN, buff=0.8)
        note = style.label(
            "surprise decomposes: what X alone tells you, plus what Y adds given X",
            size=26,
        ).next_to(nums, DOWN, buff=0.8)
        self.play(Write(chain))
        self.play(Write(nums))
        self.play(FadeIn(note))
        self.wait(1.8)
        self.play(*map(FadeOut, [chain, nums, note]))

    def beat_mle_bridge(self):
        head = style.title_card(
            "Fitting by maximum likelihood IS minimizing KL",
            "empirical P̂ from 400 samples of Binomial(5, θ*= 0.62); sweep the model θ",
        ).to_edge(UP)
        self.play(FadeIn(head))

        # left: empirical histogram vs model pmf that follows theta
        theta = ValueTracker(0.20)
        hist_bars = bar_chart(P_HAT, style.TEAL, -5.9, width=0.5, hmax=3.6,
                              labels=[str(k) for k in range(6)])

        def model_bars():
            pmf = km.binom_pmf(theta.get_value())
            bars = VGroup()
            for i, p in enumerate(pmf):
                b = Rectangle(width=0.22, height=max(3.6 * p, 0.015))
                b.set_fill(style.YELLOW, 0.95).set_stroke(width=0)
                b.move_to([-5.9 + i * 0.75 + 0.18, -1.3 + 3.6 * p / 2, 0])
                bars.add(b)
            return bars

        model = always_redraw(model_bars)
        tags = VGroup(
            style.eq(r"\hat{P}\ \text{(data)}", size=26, color=style.TEAL),
            style.eq(r"P_\theta\ \text{(model)}", size=26, color=style.YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([-4.6, 1.6, 0])

        # right: the two curves, mirrored
        axes = Axes(
            x_range=[0.05, 0.95, 0.2], y_range=[-3.4, 1.6, 1],
            x_length=6.0, y_length=4.2, tips=False,
            axis_config={"stroke_color": style.GRAY},
        ).shift(RIGHT * 2.9 + DOWN * 0.55)
        kl_clip = np.minimum(KL_C, 1.55)      # keep curves inside the axes
        ll_clip = np.maximum(LL_C, -3.35)
        kl_plot = axes.plot_line_graph(
            THETAS, kl_clip, line_color=style.RED, add_vertex_dots=False,
            stroke_width=3)
        ll_plot = axes.plot_line_graph(
            THETAS, ll_clip, line_color=style.GREEN, add_vertex_dots=False,
            stroke_width=3)
        kl_tag = style.eq(r"KL(\hat{P}\|P_\theta)", size=26, color=style.RED)
        kl_tag.next_to(axes.coords_to_point(0.82, min(KL_C[-8], 1.4)), UP, buff=0.1)
        ll_tag = style.eq(r"\tfrac1n\sum \log P_\theta(x^{(i)})", size=26,
                          color=style.GREEN)
        ll_tag.next_to(axes.coords_to_point(0.85, LL_C[-20]), DOWN, buff=0.1)
        xlab = style.eq(r"\theta", size=28, color=style.GRAY)
        xlab.next_to(axes, DOWN, buff=0.15)

        def theta_marker():
            t = theta.get_value()
            return DashedLine(axes.coords_to_point(t, -3.4),
                              axes.coords_to_point(t, 1.6),
                              stroke_color=style.YELLOW, stroke_width=2)
        marker = always_redraw(theta_marker)

        self.play(FadeIn(hist_bars), FadeIn(tags), Create(axes), FadeIn(xlab))
        self.add(model, marker)
        self.play(Create(kl_plot), FadeIn(kl_tag))
        self.play(Create(ll_plot), FadeIn(ll_tag))
        self.play(theta.animate.set_value(0.9), run_time=3.0,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(theta.animate.set_value(THETA_MLE), run_time=2.0,
                  rate_func=rate_functions.ease_in_out_sine)

        star = Dot(axes.coords_to_point(THETA_MLE, KL_C.min()),
                   color=style.YELLOW, radius=0.07)
        verdict = style.eq(
            rf"\arg\min KL = \arg\max \text{{loglik}} = \bar{{x}}/5 = {THETA_MLE:.3f}",
            size=30, color=style.YELLOW,
        ).to_edge(DOWN, buff=0.4)
        verdict.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)
        self.play(FadeIn(star), Write(verdict))
        self.wait(1.0)

        identity = style.eq(
            r"KL(\hat{P}\|P_\theta) = -H(\hat{P}) - \tfrac1n\textstyle\sum\log P_\theta"
            r"\quad\text{— same curve, flipped by a constant}",
            size=26, color=style.GRAY,
        ).next_to(verdict, UP, buff=0.2)
        identity.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)
        self.play(FadeIn(identity))
        self.wait(2.2)
