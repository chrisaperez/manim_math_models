"""Demo 8 — Newton vs Gradient Descent: why curvature information wins.

The canyon from Demo 1, revisited: GD zig-zags ~kappa steps; Newton multiplies
by H^{-1} first — un-squashing the space to kappa = 1 — and lands in one.
Then the honest part: Rosenbrock, damping, and BFGS learning curvature as it
goes.

Render:
    manim render -qh scenes/newton_conditioning.py NewtonConditioning
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
    ApplyMatrix,
    Arrow,
    Axes,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    ReplacementTransform,
    Scene,
    Text,
    TransformMatchingTex,
    VGroup,
    VMobject,
    Write,
)

from core import style  # noqa: E402
from core.sims import gradient_descent as gd  # noqa: E402
from core.sims import newton as nt  # noqa: E402

style.apply_house_style()

LAMBDAS = np.array([0.3, 15.0])          # kappa = 50
ANGLE = np.deg2rad(25)
A = gd.make_quadratic(LAMBDAS, ANGLE)
X0 = gd.rotation(ANGLE) @ np.array([2.4, 0.9])
S = 0.95


def pt(v):
    return np.array([v[0] * S, v[1] * S, 0.0])


def polyline(path, color, width=3):
    line = VMobject(stroke_color=color, stroke_width=width)
    line.set_points_as_corners([pt(p) for p in path])
    return line


def contours(matrix, levels, scale=1.0):
    group = VGroup()
    for i, level in enumerate(levels):
        c = VMobject(stroke_color=style.BLUE, stroke_width=1.8,
                     stroke_opacity=0.8 - 0.12 * i)
        c.set_points_smoothly(
            [pt(p * scale) for p in gd.contour_ellipse(matrix, level)])
        group.add(c)
    return group


class NewtonConditioning(Scene):
    def construct(self):
        self.beat_zigzag()
        self.beat_one_leap()
        self.beat_algebra()
        self.beat_whiten()
        self.beat_rosenbrock()
        self.beat_ledger()

    def beat_zigzag(self):
        title = style.title_card(
            "Newton vs Gradient Descent",
            "same canyon as Demo 1 — κ = 50 this time",
        ).to_edge(UP)
        self.play(FadeIn(title))
        self.title = title

        self.curves = contours(A, [0.3, 1.0, 2.4, 4.5])
        self.play(Create(self.curves), run_time=1.2)

        _, grad, _ = nt.quadratic(A)
        alpha = float(2 / (LAMBDAS.sum()))
        path, _ = nt.gd_run(grad, X0, alpha, steps=260, tol=1e-6)
        n_gd = nt.gd_steps_to_tol(A, X0, tol=1e-6)
        traj = polyline(path, style.YELLOW, 2.5)
        counter = style.eq(
            rf"\text{{GD (optimal }}\alpha^{{*}}\text{{): }} {n_gd}"
            r"\ \text{steps to } \|\nabla J\| < 10^{-6}",
            size=30, color=style.YELLOW,
        ).to_edge(DOWN, buff=0.5)
        self.play(Create(traj), Write(counter), run_time=3.2)
        self.wait(0.8)
        self.gd_traj, self.gd_counter = traj, counter

    def beat_one_leap(self):
        leap = Arrow(pt(X0), pt([0, 0]), buff=0, color=style.GREEN,
                     stroke_width=6, max_tip_length_to_length_ratio=0.08)
        counter2 = style.eq(
            r"\text{Newton: } 1 \text{ step.}", size=34, color=style.GREEN,
        ).next_to(self.gd_counter, UP, buff=0.25)
        self.play(Create(leap), Write(counter2), run_time=1.4)
        self.wait(1.2)
        self.leap, self.newton_counter = leap, counter2

    def beat_algebra(self):
        eq1 = style.eq(r"\theta^{+} = \theta - \alpha\,\nabla J", size=38)
        eq2 = style.eq(r"\theta^{+} = \theta - H^{-1}\nabla J", size=38)
        eq3 = style.eq(
            r"\text{quadratic: } H^{-1}\nabla J = A^{-1}(A\theta) = \theta"
            r"\ \Rightarrow\ \theta^{+} = 0", size=38, color=style.GREEN)
        for e in (eq1, eq2, eq3):
            e.to_corner(UP + RIGHT, buff=0.5)
            e.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)
        self.play(Write(eq1))
        self.wait(0.8)
        self.play(ReplacementTransform(eq1, eq2))
        self.wait(0.8)
        self.play(ReplacementTransform(eq2, eq3))
        self.wait(1.2)
        self.eq = eq3

    def beat_whiten(self):
        W = nt.whitening_map(A)
        caption = style.label(
            "Newton = gradient descent in the metric of the Hessian: κ → 1",
            size=28, color=style.TEAL,
        ).to_edge(DOWN, buff=0.5)
        world = VGroup(self.curves, self.gd_traj, self.leap)
        self.play(
            ApplyMatrix(W / np.max(np.abs(W)) * 1.6, world),
            ReplacementTransform(self.gd_counter, caption),
            FadeOut(self.newton_counter),
            run_time=2.2,
        )
        self.wait(1.4)
        self.play(FadeOut(world), FadeOut(caption), FadeOut(self.eq),
                  FadeOut(self.title))

    def beat_rosenbrock(self):
        header = style.title_card(
            "The honest slide: Rosenbrock's banana",
            "no quadratic to solve exactly — Newton needs damping; BFGS learns H on the fly",
        ).to_edge(UP)
        self.play(FadeIn(header))
        self.header = header

        axes = Axes(
            x_range=[-2, 2, 1], y_range=[-1, 3, 1],
            x_length=7.5, y_length=4.6, tips=False,
            axis_config={"stroke_color": style.GRAY, "stroke_width": 1.5},
        ).shift(DOWN * 0.7 + LEFT * 1.6)
        f, grad, hess = nt.rosenbrock()

        # smooth log-loss glow: bright where f is small (the banana valley)
        from manim import ImageMobject  # noqa: PLC0415
        xs = np.linspace(-2, 2, 360)
        ys = np.linspace(-1, 3, 270)
        Xg, Yg = np.meshgrid(xs, ys)
        Fg = (1 - Xg) ** 2 + 100 * (Yg - Xg**2) ** 2
        V = 1.0 / (1.0 + np.log1p(Fg)) ** 2
        img = np.zeros((*V.shape, 3), dtype=np.uint8)
        img[..., 0] = (V * 60).astype(np.uint8)
        img[..., 1] = (V * 150).astype(np.uint8)
        img[..., 2] = (V * 190).astype(np.uint8)
        glow = ImageMobject(img[::-1])
        glow.stretch_to_fit_width(7.5).stretch_to_fit_height(4.6)
        glow.move_to(axes.coords_to_point(0, 1))
        floor = VMobject(stroke_color=style.GRAY, stroke_width=1.6)
        floor.set_points_smoothly(
            [axes.coords_to_point(x, x * x) for x in np.linspace(-1.7, 1.7, 60)])
        floor.set_stroke(opacity=0.5)
        rings = VGroup(floor)
        self.play(Create(axes), FadeIn(glow), Create(rings), run_time=1.6)
        self.glow = glow

        x0 = np.array([-1.2, 1.0])
        path_damped = nt.newton_run(grad, hess, x0, damped=True, f=f)
        path_bfgs, _ = nt.bfgs_run(f, grad, x0, steps=400, tol=1e-6)

        def to_axes(path, color, width=3):
            line = VMobject(stroke_color=color, stroke_width=width)
            line.set_points_as_corners(
                [axes.coords_to_point(p[0], p[1]) for p in path])
            return line

        start = Dot(axes.coords_to_point(*x0), color=style.WHITE, radius=0.06)
        goal = Dot(axes.coords_to_point(1, 1), color=style.GREEN, radius=0.06)
        newton_line = to_axes(path_damped, style.GREEN)
        bfgs_line = to_axes(path_bfgs, style.YELLOW, 2.4)

        chips = VGroup(
            style.eq(rf"\text{{damped Newton: }} {len(path_damped) - 1}"
                     r"\text{ steps}", size=28, color=style.GREEN),
            style.eq(rf"\text{{BFGS: }} {len(path_bfgs) - 1}\text{{ steps}}",
                     size=28, color=style.YELLOW),
            style.eq(r"\text{GD } (\alpha = 10^{-3})\text{: }>20{,}000",
                     size=28, color=style.RED),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        chips.shift(RIGHT * 4.4 + UP * 0.4)

        self.play(FadeIn(start), FadeIn(goal))
        self.play(Create(newton_line), Write(chips[0]), run_time=1.8)
        self.play(Create(bfgs_line), Write(chips[1]), run_time=1.8)
        self.play(Write(chips[2]))
        secant = style.eq(
            r"\text{BFGS: least-change update with } B_{k+1}s_k = y_k"
            r"\quad\text{(the secant condition)}",
            size=26, color=style.GRAY,
        ).to_edge(DOWN, buff=0.4)
        self.play(Write(secant))
        self.wait(1.8)
        self.play(*map(FadeOut, [axes, rings, self.glow, newton_line,
                                 bfgs_line, chips, secant, start, goal,
                                 header]))

    def beat_ledger(self):
        rows = VGroup(
            style.eq(r"\nabla J:\ O(d)\ \text{per step}\quad\times\quad"
                     r" O(\kappa)\ \text{steps}", size=34, color=style.YELLOW),
            style.eq(r"H^{-1}\nabla J:\ O(d^{3})\ \text{per step}\quad\times"
                     r"\quad O(1)\ \text{steps}", size=34, color=style.GREEN),
            style.eq(r"\text{BFGS}:\ O(d^{2})\ \text{per step}\quad\times\quad"
                     r"\text{few}", size=34, color=style.TEAL),
        ).arrange(DOWN, buff=0.5)
        self.play(Write(rows[0]))
        self.play(Write(rows[1]))
        self.play(Write(rows[2]))
        coda = style.label(
            "deep learning uses neither pure GD nor H — but that is another video",
            size=26,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(coda))
        self.wait(2.4)
