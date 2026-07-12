"""Demo 2 — Batched Neural-Network Gradients: backprop as matrix flow.

One dense layer Z = XW + b, A = sigma(Z). The batch is a cloud of points; the
upstream gradient is a cloud of error arrows; backward-through-W is forward-
through-W^T; dW is a pile of rank-1 outer-product votes.

Render:
    manim render -qh scenes/backprop_flow.py BackpropFlow
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
    Arrow,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Indicate,
    MathTex,
    NumberPlane,
    Rectangle,
    ReplacementTransform,
    Scene,
    Square,
    Text,
    TransformMatchingTex,
    VGroup,
    Write,
)

from core import style  # noqa: E402
from core.sims import backprop as bp  # noqa: E402

style.apply_house_style()

# ------------------------------------------------------------------ data
RNG = np.random.default_rng(7)
ANGLES = np.linspace(0, 2 * np.pi, 8, endpoint=False)
X = np.stack([2.1 * np.cos(ANGLES), 2.1 * np.sin(ANGLES)], axis=1)
X[2] = [3.4, 1.3]                       # one point that will saturate sigma
W = np.array([[1.3, 0.6], [-0.5, 1.0]])
B = np.array([0.4, -0.3])
Y = RNG.uniform(0.25, 0.75, size=(8, 2))

BW = bp.backward(X, W, B, Y)
POINT_COLORS = [style.BLUE, style.TEAL, style.YELLOW, style.GREEN,
                style.PURPLE, style.BLUE, style.TEAL, style.GREEN]
S = 0.72                                 # world -> scene scale


def P(v) -> np.ndarray:
    return np.array([v[0] * S, v[1] * S, 0.0])


def heat_square(value: float, vmax: float, side: float = 0.42) -> Square:
    """Signed-value cell: yellow positive, blue negative, black zero."""
    t = float(np.clip(abs(value) / vmax, 0, 1))
    color = style.YELLOW if value >= 0 else style.BLUE
    return Square(side).set_stroke(style.GRAY, 1).set_fill(color, opacity=0.15 + 0.8 * t)


def heat_matrix(M: np.ndarray, vmax: float, side: float = 0.42) -> VGroup:
    rows = VGroup()
    for i in range(M.shape[0]):
        row = VGroup(*[heat_square(M[i, j], vmax, side) for j in range(M.shape[1])])
        row.arrange(RIGHT, buff=0.06)
        rows.add(row)
    rows.arrange(DOWN, buff=0.06)
    return rows


class BackpropFlow(Scene):
    def construct(self):
        self.beat_forward()
        self.beat_shapes()
        self.beat_error_arrows()
        self.beat_backward_flow()
        self.beat_outer_products()
        self.beat_check()

    # Forward: the batch is a cloud; W is one map applied to all of it.
    def beat_forward(self):
        title = style.title_card(
            "Backpropagation, batched",
            "a batch is a cloud of points — a layer is one map on all of them",
        ).to_edge(UP)
        self.play(FadeIn(title))
        self.wait(0.6)
        self.play(title.animate.scale(0.6).to_corner(UP + LEFT, buff=0.3))
        self.title = title

        plane = NumberPlane(
            x_range=[-6, 6, 1], y_range=[-4, 4, 1],
            background_line_style={"stroke_color": "#242424", "stroke_width": 1},
        )
        self.play(FadeIn(plane))
        self.plane = plane

        dots = VGroup(*[
            Dot(P(x), radius=0.075, color=c) for x, c in zip(X, POINT_COLORS)
        ])
        x_label = style.eq(r"X \in \mathbb{R}^{n \times d}", size=34, color=style.WHITE)
        x_label.to_corner(UP + RIGHT, buff=0.5)
        x_label.add_background_rectangle(color=style.BACKGROUND, opacity=0.85)
        self.play(FadeIn(dots), Write(x_label))
        self.ghost_inputs = dots.copy().set_opacity(0.28)
        self.wait(0.6)

        # Z = XW + b : linear map sweeps the whole batch at once
        z_eq = style.eq("Z", "=", "X W + b", size=34, color=style.TEAL)
        z_eq.next_to(x_label, DOWN, buff=0.25).align_to(x_label, RIGHT)
        z_eq.add_background_rectangle(color=style.BACKGROUND, opacity=0.85)
        self.add(self.ghost_inputs)
        self.play(
            ApplyMatrix(W.T @ np.eye(2), dots),  # manim right-multiplies columns
            Write(z_eq),
            run_time=2.0,
        )
        shift = P(B)
        self.play(dots.animate.shift(shift), run_time=0.7)
        self.wait(0.5)

        # sigma squashes everything into the unit square
        a_eq = style.eq("A", "=", r"\sigma(Z)", size=34, color=style.GREEN)
        a_eq.next_to(z_eq, DOWN, buff=0.25).align_to(z_eq, RIGHT)
        a_eq.add_background_rectangle(color=style.BACKGROUND, opacity=0.85)
        unit_box = Rectangle(width=S, height=S).move_to(P([0.5, 0.5]))
        unit_box.set_stroke(style.GREEN, 2)

        Zs = BW["Z"]
        As = BW["A"]
        self.play(Create(unit_box), Write(a_eq))
        self.play(
            *[d.animate.move_to(P(a)) for d, a in zip(dots, As)],
            run_time=1.8,
        )
        sat_note = style.label("the far-out point saturates — remember it", size=22,
                               color=style.RED).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(sat_note), Indicate(dots[2], color=style.RED))
        self.wait(0.8)
        self.play(FadeOut(sat_note))

        # zoom into the unit square where the outputs live
        world = VGroup(plane, self.ghost_inputs, dots, unit_box)
        self.play(world.animate.scale(3.2, about_point=P([0.5, 0.5])).shift(
            -P([0.5, 0.5]) * 0  # keep centered on unit box
        ), run_time=1.6)
        self.dots = dots
        self.unit_box = unit_box
        self.eq_stack = VGroup(x_label, z_eq, a_eq)

    # The shape ribbon: dimensions force every formula.
    def beat_shapes(self):
        ribbon = style.eq(
            r"X\,(n\times d)", r"\;\to\;", r"Z\,(n\times k)",
            r"\;\to\;", r"A\,(n\times k)", r"\;\to\;", r"L \in \mathbb{R}",
            size=30, color=style.GRAY,
        ).to_edge(DOWN, buff=0.35)
        ribbon.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)
        self.play(Write(ribbon))
        self.wait(1.0)
        self.ribbon = ribbon

    # Upstream gradient: one red error arrow per data point.
    def beat_error_arrows(self):
        g_eq = style.eq(
            r"G = \frac{\partial L}{\partial A}", r"\in \mathbb{R}^{n \times k}",
            size=34, color=style.RED,
        )
        g_eq.to_corner(UP + RIGHT, buff=0.5)
        g_eq.add_background_rectangle(color=style.BACKGROUND, opacity=0.85)
        self.play(ReplacementTransform(self.eq_stack, g_eq))

        # arrows drawn at the zoomed positions: dots moved, so anchor to dots
        SCALE = 7.0     # error magnitudes are ~1/n; scale for visibility
        arrows = VGroup()
        for d, g in zip(self.dots, BW["G"]):
            vec = np.array([g[0], g[1], 0.0]) * S * SCALE * 3.2
            arrows.add(Arrow(d.get_center(), d.get_center() - vec,
                             buff=0, stroke_width=3.5, color=style.RED,
                             max_tip_length_to_length_ratio=0.25))
        note = style.label("how L wants each output to move", size=24, color=style.RED)
        note.next_to(g_eq, DOWN, buff=0.3).align_to(g_eq, RIGHT)
        self.play(FadeIn(arrows, lag_ratio=0.1), FadeIn(note), run_time=1.5)
        self.wait(0.8)

        # Hadamard gate: multiply by sigma'(Z); the saturated point's arrow dies
        gate = style.eq(
            r"\frac{\partial L}{\partial Z}", "=",
            r"G \odot \sigma'(Z)", size=34, color=style.YELLOW,
        )
        gate.move_to(g_eq).align_to(g_eq, RIGHT)
        gate.add_background_rectangle(color=style.BACKGROUND, opacity=0.85)
        gate_note = style.label("saturated unit ⇒ σ′ ≈ 0 ⇒ dead gradient", size=24,
                                color=style.YELLOW)
        gate_note.next_to(gate, DOWN, buff=0.3).align_to(gate, RIGHT)

        sp = bp.sigmoid_prime(BW["Z"])
        new_arrows = VGroup()
        for d, g, s in zip(self.dots, BW["G"], sp):
            vec = np.array([g[0] * s[0], g[1] * s[1], 0.0]) * S * SCALE * 3.2 * 4
            new_arrows.add(Arrow(d.get_center(), d.get_center() - vec,
                                 buff=0, stroke_width=3.5, color=style.YELLOW,
                                 max_tip_length_to_length_ratio=0.25))
        self.play(
            ReplacementTransform(g_eq, gate),
            ReplacementTransform(note, gate_note),
            ReplacementTransform(arrows, new_arrows),
            run_time=1.6,
        )
        self.play(Indicate(self.dots[2], color=style.RED, scale_factor=1.6))
        self.wait(1.0)
        self.arrows = new_arrows
        self.gate_group = VGroup(gate, gate_note)

    # THE beat: error flows backward through W^T onto the inputs.
    def beat_backward_flow(self):
        back_eq = style.eq(
            r"\frac{\partial L}{\partial X}", "=",
            r"\frac{\partial L}{\partial Z}", r"W^{\top}",
            size=36, color=style.WHITE,
        )
        back_eq.to_corner(UP + RIGHT, buff=0.5)
        back_eq.add_background_rectangle(color=style.BACKGROUND, opacity=0.85)
        dims = style.eq(
            r"(n\times k)\,(k\times d) \;\to\; (n\times d)",
            size=26, color=style.GRAY,
        ).next_to(back_eq, DOWN, buff=0.25).align_to(back_eq, RIGHT)
        dims.add_background_rectangle(color=style.BACKGROUND, opacity=0.85)

        # gradient arrows land on the (ghosted) input cloud, mapped through W^T
        dX = BW["dX"]
        SCALE = 7.0
        input_arrows = VGroup()
        for ghost, g in zip(self.ghost_inputs, dX):
            start = ghost.get_center()
            vec = np.array([g[0], g[1], 0.0]) * S * SCALE * 3.2 * 4
            input_arrows.add(Arrow(start, start - vec, buff=0,
                                   stroke_width=3.5, color=style.TEAL,
                                   max_tip_length_to_length_ratio=0.25))

        self.play(
            ReplacementTransform(self.gate_group, VGroup(back_eq, dims)),
            run_time=1.0,
        )
        self.play(
            ReplacementTransform(self.arrows, input_arrows),
            self.ghost_inputs.animate.set_opacity(0.9),
            run_time=2.2,
        )
        caption = style.label(
            "backward through W  =  forward through Wᵀ", size=26, color=style.TEAL,
        ).to_edge(DOWN, buff=0.35)
        caption.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)
        self.play(ReplacementTransform(self.ribbon, caption))
        self.wait(1.4)
        self.play(*map(FadeOut, [input_arrows, caption, back_eq, dims,
                                 self.plane, self.ghost_inputs, self.dots,
                                 self.unit_box]))

    # dW as a sum of rank-1 outer-product votes.
    def beat_outer_products(self):
        self.play(FadeOut(self.title))
        votes = bp.outer_product_decomp(X, BW["dZ"])
        vmax = float(np.abs(votes).max())
        eq = style.eq(
            r"\frac{\partial L}{\partial W}", "=", r"X^{\top}\frac{\partial L}{\partial Z}",
            "=", r"\sum_{i=1}^{n} x^{(i)\top}\,\delta^{(i)}",
            size=38,
        ).to_edge(UP, buff=0.6)
        self.play(Write(eq))

        cells = VGroup()
        for i in (0, 1, 2):
            block = VGroup(
                heat_matrix(votes[i], vmax),
                style.eq(rf"x^{{({i+1})\top}}\delta^{{({i+1})}}", size=24,
                         color=POINT_COLORS[i]),
            ).arrange(DOWN, buff=0.25)
            cells.add(block)
            if i < 2:
                cells.add(style.eq("+", size=36, color=style.GRAY))
        cells.add(style.eq(r"+\;\cdots\;=", size=36, color=style.GRAY))
        total = VGroup(
            heat_matrix(BW["dW"], float(np.abs(BW["dW"]).max())),
            style.eq(r"\partial L/\partial W", size=24, color=style.YELLOW),
        ).arrange(DOWN, buff=0.25)
        cells.add(total)
        cells.arrange(RIGHT, buff=0.45).next_to(eq, DOWN, buff=0.8)

        note = style.label(
            "every data point stamps its own rank-1 vote on the weight update",
            size=26,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(cells, lag_ratio=0.15), run_time=2.2)
        self.play(FadeIn(note))
        self.wait(1.6)
        self.play(FadeOut(cells), FadeOut(note), FadeOut(eq))

    # The sanity ritual: analytic gradient vs finite differences.
    def beat_check(self):
        fd = bp.fd_grad("W", X, W, B, Y)
        analytic = BW["dW"][0, 0]
        numeric = fd[0, 0]
        lines = VGroup(
            style.eq(r"\text{analytic:}\quad \partial L/\partial W_{11} = "
                     + f"{analytic:.8f}", size=34, color=style.TEAL),
            style.eq(r"\text{finite differences:}\quad "
                     + f"{numeric:.8f}", size=34, color=style.YELLOW),
            style.eq(r"\text{agreement} < 10^{-6}\ \checkmark", size=34,
                     color=style.GREEN),
        ).arrange(DOWN, buff=0.45)
        self.play(Write(lines[0]))
        self.play(Write(lines[1]))
        self.play(Write(lines[2]))
        self.wait(1.0)
        end = style.label(
            "stack this layer a few times and you have the MNIST net of PS2.5",
            size=26,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(end))
        self.wait(2.0)
