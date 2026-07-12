"""Demo 1 — GD Convergence, Learning Rates & Hessian Curvature.

Scenes:
    ValleyOpener  — 3D loss valley; same bowl, two fates (safe vs divergent alpha).
    EigenSplit    — the full story: decoupling GD in the Hessian eigenbasis,
                    per-mode contraction factors, and the 2/beta_max cliff.

Render:
    manim render -qh scenes/gd_convergence.py ValleyOpener EigenSplit
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from manim import (  # noqa: E402
    DEGREES,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Arrow,
    Axes,
    Create,
    DashedLine,
    Dot,
    Dot3D,
    FadeIn,
    FadeOut,
    Flash,
    Indicate,
    Line,
    NumberPlane,
    Rectangle,
    ReplacementTransform,
    Scene,
    Surface,
    Text,
    ThreeDAxes,
    ThreeDScene,
    TracedPath,
    TransformMatchingTex,
    ValueTracker,
    VGroup,
    VMobject,
    Write,
    always_redraw,
    rate_functions,
)

from core import style  # noqa: E402
from core.sims import gradient_descent as gd  # noqa: E402

style.apply_house_style()

# One quadratic to rule the whole demo: lambda = (0.5, 4), rotated 30 degrees.
LAMBDAS = np.array([0.5, 4.0])
ANGLE = np.deg2rad(30)
A = gd.make_quadratic(LAMBDAS, ANGLE)
# Start with real weight on BOTH eigenvectors so the steep-mode bounce is visible:
# theta0 = Q @ (2.6, 1.0) — eigen-coordinates (c1, c2) = (2.6, 1.0).
THETA0 = gd.rotation(ANGLE) @ np.array([2.6, 1.0])
ALPHA_SAFE = 0.42
ALPHA_DIV = 0.62
ALPHA_CRIT = gd.convergence_threshold(A)  # = 0.5
CONTOUR_LEVELS = [0.35, 1.0, 2.2, 4.0]


def path_polyline(path: np.ndarray, scale: float = 1.0, color=style.BLUE) -> VMobject:
    pts = [np.array([p[0] * scale, p[1] * scale, 0.0]) for p in path]
    line = VMobject(stroke_color=color, stroke_width=3)
    line.set_points_as_corners(pts)
    return line


def contour_group(matrix: np.ndarray, scale: float = 1.0) -> VGroup:
    group = VGroup()
    for i, level in enumerate(CONTOUR_LEVELS):
        pts = gd.contour_ellipse(matrix, level)
        curve = VMobject(
            stroke_color=style.BLUE,
            stroke_width=2,
            stroke_opacity=0.85 - 0.15 * i,
        )
        curve.set_points_smoothly(
            [np.array([p[0] * scale, p[1] * scale, 0.0]) for p in pts]
        )
        group.add(curve)
    return group


def clip_path(path: np.ndarray, radius: float) -> np.ndarray:
    """Truncate a (possibly divergent) path once it leaves |theta| <= radius."""
    norms = np.linalg.norm(path, axis=1)
    outside = np.nonzero(norms > radius)[0]
    if outside.size == 0:
        return path
    return path[: outside[0] + 1]


class ValleyOpener(ThreeDScene):
    """Same bowl, same start, two learning rates — glide vs slingshot."""

    Z_SCALE = 1.0 / 3.0  # visual squash of J values

    def lift(self, axes: ThreeDAxes, theta: np.ndarray) -> np.ndarray:
        j = float(gd.loss(A, theta)[0])
        return axes.c2p(theta[0], theta[1], j * self.Z_SCALE + 0.04)

    def construct(self):
        axes = ThreeDAxes(
            x_range=[-3.2, 3.2, 1],
            y_range=[-3.2, 3.2, 1],
            z_range=[0, 4.5, 1],
            x_length=9,
            y_length=9,
            z_length=4,
        )
        surface = Surface(
            lambda u, v: axes.c2p(
                u, v, float(gd.loss(A, np.array([u, v]))[0]) * self.Z_SCALE
            ),
            u_range=[-3.0, 3.0],
            v_range=[-3.0, 3.0],
            resolution=(40, 40),
            fill_opacity=0.25,
            stroke_width=0.6,
            stroke_opacity=0.45,
            checkerboard_colors=None,
            fill_color="#1d3a4a",
            stroke_color=style.BLUE,
        )
        floor_contours = contour_group(A).set_stroke(opacity=0.5)

        title = style.title_card(
            "Why does gradient descent explode?",
            "J(θ) = ½ θᵀAθ   —   same bowl, same start",
        ).to_edge(UP)
        for part in title:
            part.add_background_rectangle(color=style.BACKGROUND, opacity=0.72)
        self.add_fixed_in_frame_mobjects(title)

        self.set_camera_orientation(phi=62 * DEGREES, theta=-50 * DEGREES, zoom=0.85)
        self.play(Create(axes), FadeIn(surface), FadeIn(floor_contours), run_time=2)
        self.begin_ambient_camera_rotation(rate=0.04)

        # Run 1: safe alpha — the ball glides home.
        alpha_tag = Text(
            f"α = {ALPHA_SAFE}  (safe)", font_size=30, color=style.GREEN,
            font=style.SERIF,
        ).to_corner(DOWN + LEFT, buff=0.6)
        self.add_fixed_in_frame_mobjects(alpha_tag)

        safe = gd.gd_path(A, THETA0, ALPHA_SAFE, steps=22)
        ball = Dot3D(self.lift(axes, safe[0]), radius=0.09, color=style.YELLOW)
        trail = TracedPath(
            ball.get_center, stroke_color=style.YELLOW, stroke_width=4
        )
        self.add(trail, ball)
        for theta in safe[1:]:
            self.play(
                ball.animate.move_to(self.lift(axes, theta)),
                run_time=0.16,
                rate_func=rate_functions.ease_in_out_sine,
            )
        self.wait(0.8)

        # Run 2: bump alpha past the cliff — the ball ricochets out.
        alpha_tag2 = Text(
            f"α = {ALPHA_DIV}  (diverges!)", font_size=30, color=style.RED,
            font=style.SERIF,
        ).to_corner(DOWN + LEFT, buff=0.6)
        self.play(FadeOut(ball), FadeOut(trail))
        self.add_fixed_in_frame_mobjects(alpha_tag2)
        self.remove(alpha_tag)

        div = clip_path(gd.gd_path(A, THETA0, ALPHA_DIV, steps=12), radius=3.1)
        ball2 = Dot3D(self.lift(axes, div[0]), radius=0.09, color=style.RED)
        trail2 = TracedPath(
            ball2.get_center, stroke_color=style.RED, stroke_width=4
        )
        self.add(trail2, ball2)
        for theta in div[1:]:
            self.play(
                ball2.animate.move_to(self.lift(axes, theta)),
                run_time=0.28,
                rate_func=rate_functions.ease_in_out_sine,
            )
        self.wait(0.4)

        question = Text(
            "The answer is curvature — and it's directional.",
            font_size=32,
            color=style.WHITE,
            font=style.SERIF,
        ).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(question)
        self.remove(question)
        self.play(FadeOut(alpha_tag2), FadeIn(question))
        self.wait(1.6)


class EigenSplit(Scene):
    """Decouple GD in the eigenbasis; watch each mode contract by (1 - alpha*lambda_i)."""

    DIAGRAM_SCALE = 0.85

    # ------------------------------------------------------------------ beats
    def construct(self):
        self.beat_equations()
        self.beat_scalar_warmup()
        self.beat_rotate_to_eigenbasis()
        self.beat_mode_split()
        self.beat_alpha_sweep()
        self.beat_conclusion()

    # Beat 1: the update rule is a fixed linear map.
    def beat_equations(self):
        eq1 = style.eq(
            r"\theta_{t+1}", "=", r"\theta_t", "-", r"\alpha", r"\nabla J(\theta_t)"
        )
        eq2 = style.eq(r"\theta_{t+1}", "=", r"(I - \alpha A)", r"\theta_t")
        note = style.label("one fixed matrix, applied over and over", size=26)
        note.next_to(eq2, DOWN, buff=0.4)

        self.play(Write(eq1))
        self.wait(0.8)
        self.play(TransformMatchingTex(eq1, eq2), run_time=1.2)
        self.play(FadeIn(note))
        self.wait(1.0)
        self.eq_header = eq2
        self.play(FadeOut(note), eq2.animate.scale(0.75).to_edge(UP, buff=0.35))
        eq2.add_background_rectangle(color=style.BACKGROUND, opacity=0.85)

    # Beat 2: the scalar case already contains the whole story.
    def beat_scalar_warmup(self):
        panels = VGroup(
            self.scalar_panel(0.35, "monotone decay", style.GREEN),
            self.scalar_panel(1.65, "oscillating decay", style.YELLOW),
            self.scalar_panel(2.15, "divergence", style.RED),
        ).arrange(RIGHT, buff=0.7).next_to(self.eq_header, DOWN, buff=0.6)

        rule = style.eq(
            r"c_{t+1} = (1-\alpha\lambda)\,c_t", size=34
        ).next_to(panels, DOWN, buff=0.45)
        verdict = style.eq(
            r"|1-\alpha\lambda| < 1 \iff \alpha < 2/\lambda", size=34,
            color=style.YELLOW,
        ).next_to(rule, DOWN, buff=0.25)

        self.play(FadeIn(panels, lag_ratio=0.2), run_time=1.6)
        self.play(Write(rule))
        self.wait(0.6)
        self.play(Write(verdict))
        self.wait(1.4)
        self.play(FadeOut(panels), FadeOut(rule), FadeOut(verdict))

    def scalar_panel(self, alpha: float, caption: str, color) -> VGroup:
        lam = 1.0
        steps = 9
        c = (1 - alpha * lam) ** np.arange(steps + 1)
        axes = Axes(
            x_range=[0, steps, 3],
            y_range=[-2.2, 2.2, 1],
            x_length=3.2,
            y_length=2.2,
            tips=False,
            axis_config={"stroke_color": style.GRAY, "stroke_width": 1.5},
        )
        dots = VGroup(
            *[Dot(axes.c2p(t, np.clip(v, -2.1, 2.1)), radius=0.045, color=color)
              for t, v in enumerate(c)]
        )
        stems = VGroup(
            *[Line(axes.c2p(t, 0), d.get_center(), stroke_color=color,
                   stroke_width=2, stroke_opacity=0.6)
              for t, d in enumerate(dots)]
        )
        tag = style.label(f"αλ = {alpha}", size=22, color=color)
        cap = style.label(caption, size=20)
        VGroup(tag, cap).arrange(DOWN, buff=0.1).next_to(axes, DOWN, buff=0.2)
        return VGroup(axes, stems, dots, tag, cap)

    # Beat 3: rotated contours -> rotate the world into the eigenbasis.
    def beat_rotate_to_eigenbasis(self):
        s = self.DIAGRAM_SCALE
        plane = NumberPlane(
            x_range=[-8, 8, 1],
            y_range=[-5, 5, 1],
            background_line_style={
                "stroke_color": "#242424", "stroke_width": 1,
            },
        )
        contours = contour_group(A, scale=s)
        path = gd.gd_path(A, THETA0, ALPHA_SAFE, steps=22)
        traj = path_polyline(path, scale=s, color=style.YELLOW)
        start_dot = Dot(traj.get_start(), color=style.YELLOW, radius=0.06)

        coupled = style.label(
            "standard coordinates: modes are coupled, the path curves", size=26,
            color=style.WHITE,
        ).to_edge(DOWN, buff=0.4)

        self.plane = plane
        self.play(FadeIn(plane), Create(contours), run_time=1.4)
        self.play(FadeIn(start_dot), Create(traj), run_time=2.2)
        self.play(FadeIn(coupled))
        self.wait(1.2)

        # A = Q Lambda Q^T -> rotate everything by Q^T
        decomp = style.eq(r"A = Q\Lambda Q^{\top}", size=36, color=style.TEAL)
        decomp.next_to(self.eq_header, DOWN, buff=0.3)
        decomp.add_background_rectangle(color=style.BACKGROUND, opacity=0.85)
        self.play(Write(decomp))
        self.wait(0.6)

        world = VGroup(contours, traj, start_dot)
        uncoupled = style.label(
            "eigen-coordinates: the ellipse aligns — every axis is now independent",
            size=26, color=style.WHITE,
        ).to_edge(DOWN, buff=0.4)
        self.play(
            Rotate := world.animate.rotate(-ANGLE, about_point=ORIGIN),
            ReplacementTransform(coupled, uncoupled),
            run_time=2.0,
        )
        self.wait(1.2)
        self.world = world
        self.decomp = decomp
        self.footer = uncoupled

    # Beat 4: split screen — the 2D walk is two 1D problems.
    def beat_mode_split(self):
        self.play(
            self.world.animate.scale(0.8, about_point=ORIGIN).shift(LEFT * 3.4),
            FadeOut(self.footer),
            run_time=1.2,
        )

        lambdas, _ = gd.eigen_frame(A)
        path_e = gd.to_eigen(A, gd.gd_path(A, THETA0, ALPHA_SAFE, steps=22))

        # Two 1D number lines, one per eigenmode.
        panels = VGroup()
        mode_dots = []
        for i in (0, 1):
            color = style.MODE_COLORS[i]
            line = Line(LEFT * 2.2, RIGHT * 2.2, stroke_color=style.GRAY)
            zero = Line(UP * 0.12, DOWN * 0.12, stroke_color=style.WHITE)
            factor = 1 - ALPHA_SAFE * lambdas[i]
            tag = style.eq(
                rf"c_{{{i+1}}}(t) = ({factor:+.2f})^t\, c_{{{i+1}}}(0)",
                size=28, color=color,
            )
            panel = VGroup(line, zero, tag.next_to(line, UP, buff=0.25))
            panels.add(panel)
            mode_dots.append(Dot(radius=0.075, color=color))
        panels.arrange(DOWN, buff=1.1).shift(RIGHT * 3.6)

        header = style.label("each mode bounces alone", size=26, color=style.WHITE)
        header.next_to(panels, UP, buff=0.5)

        self.play(FadeIn(panels), FadeIn(header))

        # Sync the 2D dot with the two 1D dots over the iterates.
        t_tracker = ValueTracker(0.0)
        unit = 2.2 / 3.0  # eigencoord -> panel offset scaling

        def dot_updater_2d(mob):
            idx = min(int(t_tracker.get_value()), len(path_e) - 1)
            p = path_e[idx] @ gd.eigen_frame(A)[1].T  # back to standard coords
            # world was rotated by -ANGLE, so display in rotated frame:
            q = gd.rotation(-ANGLE) @ p
            mob.move_to(
                np.array([q[0], q[1], 0]) * self.DIAGRAM_SCALE * 0.8 + LEFT * 3.4
            )

        walker = Dot(color=style.WHITE, radius=0.07)
        walker.add_updater(dot_updater_2d)

        for i in (0, 1):
            def mode_updater(mob, i=i):
                idx = min(int(t_tracker.get_value()), len(path_e) - 1)
                line = panels[i][0]
                mob.move_to(line.get_center() + RIGHT * path_e[idx, i] * unit)
            mode_dots[i].add_updater(mode_updater)

        self.add(walker, *mode_dots)
        self.play(
            t_tracker.animate.set_value(len(path_e) - 1),
            run_time=6.0,
            rate_func=rate_functions.linear,
        )
        self.wait(0.8)

        walker.clear_updaters()
        for d in mode_dots:
            d.clear_updaters()
        self.play(FadeOut(VGroup(walker, *mode_dots, panels, header)))

    # Beat 5: sweep alpha; the steep mode hits -1 first and the path slingshots.
    def beat_alpha_sweep(self):
        lambdas, _ = gd.eigen_frame(A)
        alpha = ValueTracker(ALPHA_SAFE)

        # Factor gauges on the right: f_i = 1 - alpha*lambda_i, danger at -1.
        gauges = VGroup()
        for i in (0, 1):
            color = style.MODE_COLORS[i]
            rail = Line(LEFT * 1.9, RIGHT * 1.9, stroke_color=style.GRAY)
            lo = style.eq("-1", size=22, color=style.RED).next_to(
                rail.get_start(), DOWN, buff=0.15
            )
            hi = style.eq("+1", size=22, color=style.GRAY).next_to(
                rail.get_end(), DOWN, buff=0.15
            )
            tag = style.eq(
                rf"1-\alpha\lambda_{{{i+1}}}", size=26, color=color
            ).next_to(rail, UP, buff=0.18)
            gauges.add(VGroup(rail, lo, hi, tag))
        gauges.arrange(DOWN, buff=1.0).shift(RIGHT * 3.7 + DOWN * 0.4)

        def gauge_dot(i):
            def build():
                f = 1 - alpha.get_value() * lambdas[i]
                rail = gauges[i][0]
                x = np.interp(f, [-1.4, 1.0], [-1.9, 1.9])
                dot = Dot(rail.get_center() + RIGHT * x, radius=0.08,
                          color=style.MODE_COLORS[i])
                if f <= -1.0:
                    dot.set_color(style.RED)
                return dot
            return always_redraw(build)

        dots = [gauge_dot(0), gauge_dot(1)]

        alpha_read = always_redraw(
            lambda: style.eq(
                rf"\alpha = {alpha.get_value():.2f}", size=34,
                color=(style.RED if alpha.get_value() > ALPHA_CRIT else style.WHITE),
            ).next_to(gauges, UP, buff=0.6)
        )

        # Live trajectory in the (rotated) eigen frame on the left.
        def build_traj():
            p = gd.to_eigen(A, gd.gd_path(A, THETA0, alpha.get_value(), steps=26))
            p = clip_path(p, radius=6.0)
            line = path_polyline(p, scale=self.DIAGRAM_SCALE * 0.8,
                                 color=style.YELLOW)
            line.shift(LEFT * 3.4)
            if gd.worst_rate(A, alpha.get_value()) >= 1.0:
                line.set_stroke(style.RED)
            return line

        traj = always_redraw(build_traj)

        crit_line = DashedLine(UP * 0.4, DOWN * 0.4, stroke_color=style.RED)
        crit_label = style.eq(r"\alpha = 2/\beta_{\max}", size=24, color=style.RED)

        self.play(FadeIn(gauges), FadeIn(alpha_read), *[FadeIn(d) for d in dots])
        self.add(traj)
        self.play(alpha.animate.set_value(0.48), run_time=3.0)
        self.wait(0.5)

        # place the critical marker on the steep gauge (mode 2)
        rail2 = gauges[1][0]
        x_crit = np.interp(1 - ALPHA_CRIT * lambdas[1], [-1.4, 1.0], [-1.9, 1.9])
        crit_line.move_to(rail2.get_center() + RIGHT * x_crit)
        crit_label.next_to(crit_line, DOWN, buff=0.2)
        self.play(Create(crit_line), FadeIn(crit_label))
        self.wait(0.4)

        self.play(alpha.animate.set_value(0.56), run_time=2.5)
        self.play(Flash(dots[1], color=style.RED, flash_radius=0.5))
        self.wait(1.2)
        self.play(alpha.animate.set_value(0.44), run_time=1.5)
        self.wait(0.8)

        traj.clear_updaters()
        for d in dots:
            d.clear_updaters()
        alpha_read.clear_updaters()
        self.play(
            FadeOut(VGroup(gauges, crit_line, crit_label, traj,
                           alpha_read, *dots, self.world, self.decomp,
                           self.plane))
        )

    # Beat 6: the theorem, and where beta_max comes from in real life.
    def beat_conclusion(self):
        thm = style.eq(
            r"\text{GD converges}", r"\iff", r"\alpha < \frac{2}{\beta_{\max}}",
            size=48,
        )
        sub = style.eq(
            r"\beta_{\max} = \lambda_{\max}\!\left(\nabla^2 J\right)",
            size=34, color=style.TEAL,
        ).next_to(thm, DOWN, buff=0.5)
        kappa = style.eq(
            r"\text{and even below the cliff, the slow mode sets your speed: }"
            r"\ \text{rate} = \frac{\kappa - 1}{\kappa + 1}",
            size=30, color=style.GRAY,
        ).next_to(sub, DOWN, buff=0.6)
        tie_in = style.eq(
            r"\text{linear regression: } H = \tfrac{1}{n} X^{\top} X"
            r"\ \Rightarrow\ \text{the data sets your learning-rate budget}",
            size=30, color=style.YELLOW,
        ).to_edge(DOWN, buff=0.7)

        self.play(ReplacementTransform(self.eq_header, thm), run_time=1.2)
        self.play(Write(sub))
        self.play(FadeIn(kappa))
        self.wait(1.0)
        self.play(Write(tie_in))
        self.wait(2.5)
