"""Demo 6 — PageRank: the random surfer, leaks, traps, and teleportation.

Rank is liquid: one multiply by the column-stochastic M pours every node's
rank down its out-links. Dead ends drain the system, spider traps hoard it,
teleportation r' = beta*M r + (1-beta)/n fixes it — and power iteration
converges to the principal eigenvector at rate ~beta.

Render:
    manim render -qh scenes/pagerank_surfer.py PageRankSurfer
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
    Arrow,
    Circle,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    MoveAlongPath,
    Line,
    Rectangle,
    ReplacementTransform,
    Scene,
    Text,
    TransformMatchingTex,
    VGroup,
    Write,
)

from core import style  # noqa: E402
from core.sims import pagerank as pr  # noqa: E402

style.apply_house_style()

POS = {
    0: (-2.2, 0.6), 1: (0.0, 1.6), 2: (0.4, -0.6),
    3: (2.6, 0.9), 4: (2.0, -1.5), 5: (-0.8, -1.9),
    6: (4.6, -0.4), 7: (5.9, 0.6),           # trap nodes
}
RNG = np.random.default_rng(3)


def p3(i):
    return np.array([POS[i][0], POS[i][1], 0.0])


class PageRankSurfer(Scene):
    def construct(self):
        self.build_core()
        self.beat_surfer()
        self.beat_flow_steps()
        self.beat_trap()
        self.beat_teleport()
        self.beat_convergence()

    # ---------------------------------------------------------------- helpers
    def make_ring(self, i, r, color=style.BLUE):
        ring = Circle(radius=self.radius(r), stroke_width=3, stroke_color=color)
        ring.set_fill(color, opacity=0.35)
        ring.move_to(p3(i))
        return ring

    def make_tag(self, i, r):
        return style.eq(f"{r:.2f}", size=22, color=style.WHITE).move_to(p3(i))

    @staticmethod
    def radius(r):
        return float(0.22 + 1.5 * r)

    def redraw_nodes(self, r, color=style.BLUE):
        anims = []
        for i in self.rings:
            new_tag = self.make_tag(i, r[i])
            anims.append(self.rings[i].animate.become(
                self.make_ring(i, r[i], color)))
            anims.append(ReplacementTransform(self.tags[i], new_tag))
            self.tags[i] = new_tag
        return anims

    def mass_meter(self, mass):
        bar = Rectangle(width=2.4 * mass, height=0.22)
        bar.set_fill(style.GREEN if mass > 0.999 else style.RED, 0.9)
        bar.set_stroke(width=0)
        bar.move_to([-5.2 + 1.2 * mass, 3.0, 0])
        frame = Rectangle(width=2.4, height=0.26).set_stroke(style.GRAY, 1.5)
        frame.move_to([-4.0, 3.0, 0])
        tag = style.eq(rf"\Sigma r = {mass:.3f}", size=26,
                       color=style.GREEN if mass > 0.999 else style.RED)
        tag.next_to(frame, DOWN, buff=0.15)
        return VGroup(bar, frame, tag)

    def edge_arrow(self, a, b, color=style.GRAY):
        va, vb = p3(a), p3(b)
        d = (vb - va) / np.linalg.norm(vb - va)
        return Arrow(va + d * 0.42, vb - d * 0.42, buff=0,
                     stroke_width=2.5, color=color,
                     max_tip_length_to_length_ratio=0.10)

    # ---------------------------------------------------------------- beats
    def build_core(self):
        title = style.title_card(
            "PageRank — importance as a fixed point",
            "a page matters if important pages link to it; break the circle with a random surfer",
        ).to_edge(UP)
        self.play(FadeIn(title))
        self.title = title

        self.n = 6
        self.M = pr.link_matrix(pr.DEMO_EDGES, 6)
        self.r = np.full(6, 1 / 6)
        self.edges = VGroup(*[self.edge_arrow(a, b) for a, b in pr.DEMO_EDGES])
        self.rings = {i: self.make_ring(i, self.r[i]) for i in range(6)}
        self.tags = {i: self.make_tag(i, self.r[i]) for i in range(6)}
        self.meter = self.mass_meter(1.0)
        self.play(Create(self.edges),
                  *[FadeIn(g) for g in self.rings.values()],
                  *[FadeIn(g) for g in self.tags.values()],
                  FadeIn(self.meter), run_time=1.6)

    def beat_surfer(self):
        surfer = Dot(p3(0), radius=0.11, color=style.YELLOW)
        self.play(FadeIn(surfer))
        adj = {}
        for a, b in pr.DEMO_EDGES:
            adj.setdefault(a, []).append(b)
        here = 0
        for _ in range(5):
            nxt = adj[here][RNG.integers(len(adj[here]))]
            seg = Line(p3(here), p3(nxt))
            self.play(MoveAlongPath(surfer, seg), run_time=0.5)
            here = nxt
        note = style.label(
            "importance = fraction of time the surfer spends on you",
            size=26).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(note))
        self.wait(0.8)
        self.play(FadeOut(surfer), FadeOut(note), FadeOut(self.title))

    def flow_pulses(self, edges):
        anims = []
        for a, b in edges:
            dot = Dot(p3(a), radius=0.06, color=style.TEAL)
            self.add(dot)
            anims.append(MoveAlongPath(dot, Line(p3(a), p3(b)), run_time=0.7))
        return anims

    def beat_flow_steps(self):
        eq = style.eq("r'", "=", "M r", size=40, color=style.TEAL)
        eq.to_corner(UP + RIGHT, buff=0.6)
        eq.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)
        colsum = style.eq(r"\text{every column of } M \text{ sums to } 1"
                          r"\ \Rightarrow\ \Sigma r \text{ conserved}",
                          size=26, color=style.GRAY)
        colsum.to_edge(DOWN, buff=0.5)
        self.play(Write(eq), FadeIn(colsum))
        for _ in range(3):
            pulses = self.flow_pulses(pr.DEMO_EDGES)
            self.play(*pulses)
            self.remove(*[a.mobject for a in pulses])
            self.r = pr.step(self.M, self.r)
            meter_new = self.mass_meter(pr.total_mass(self.r))
            self.play(*self.redraw_nodes(self.r),
                      ReplacementTransform(self.meter, meter_new),
                      run_time=0.8)
            self.meter = meter_new
        self.wait(0.6)
        self.play(FadeOut(colsum))
        self.eq = eq

    def beat_trap(self):
        # add the 2-cycle trap
        trap_edges = [(2, 6), (6, 7), (7, 6)]
        edges, n = pr.with_trap()
        self.M = pr.link_matrix(edges, n)
        self.r = np.append(self.r * 6 / 6, [0.0, 0.0])
        self.r = self.r / self.r.sum()
        new_arrows = VGroup(*[self.edge_arrow(a, b, style.RED)
                              for a, b in trap_edges])
        for i in (6, 7):
            self.rings[i] = self.make_ring(i, self.r[i], style.RED)
            self.tags[i] = self.make_tag(i, self.r[i])
        self.trap_arrows = new_arrows
        warn = style.label("a spider trap joins the web", size=26,
                           color=style.RED).to_edge(DOWN, buff=0.5)
        self.play(Create(new_arrows), FadeIn(self.rings[6]), FadeIn(self.tags[6]),
                  FadeIn(self.rings[7]), FadeIn(self.tags[7]), FadeIn(warn))

        for _ in range(6):
            self.r = pr.step(self.M, self.r)
            self.play(*self.redraw_nodes(self.r), run_time=0.55)
        hoard = style.label(
            f"mass conserved (Σr = {pr.total_mass(self.r):.3f}) — but the trap is eating the web",
            size=26, color=style.RED).to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(warn, hoard))
        self.wait(1.0)
        self.trap_note = hoard

    def beat_teleport(self):
        eq2 = style.eq("r'", "=", r"\beta M r", "+",
                       r"(1-\beta)\tfrac{\mathbf{1}}{n}", size=40,
                       color=style.YELLOW)
        eq2.to_corner(UP + RIGHT, buff=0.6)
        eq2.add_background_rectangle(color=style.BACKGROUND, opacity=0.9)
        dashed = VGroup(*[
            DashedLine(p3(6), p3(i), stroke_color=style.YELLOW,
                       stroke_width=1.6, stroke_opacity=0.7)
            for i in (0, 1, 3, 5)
        ])
        fix = style.label(
            "with probability 1−β the surfer teleports anywhere — no trap can hold it",
            size=26, color=style.YELLOW).to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(self.eq, eq2), Create(dashed),
                  ReplacementTransform(self.trap_note, fix))
        self.eq = eq2

        for _ in range(7):
            self.r = pr.google_step(self.M, self.r, beta=0.85)
            self.play(*self.redraw_nodes(self.r), run_time=0.45)
        meter_new = self.mass_meter(pr.total_mass(self.r))
        self.play(ReplacementTransform(self.meter, meter_new))
        self.meter = meter_new
        self.wait(0.8)
        self.play(FadeOut(dashed), FadeOut(fix))

    def beat_convergence(self):
        # bar chart of r vs the true eigenvector
        r_star = pr.eig_rank(self.M, beta=0.85)
        r_now, hist, iters = pr.power_iterate(self.M, beta=0.85, tol=1e-10)
        bars = VGroup()
        W = 0.55
        for i in range(len(r_now)):
            b = Rectangle(width=W, height=max(4.5 * r_now[i], 0.02))
            b.set_fill(style.TEAL, 0.85).set_stroke(width=0)
            b.move_to([-3.0 + i * (W + 0.28), -2.6 + 4.5 * r_now[i] / 2, 0])
            tick = Line([-W / 2, 0, 0], [W / 2, 0, 0], stroke_color=style.YELLOW,
                        stroke_width=4)
            tick.move_to([-3.0 + i * (W + 0.28), -2.6 + 4.5 * r_star[i], 0])
            lab = style.eq(str(i), size=20, color=style.GRAY)
            lab.move_to([-3.0 + i * (W + 0.28), -2.85, 0])
            bars.add(VGroup(b, tick, lab))
        caption = style.eq(
            rf"\text{{power iteration: converged in }} {iters}"
            r"\text{ steps; yellow ticks = the eigenvector itself}",
            size=28, color=style.WHITE,
        ).to_edge(DOWN, buff=0.35)
        self.play(*[FadeOut(g) for g in self.rings.values()],
                  *[FadeOut(g) for g in self.tags.values()],
                  FadeOut(self.edges), FadeOut(self.trap_arrows),
                  FadeOut(self.meter))
        self.play(FadeIn(bars, lag_ratio=0.08), Write(caption))
        rate = style.eq(
            r"\|r_{t+1}-r^{*}\| \approx \beta\,\|r_t - r^{*}\|"
            r" \quad\text{— geometric, rate } \beta",
            size=30, color=style.GRAY,
        ).move_to([0, 2.0, 0])
        self.play(Write(rate))
        self.wait(2.4)
