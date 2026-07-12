"""Demo 4 — GNN Message Passing & the Depth Barrier.

Two graphs whose red nodes are indistinguishable for k = 0..3 layers of
sum-aggregation message passing — and split at exactly k = 4, because that is
when the structural difference finally enters the receptive field.

Render:
    manim render -qh scenes/gnn_depth.py GnnDepth
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
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Flash,
    Line,
    MoveAlongPath,
    ReplacementTransform,
    Scene,
    Text,
    VGroup,
    Write,
)

from core import style  # noqa: E402
from core.sims import gnn  # noqa: E402

style.apply_house_style()

CYCLE, BRANCHED, RED = gnn.demo_graphs()

POS_CYCLE = {
    "r": (-5.7, -1.9), "a": (-4.5, -1.0),
    "b1": (-3.3, -0.3), "b2": (-4.7, 0.4), "c": (-3.5, 1.1),
}
POS_BRANCH = {
    "r": (0.9, -1.9), "a": (2.1, -1.0),
    "b1": (3.3, -0.3), "c1": (4.3, 0.4), "d": (5.3, 1.1),
    "b2": (1.6, 0.3), "c2": (1.1, 1.4),
}
EDGES_CYCLE = [("r", "a"), ("a", "b1"), ("a", "b2"), ("b1", "c"), ("b2", "c")]
EDGES_BRANCH = [("r", "a"), ("a", "b1"), ("a", "b2"), ("b1", "c1"),
                ("c1", "d"), ("b2", "c2")]

HIST_C = gnn.run_layers(CYCLE, 4)
HIST_B = gnn.run_layers(BRANCHED, 4)


def p3(xy) -> np.ndarray:
    return np.array([xy[0], xy[1], 0.0])


class GnnDepth(Scene):
    def construct(self):
        self.build_graphs()
        self.run_message_passing()
        self.explain_divergence()
        self.bfs_coda()

    def make_graph(self, pos, edges, caption, cx):
        lines = VGroup(*[
            Line(p3(pos[u]), p3(pos[v]), stroke_color=style.GRAY,
                 stroke_width=2.5) for u, v in edges
        ])
        nodes, labels = {}, {}
        for v, xy in pos.items():
            ring = Circle(radius=0.34, stroke_width=3).move_to(p3(xy))
            ring.set_stroke(style.RED if v == "r" else style.BLUE)
            ring.set_fill(style.BACKGROUND, opacity=1.0)
            lab = style.eq("1", size=28).move_to(ring)
            nodes[v], labels[v] = ring, lab
        cap = style.label(caption, size=26, color=style.WHITE)
        cap.move_to([cx, -2.9, 0])
        return lines, nodes, labels, cap

    def build_graphs(self):
        title = style.title_card(
            "How deep must a GNN look?",
            "sum aggregation:  h(k+1) = h(k) + Σ neighbors h(k)   —   all nodes start at 1",
        ).to_edge(UP)
        self.play(FadeIn(title))
        self.title = title

        self.lc, self.nc, self.tc, cap_c = self.make_graph(
            POS_CYCLE, EDGES_CYCLE, "cycle graph", -4.5)
        self.lb, self.nb, self.tb, cap_b = self.make_graph(
            POS_BRANCH, EDGES_BRANCH, "branched path graph", 3.1)

        self.play(
            Create(self.lc), Create(self.lb),
            *[FadeIn(m) for m in list(self.nc.values()) + list(self.nb.values())],
            *[FadeIn(m) for m in list(self.tc.values()) + list(self.tb.values())],
            FadeIn(cap_c), FadeIn(cap_b),
            run_time=1.8,
        )

        self.k_tag = style.eq("k = 0", size=40, color=style.YELLOW)
        self.k_tag.move_to([0, 1.9, 0])
        self.banner = style.eq(
            r"h_r = 1 \;=\; 1\ \checkmark", size=34, color=style.GREEN,
        ).move_to([0, -3.55, 0])
        self.play(Write(self.k_tag), Write(self.banner))
        self.wait(0.6)

    def pulse(self, pos, edges):
        anims = []
        for u, v in edges:
            for a, b in ((u, v), (v, u)):
                seg = Line(p3(pos[a]), p3(pos[b]))
                dot = Dot(p3(pos[a]), radius=0.055, color=style.YELLOW)
                self.add(dot)
                anims.append(MoveAlongPath(dot, seg, run_time=0.75))
        return anims

    def run_message_passing(self):
        for k in range(1, 5):
            pulses = (self.pulse(POS_CYCLE, EDGES_CYCLE)
                      + self.pulse(POS_BRANCH, EDGES_BRANCH))
            self.play(*pulses)
            self.remove(*[a.mobject for a in pulses])

            new_labels = {}
            anims = []
            for hist, labels, nodes in ((HIST_C, self.tc, self.nc),
                                        (HIST_B, self.tb, self.nb)):
                vmax = max(h[k] for h in hist.values())
                for v, lab in labels.items():
                    val = hist[v][k]
                    nl = style.eq(str(val), size=24 if val < 100 else 20)
                    nl.move_to(nodes[v])
                    anims.append(ReplacementTransform(lab, nl))
                    new_labels[(id(labels), v)] = nl
                    heat = 0.15 + 0.55 * val / vmax
                    anims.append(nodes[v].animate.set_fill(
                        "#3d3410", opacity=heat))
            for v in self.tc:
                self.tc[v] = new_labels[(id(self.tc), v)]
            for v in self.tb:
                self.tb[v] = new_labels[(id(self.tb), v)]

            hc, hb = HIST_C[RED][k], HIST_B[RED][k]
            k_new = style.eq(f"k = {k}", size=40, color=style.YELLOW)
            k_new.move_to(self.k_tag)
            if hc == hb:
                banner_new = style.eq(
                    rf"h_r = {hc} \;=\; {hb}\ \checkmark", size=34,
                    color=style.GREEN,
                ).move_to(self.banner)
            else:
                banner_new = style.eq(
                    rf"h_r = {hc} \;\neq\; {hb}", size=38, color=style.RED,
                ).move_to(self.banner)
            anims += [ReplacementTransform(self.k_tag, k_new),
                      ReplacementTransform(self.banner, banner_new)]
            self.k_tag, self.banner = k_new, banner_new
            self.play(*anims, run_time=1.0)
            self.wait(0.7 if hc == hb else 0.2)

        self.play(
            Flash(self.nc["r"], color=style.RED, flash_radius=0.55),
            Flash(self.nb["r"], color=style.RED, flash_radius=0.55),
        )
        verdict = style.label(
            "4 layers of message passing are needed to distinguish the two red nodes",
            size=26, color=style.RED,
        ).move_to([0, -3.55, 0])
        self.play(ReplacementTransform(self.banner, verdict))
        self.banner = verdict
        self.wait(1.4)

    def explain_divergence(self):
        # highlight the structures that differ, 3+ hops from red
        close_edge = Line(p3(POS_CYCLE["b2"]), p3(POS_CYCLE["c"]),
                          stroke_color=style.YELLOW, stroke_width=6)
        leaves = VGroup(
            Circle(radius=0.42, stroke_color=style.YELLOW, stroke_width=4)
            .move_to(p3(POS_BRANCH["c2"])),
            Circle(radius=0.42, stroke_color=style.YELLOW, stroke_width=4)
            .move_to(p3(POS_BRANCH["d"])),
        )
        note = style.label(
            "the difference lives 3–4 hops from red — invisible to any shallower net",
            size=26, color=style.YELLOW,
        ).move_to([0, -3.55, 0])
        self.play(Create(close_edge), Create(leaves),
                  ReplacementTransform(self.banner, note))
        self.banner = note
        self.wait(1.6)
        self.play(FadeOut(close_edge), FadeOut(leaves))

    def bfs_coda(self):
        # receptive-field rings around red: k layers see exactly k hops
        dist = gnn.hop_distances(BRANCHED, "r")
        rings = VGroup()
        ring_labels = VGroup()
        for k in range(1, 5):
            members = [v for v, d in dist.items() if d == k]
            circles = VGroup(*[
                Circle(radius=0.44, stroke_color=style.TEAL, stroke_width=3,
                       stroke_opacity=0.9).move_to(p3(POS_BRANCH[v]))
                for v in members
            ])
            tag = style.eq(f"{k}", size=26, color=style.TEAL)
            tag.next_to(circles[0], UP, buff=0.08)
            rings.add(circles)
            ring_labels.add(tag)

        note = style.label(
            "k layers see exactly k hops. No more.  (sum aggregation ≈ the Weisfeiler–Leman test)",
            size=26, color=style.TEAL,
        ).move_to([0, -3.55, 0])
        self.play(ReplacementTransform(self.banner, note))
        for k in range(4):
            self.play(Create(rings[k]), FadeIn(ring_labels[k]), run_time=0.7)
        self.wait(2.2)
