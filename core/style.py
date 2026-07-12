"""Shared 3b1b-flavored look for every scene: palette, background, text presets.

This is the only module in core/ that imports manim; core/sims/ stays pure numpy.
"""

from __future__ import annotations

from manim import (
    DOWN,
    LEFT,
    UP,
    Mobject,
    MathTex,
    Text,
    VGroup,
    config,
)

# Math-textbook canvas: pure black, serif type (Chris's call after Demo 1)
BACKGROUND = "#000000"
SERIF = "Times New Roman"

# Palette (manim's own 3b1b-derived hexes, named for their role here)
BLUE = "#58C4DD"      # primary objects, trajectories
YELLOW = "#FFD35A"    # highlights, the "aha" object
GREEN = "#83C167"     # success / convergence
RED = "#FC6255"       # error / divergence / gradients
TEAL = "#5CD0B3"      # secondary accents
PURPLE = "#9A72AC"    # tertiary accents
GRAY = "#888888"      # de-emphasized structure
WHITE = "#ECECF1"

MODE_COLORS = [TEAL, YELLOW, PURPLE, GREEN]  # per-eigenmode coloring


def apply_house_style() -> None:
    """Call at module import time in every scene file."""
    config.background_color = BACKGROUND


def title_card(text: str, sub: str | None = None) -> VGroup:
    """Big top-of-scene title with optional gray subtitle."""
    title = Text(text, font_size=44, color=WHITE, weight="BOLD", font=SERIF)
    group = VGroup(title)
    if sub is not None:
        subtitle = Text(sub, font_size=26, color=GRAY, font=SERIF)
        subtitle.next_to(title, DOWN, buff=0.25)
        group.add(subtitle)
    return group


def eq(*parts: str, size: float = 40, color: str = WHITE) -> MathTex:
    """MathTex with house defaults; pass substrings to enable TransformMatchingTex."""
    return MathTex(*parts, font_size=size, color=color)


def label(text: str, size: float = 24, color: str = GRAY) -> Text:
    return Text(text, font_size=size, color=color, font=SERIF)


def pin_top_left(mobject: Mobject, buff: float = 0.5) -> Mobject:
    return mobject.to_corner(UP + LEFT, buff=buff)
