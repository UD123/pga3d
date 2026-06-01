"""Generate the figures embedded in the README using pga3d_display helpers.

Run from the repository root:

    python doc/generate_figures.py
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for figure generation
import matplotlib.pyplot as plt

# Make the repo root importable when running from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pga3d import Point, Line, Plane, Translator, Rotor  # noqa: E402
from pga3d_display import (  # noqa: E402
    plot_points,
    plot_lines,
    plot_plane,
)

DOC_DIR = os.path.dirname(os.path.abspath(__file__))


def _save(fig, name):
    path = os.path.join(DOC_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_point():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_points([Point(1, 2, 3)], ax=ax)
    ax.legend()
    ax.set_title("Single Point")
    _save(fig, "fig_point.png")


def fig_line():
    p1, p2 = Point(0, 0, 0), Point(1, 1, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_points([p1, p2], ax=ax)
    plot_lines([(p1, p2)], ax=ax)
    ax.legend()
    ax.set_title("Line from Two Points")
    _save(fig, "fig_line.png")


def fig_plane():
    p1, p2, p3 = Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_points([p1, p2, p3], ax=ax)
    plot_plane([p1, p2, p3], ax=ax)
    ax.legend()
    ax.set_title("Plane from Three Points")
    _save(fig, "fig_plane.png")


def fig_transformations():
    p = Point(1, 0, 0)
    t = Translator.from_xyz(1, 0, 0)
    r = Rotor.from_angle_and_line(math.pi / 2, Line.from_xyz(0, 0, 1))
    m = t * r
    p_t = t.project(p)
    p_tr = m.project(p)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_points([p, p_t, p_tr], ax=ax, color="C1")
    ax.legend()
    ax.set_title("Point, Translated, and Translated+Rotated")
    _save(fig, "fig_transformations.png")


def fig_projection():
    p1 = Point(2, 3, 4)
    p2 = Point(20, 3, 7)
    p3 = Point(9, 12, 17)
    line = Line.from_points(p3, p1)
    plane = Plane.from_points(p1, p2, p3)
    p_proj = p1.project_onto(line)
    plane.project_onto(Point(0, 0, 0))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_points([p1, p_proj], ax=ax, color="C2")
    plot_lines([(p3, p1)], ax=ax)
    plot_plane([p1, p2, p3], ax=ax)
    ax.legend()
    ax.set_title("Projection of Point onto Line")
    _save(fig, "fig_projection.png")


def fig_scene():
    p0 = Point(0, 0, 0)
    p1 = Point(2, 3, 4)
    p2 = Point(20, 3, 7)
    p3 = Point(9, 12, 17)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_points([p0, p1, p2, p3], ax=ax)
    plot_lines([(p1, p2), (p2, p3), (p3, p1)], ax=ax)
    plot_plane([p1, p2, p3], ax=ax)
    ax.legend()
    ax.set_title("PGA3D Example Objects")
    _save(fig, "fig_scene.png")


def main():
    fig_point()
    fig_line()
    fig_plane()
    fig_transformations()
    fig_projection()
    fig_scene()


if __name__ == "__main__":
    main()
