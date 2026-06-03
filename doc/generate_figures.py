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
from pga3d.pga3d_display import (  # noqa: E402
    plot_points,
    plot_lines,
    plot_plane,
    plot_line_object,
    plot_plane_object,
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


# ---------------------------------------------------------------------------
# Meet (^ outer product) and Join (& regressive product) figures
# ---------------------------------------------------------------------------

def _algebra_point_to_point(p_alg):
    z = p_alg.value[11]
    y = p_alg.value[12]
    x = p_alg.value[13]
    w = p_alg.value[14]
    if abs(w) < 1e-12:
        return None
    return Point(x / w, y / w, z / w)


def fig_join_two_points():
    p1 = Point(0, 0, 0)
    p2 = Point(4, 3, 2)
    line = p1 & p2  # join of two points = line
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_points([p1, p2], ax=ax, color="C0")
    plot_line_object(line, ax=ax, color="C3", length=6.0,
                     label="Join: p1 & p2", center=(2, 1.5, 1))
    ax.legend()
    ax.set_title("Join of two points -> line")
    _save(fig, "fig_join_two_points.png")


def fig_join_three_points():
    p1 = Point(3, 0, 0)
    p2 = Point(0, 3, 0)
    p3 = Point(0, 0, 3)
    plane = p1 & p2 & p3  # join of three points = plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_points([p1, p2, p3], ax=ax, color="C0")
    plot_lines([(p1, p2), (p2, p3), (p3, p1)], ax=ax, color="C0")
    plot_plane_object(plane, ax=ax, color="C3", alpha=0.3,
                      center=(1, 1, 1), size=4.0,
                      label="Join: p1 & p2 & p3")
    ax.legend()
    ax.set_title("Join of three points -> plane")
    _save(fig, "fig_join_three_points.png")


def fig_join_line_and_point():
    a = Point(0, 0, 0)
    b = Point(4, 0, 0)
    line = a & b
    p = Point(2, 4, 3)
    plane = line & p  # join of line and point = plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_points([a, b, p], ax=ax, color="C0")
    plot_line_object(line, ax=ax, color="C2", length=4.0,
                     label="input line", center=(2, 0, 0))
    plot_plane_object(plane, ax=ax, color="C3", alpha=0.3,
                      center=(2, 1.5, 1), size=4.0,
                      label="Join: line & p")
    ax.legend()
    ax.set_title("Join of line and point -> plane")
    _save(fig, "fig_join_line_point.png")


def fig_meet_two_planes():
    pi1 = Plane.from_abcd(1, 0, 0, 0)   # x = 0
    pi2 = Plane.from_abcd(0, 1, 0, 0)   # y = 0
    line = pi1 ^ pi2  # meet of two planes = line
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_plane_object(pi1, ax=ax, color="C0", alpha=0.2,
                      center=(0, 0, 0), size=4.0, label="plane 1 (x=0)")
    plot_plane_object(pi2, ax=ax, color="C1", alpha=0.2,
                      center=(0, 0, 0), size=4.0, label="plane 2 (y=0)")
    plot_line_object(line, ax=ax, color="C3", length=5.0,
                     label="Meet: pi1 ^ pi2", center=(0, 0, 0))
    ax.legend()
    ax.set_title("Meet of two planes -> line")
    _save(fig, "fig_meet_two_planes.png")


def fig_meet_three_planes():
    pi1 = Plane.from_abcd(1, 0, 0, -2)  # x = 2
    pi2 = Plane.from_abcd(0, 1, 0, -3)  # y = 3
    pi3 = Plane.from_abcd(0, 0, 1, -1)  # z = 1
    p_alg = pi1 ^ pi2 ^ pi3  # meet of three planes = point
    pt = _algebra_point_to_point(p_alg)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_plane_object(pi1, ax=ax, color="C0", alpha=0.2,
                      center=(2, 3, 1), size=3.0, label="plane 1 (x=2)")
    plot_plane_object(pi2, ax=ax, color="C1", alpha=0.2,
                      center=(2, 3, 1), size=3.0, label="plane 2 (y=3)")
    plot_plane_object(pi3, ax=ax, color="C2", alpha=0.2,
                      center=(2, 3, 1), size=3.0, label="plane 3 (z=1)")
    if pt is not None:
        plot_points([pt], ax=ax, color="C3",
                    label_prefix="Meet: pi1^pi2^pi3 = ")
    ax.legend()
    ax.set_title("Meet of three planes -> point")
    _save(fig, "fig_meet_three_planes.png")


def fig_meet_plane_and_line():
    pi = Plane.from_abcd(0, 0, 1, -2)  # z = 2
    a = Point(-1, -1, 0)
    b = Point(3, 4, 5)
    line = a & b
    p_alg = pi ^ line  # meet of plane and line = point
    pt = _algebra_point_to_point(p_alg)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_plane_object(pi, ax=ax, color="C0", alpha=0.2,
                      center=(1, 1, 2), size=4.0, label="plane (z=2)")
    plot_line_object(line, ax=ax, color="C2", length=5.0,
                     label="input line", center=(1, 1, 2))
    if pt is not None:
        plot_points([pt], ax=ax, color="C3",
                    label_prefix="Meet: pi ^ line = ")
    ax.legend()
    ax.set_title("Meet of plane and line -> point")
    _save(fig, "fig_meet_plane_line.png")


def main():
    fig_point()
    fig_line()
    fig_plane()
    fig_transformations()
    fig_projection()
    fig_scene()
    fig_join_two_points()
    fig_join_three_points()
    fig_join_line_and_point()
    fig_meet_two_planes()
    fig_meet_three_planes()
    fig_meet_plane_and_line()


if __name__ == "__main__":
    main()
