"""
Visual smoke-test / showcase for `ganja.graph`.

Runs a series of small scenes covering every supported object kind in
PGA(2,0,1) and PGA(3,0,1):

    * 3-D points, lines, and planes
    * a rotor sandwich product (rotation of a point)
    * a translator sandwich product (translation of a line)
    * a meet / join construction (line ∩ plane → point)
    * 2-D points and lines
    * polylines drawn directly from numpy arrays

Run as a script:

    python ganja/visualization_test.py            # opens windows
    python ganja/visualization_test.py --save     # writes PNGs to doc/

The `--save` flag uses the matplotlib Agg backend and is suitable for
headless / CI environments.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

# Allow `python ganja/visualization_test.py` from the repo root.
if __package__ in (None, ""):
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from ganja import Algebra, graph
else:
    from . import Algebra, graph


# ---------------------------------------------------------------------------
# Helpers to build PGA elements from Euclidean data.
# ---------------------------------------------------------------------------

def pga3_point(PGA3, x, y, z):
    """Euclidean point (x, y, z) as a PGA(3,0,1) trivector."""
    e1, e2, e3, e0 = PGA3.basis_vectors()
    return ((e1 ^ e2 ^ e3)
            + x * (e0 ^ e2 ^ e3)
            - y * (e0 ^ e1 ^ e3)
            + z * (e0 ^ e1 ^ e2))


def pga3_plane(PGA3, a, b, c, d):
    """Plane a·x + b·y + c·z + d = 0 as a PGA(3,0,1) vector."""
    e1, e2, e3, e0 = PGA3.basis_vectors()
    return a * e1 + b * e2 + c * e3 + d * e0


def pga2_point(PGA2, x, y):
    e1, e2, e0 = PGA2.basis_vectors()
    return ((e1 ^ e2)
            + x * (e0 ^ e2)
            - y * (e0 ^ e1))


def pga2_line(PGA2, a, b, c):
    """Line a·x + b·y + c = 0 as a PGA(2,0,1) vector."""
    e1, e2, e0 = PGA2.basis_vectors()
    return a * e1 + b * e2 + c * e0


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

def scene_pga3_basics():
    """Three points, an axis line, and an oblique plane."""
    PGA3 = Algebra(3, 0, 1)
    e1, e2, e3, e0 = PGA3.basis_vectors()

    A = pga3_point(PGA3, 1, 0, 0)
    B = pga3_point(PGA3, 0, 1, 0)
    C = pga3_point(PGA3, 0, 0, 1)
    x_axis = e2 ^ e3                          # line along the x-axis
    plane = pga3_plane(PGA3, 1, 1, 1, -1)     # x + y + z = 1

    return graph(
        ["tab:red",  A, "A(1,0,0)",
         "tab:red",  B, "B(0,1,0)",
         "tab:red",  C, "C(0,0,1)",
         "tab:blue", x_axis, "x-axis",
         "tab:green", plane, "x+y+z=1"],
        title="PGA(3,0,1) — points, line, plane", lim=2.0,
    )


def scene_pga3_rotor():
    """Rotate a point 90° about the z-axis with a rotor sandwich."""
    PGA3 = Algebra(3, 0, 1)
    e1, e2, e3, e0 = PGA3.basis_vectors()

    P = pga3_point(PGA3, 1.5, 0.0, 0.0)
    # Rotor about the origin's z-axis (bivector e1∧e2).
    theta = math.pi / 4                       # half-angle
    R = (math.cos(theta) * PGA3.scalar(1.0)
         + math.sin(theta) * (e1 ^ e2))
    P_rot = R @ P                             # sandwich  R P ~R  (rotates by 2θ)

    arc = np.array([[math.cos(t), math.sin(t), 0.0]
                    for t in np.linspace(0, math.pi / 2, 40)]) * 1.5

    return graph(
        ["tab:red",   P,     "P",
         "tab:green", P_rot, "R·P·~R",
         "tab:gray",  arc],
        title="PGA(3,0,1) — rotor: 90° rotation about z", lim=2.0,
    )


def scene_pga3_translator():
    """Translate a line with a translator sandwich."""
    PGA3 = Algebra(3, 0, 1)
    e1, e2, e3, e0 = PGA3.basis_vectors()

    line = e2 ^ e3                            # x-axis line
    # Translator by (0, 1, 0):  T = 1 - (1/2) d (e0 ∧ e_dir)
    T = PGA3.scalar(1.0) - 0.5 * (e0 ^ e2)
    line_t = T @ line                         # translated line

    return graph(
        ["tab:blue",   line,    "x-axis",
         "tab:orange", line_t,  "translated by (0,1,0)"],
        title="PGA(3,0,1) — translator on a line", lim=2.0,
    )


def scene_pga3_meet():
    """Intersect a line with a plane to obtain a point (line ∧ plane)."""
    PGA3 = Algebra(3, 0, 1)
    e1, e2, e3, e0 = PGA3.basis_vectors()

    line = e2 ^ e3                            # x-axis
    plane = pga3_plane(PGA3, 1, 0, 0, -1)     # x = 1
    meet = line ^ plane                       # intersection point (trivector)

    return graph(
        ["tab:blue",   line,  "x-axis",
         "tab:green",  plane, "x=1",
         "tab:red",    meet,  "line ∧ plane"],
        title="PGA(3,0,1) — meet of a line and a plane", lim=2.0,
    )


def scene_pga2_basics():
    """Two points and the line through them, plus an independent line."""
    PGA2 = Algebra(2, 0, 1)

    A = pga2_point(PGA2, -1.0, -0.5)
    B = pga2_point(PGA2, +1.0, +1.0)
    join = A ^ B                              # line through A and B
    other = pga2_line(PGA2, 1, 1, -1)         # x + y = 1

    return graph(
        ["tab:red",    A, "A",
         "tab:red",    B, "B",
         "tab:blue",   join,  "A ∨ B",
         "tab:orange", other, "x+y=1"],
        title="PGA(2,0,1) — two points, joining line, extra line", lim=2.0,
    )


def scene_pga2_polyline():
    """A 2-D polyline plus a couple of points."""
    PGA2 = Algebra(2, 0, 1)
    ts = np.linspace(0, 2 * math.pi, 100)
    spiral = np.stack([0.6 * ts / (2 * math.pi) * np.cos(ts),
                       0.6 * ts / (2 * math.pi) * np.sin(ts)], axis=1)

    return graph(
        ["tab:purple", spiral,
         "tab:red", pga2_point(PGA2, 0, 0), "origin",
         "tab:red", pga2_point(PGA2, 0.6, 0), "end"],
        title="PGA(2,0,1) — polyline + points", lim=1.0,
    )


SCENES = [
    ("ganja_test_pga3_basics.png",     scene_pga3_basics),
    ("ganja_test_pga3_rotor.png",      scene_pga3_rotor),
    ("ganja_test_pga3_translator.png", scene_pga3_translator),
    ("ganja_test_pga3_meet.png",       scene_pga3_meet),
    ("ganja_test_pga2_basics.png",     scene_pga2_basics),
    ("ganja_test_pga2_polyline.png",   scene_pga2_polyline),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true",
                        help="save PNGs to doc/ instead of opening windows")
    parser.add_argument("--outdir", default=None,
                        help="output directory for --save (default: doc/)")
    args = parser.parse_args()

    if args.save:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = args.outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "doc")
    if args.save:
        os.makedirs(out_dir, exist_ok=True)

    for name, scene in SCENES:
        ax = scene()
        if args.save:
            path = os.path.join(out_dir, name)
            ax.figure.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(ax.figure)
            print(f"wrote {path}")

    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
