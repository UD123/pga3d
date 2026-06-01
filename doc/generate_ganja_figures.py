"""Generate the `ganja` package demo figures used in the README.

Run from the repository root:

    python doc/generate_ganja_figures.py
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ganja import Algebra, graph  # noqa: E402

DOC_DIR = os.path.dirname(os.path.abspath(__file__))


def _save(ax, name):
    path = os.path.join(DOC_DIR, name)
    ax.figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"wrote {path}")


def fig_pga3():
    PGA3 = Algebra(3, 0, 1)
    e1, e2, e3, e0 = PGA3.basis_vectors()

    # Point at (1, 2, 3) as a PGA trivector
    P = ((e1 ^ e2 ^ e3)
         + 1 * (e0 ^ e2 ^ e3)
         - 2 * (e0 ^ e1 ^ e3)
         + 3 * (e0 ^ e1 ^ e2))
    # Plane  x + y + z = 1
    plane = e1 + e2 + e3 - e0
    # Line along the x-axis through the origin
    line = e2 ^ e3

    ax = graph(
        [P, "P(1,2,3)", "tab:red", plane, "plane",
         "tab:green", line, "x-axis"],
        title="PGA(3,0,1) demo", lim=2.5,
    )
    _save(ax, "ganja_pga3_demo.png")


def fig_pga2():
    PGA2 = Algebra(2, 0, 1)
    e1, e2, e0 = PGA2.basis_vectors()

    # Two points in the plane: A=(1,0), B=(0,1)
    A = (e1 ^ e2) + 1 * (e0 ^ e2)
    B = (e1 ^ e2) - 1 * (e0 ^ e1)
    # Line x + y = 1
    L = e1 + e2 - e0

    ax = graph(
        [A, "A", B, "B", "tab:orange", L, "x + y = 1"],
        title="PGA(2,0,1) demo", lim=2.0,
    )
    _save(ax, "ganja_pga2_demo.png")


def main():
    fig_pga3()
    fig_pga2()


if __name__ == "__main__":
    main()
