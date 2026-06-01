"""Generate GA(3,0) example figures used in the README.

Run from the repository root:

    python doc/generate_simplega_figures.py
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

from simplega import ga30 as GA30  # noqa: E402
from simplega.ga30_visualization import (  # noqa: E402
    GA30Visualizer,
    plot_grade_decomposition,
)

Even = GA30.Even
Odd = GA30.Odd
e1, e2, e3, I3 = GA30.e1, GA30.e2, GA30.e3, GA30.I3

DOC_DIR = os.path.dirname(os.path.abspath(__file__))


def _save(fig, name):
    path = os.path.join(DOC_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_odd():
    v = 0.8 * e1 + 0.5 * e2 + (-0.6) * e3 + 0.4 * I3
    viz = GA30Visualizer(
        title="Odd  v = 0.8e1 + 0.5e2 - 0.6e3 + 0.4 I3",
        elev=25, azim=-50,
    )
    (viz.draw_axes()
        .set_limits(1.4)
        .add_odd_vector(v, color="steelblue", label="v")
        .legend(fontsize=9))
    _save(viz.fig, "ga30_odd.png")


def fig_even():
    B = Even(0.4, -0.5, 0.0, -0.7)
    viz = GA30Visualizer(
        title="Even  B = 0.4 + 0.5 e2e3 + 0.7 e1e2  (with adjoint)",
        elev=28, azim=-45,
    )
    (viz.draw_axes()
        .set_limits(1.4)
        .add_even_element(B, color="darkorange", label="B",
                          show_adjoint=True, adjoint_label="B†")
        .legend(fontsize=9))
    _save(viz.fig, "ga30_even.png")


def fig_geometric_product():
    v1 = 0.9 * e1 + 0.4 * e2
    v2 = 0.4 * e1 + 0.8 * e3
    viz = GA30Visualizer(
        title="Geometric product  v1 v2 = <v1,v2> + v1 ^ v2",
        elev=28, azim=-45,
    )
    (viz.draw_axes()
        .set_limits(1.5)
        .add_geometric_product(v1, v2)
        .legend(fontsize=9))
    _save(viz.fig, "ga30_geometric_product.png")


def fig_sandwich():
    B_rot = Even(0.0, 0.0, 0.0, math.pi / 4)
    R = B_rot.bivector_exp()
    v = 0.8 * e1 + 0.2 * e2 + 0.5 * e3
    viz = GA30Visualizer(
        title="Sandwich rotation  R v R†  (90° around z)",
        elev=28, azim=-50,
    )
    (viz.draw_axes()
        .set_limits(1.3)
        .add_sandwich_rotation(R, v)
        .legend(fontsize=9))
    _save(viz.fig, "ga30_sandwich.png")


def fig_grade_decomposition():
    B = Even(0.4, -0.5, 0.0, -0.7)
    fig = plot_grade_decomposition(
        B, title="Even(0.4, -0.5, 0.0, -0.7)",
        elev=28, azim=-45,
    )
    _save(fig, "ga30_grades_even.png")

    v = 0.8 * e1 + 0.5 * e2 + (-0.6) * e3 + 0.4 * I3
    fig = plot_grade_decomposition(
        v, title="Odd(0.4, 0.8, 0.5, -0.6)",
        elev=25, azim=-50,
    )
    _save(fig, "ga30_grades_odd.png")


def main():
    fig_odd()
    fig_even()
    fig_geometric_product()
    fig_sandwich()
    fig_grade_decomposition()


if __name__ == "__main__":
    main()
