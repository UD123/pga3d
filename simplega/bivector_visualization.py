"""
3D visualization of GA(3,0) bivectors and their adjoints using matplotlib.

Bivector geometry reminder
--------------------------
In GA(3,0) an even element  B = Even(0, x, y, z)  represents:
    (-x)*e2e3  +  (-y)*e3e1  +  (-z)*e1e2

Its dual vector (normal to the plane) is:
    n = (-x)*e1  +  (-y)*e2  +  (-z)*e3

The reverse (adjoint) of B negates the bivector components:
    B† = Even(0, -x, -y, -z) = -B
which flips the orientation: same plane, opposite rotation sense.

Usage
-----
>>> from simplega.visualization import BivectorVisualizer
>>> import simplega.ga30 as GA30
>>>
>>> viz = BivectorVisualizer()
>>> viz.draw_axes().add_basis_bivectors(show_adjoints=True)
>>> viz.legend().show()
"""

import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 – registers 3d projection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Allow `python simplega/visualization.py` from the repo root.
# When imported as a normal module sys.path already contains the repo root,
# so this insert is a harmless no-op in that case.
try:
    import simplega.ga30 as GA30
    from simplega import project, norm, adjoint, bivector_exp, inject
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import simplega.ga30 as GA30
    from simplega import project, norm, adjoint, bivector_exp, inject


# ---------------------------------------------------------------------------
# Geometry helpers (module-level, no state)
# ---------------------------------------------------------------------------

def _bivector_normal(B) -> np.ndarray:
    """
    Return the dual vector of a GA(3,0) bivector as a 3-component numpy array.

    For Even(0, x, y, z):
        e2e3 coeff = -x  →  e1 component of dual = -x
        e3e1 coeff = -y  →  e2 component of dual = -y
        e1e2 coeff = -z  →  e3 component of dual = -z
    """
    b = project(B, 2)
    return np.array([-b.x, -b.y, -b.z], dtype=float)


def _plane_frame(normal: np.ndarray):
    """
    Build an orthonormal frame (u, v) for the plane perpendicular to *normal*
    such that  u × v = n̂  (right-handed, positive orientation w.r.t. normal).
    """
    nrm = np.linalg.norm(normal)
    if nrm < 1e-14:
        return np.array([1., 0., 0.]), np.array([0., 1., 0.])
    n = normal / nrm
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)   # already unit; satisfies u × v = n
    return u, v


def _disk_verts(center: np.ndarray, u: np.ndarray, v: np.ndarray,
                radius: float, n_pts: int = 64) -> np.ndarray:
    """Return (n_pts, 3) array of vertices on a disk perimeter."""
    theta = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    return center + np.outer(np.cos(theta), radius * u) + np.outer(np.sin(theta), radius * v)


def _arc_pts(center: np.ndarray, u: np.ndarray, v: np.ndarray,
             radius: float, sign: float = 1.0, arc_frac: float = 0.75,
             n_pts: int = 48) -> np.ndarray:
    """
    Return points on an arc in the (u, v) plane for an orientation indicator.

    sign=+1  →  CCW when viewed from  u × v  (the normal)
    sign=-1  →  CW  when viewed from  u × v  (reversed orientation)
    """
    start = 0.15 * math.pi
    end   = start + sign * arc_frac * 2.0 * math.pi
    theta = np.linspace(start, end, n_pts)
    r = radius * 1.10    # slightly outside the disk
    return center + np.outer(np.cos(theta), r * u) + np.outer(np.sin(theta), r * v)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class BivectorVisualizer:
    """
    3D visualizer for GA(3,0) bivectors and their adjoints.

    Each bivector is rendered as:
    ● A filled plane patch (disk) in the bivector's 2D subspace.
    ● A dual-vector arrow normal to that plane (the orientation axis).
    ● A curved orientation arrow on the disk rim showing the rotation sense.

    The adjoint  B† = -B  has the same plane but reversed orientation,
    shown by flipping both the normal arrow and the curved arrow.

    Parameters
    ----------
    figsize : tuple
        Matplotlib figure size in inches.
    title : str
        Figure title.
    elev, azim : float
        Initial 3D viewing angles.
    """

    def __init__(self, figsize=(11, 9),
                 title="Bivector Visualization — GA(3,0)",
                 elev: float = 22.0, azim: float = -60.0):
        self.fig = plt.figure(figsize=figsize)
        self.ax  = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_xlabel("$e_1$", fontsize=12, labelpad=8)
        self.ax.set_ylabel("$e_2$", fontsize=12, labelpad=8)
        self.ax.set_zlabel("$e_3$", fontsize=12, labelpad=8)
        self.ax.set_title(title, fontsize=13, pad=12)
        self._legend_handles: list = []

    # ------------------------------------------------------------------
    # Low-level drawing primitives
    # ------------------------------------------------------------------

    def _draw_disk(self, center: np.ndarray, u: np.ndarray, v: np.ndarray,
                   radius: float, color, alpha: float):
        """Fill a disk and draw its boundary."""
        verts = _disk_verts(center, u, v, radius)
        poly  = Poly3DCollection([verts], alpha=alpha,
                                 facecolor=color, edgecolor="none")
        self.ax.add_collection3d(poly)
        loop = np.vstack([verts, verts[:1]])
        self.ax.plot(loop[:, 0], loop[:, 1], loop[:, 2],
                     color=color, lw=1.3, alpha=0.85)

    def _draw_normal_arrow(self, center: np.ndarray, normal: np.ndarray,
                           color, label: str | None = None):
        """Arrow along the dual vector (normal to the plane)."""
        mag = np.linalg.norm(normal)
        if mag < 1e-14:
            return
        length = mag
        n_hat  = normal / mag
        kw = dict(color=color, linewidth=2.2, arrow_length_ratio=0.18)
        if label:
            kw["label"] = label
        self.ax.quiver(center[0], center[1], center[2],
                       n_hat[0] * length, n_hat[1] * length, n_hat[2] * length,
                       **kw)

    def _draw_orientation_arc(self, center: np.ndarray, u: np.ndarray, v: np.ndarray,
                               radius: float, color, sign: float = 1.0):
        """
        Curved arrow on the disk rim.
        sign=+1: CCW when viewed from the normal (positive orientation).
        sign=-1: CW  when viewed from the normal (adjoint / reversed orientation).
        """
        pts = _arc_pts(center, u, v, radius, sign=sign)
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                     color=color, lw=2.5, alpha=0.9)
        # Arrowhead: a very short quiver at the arc tip
        tip = pts[-1]
        tang = pts[-1] - pts[-3]
        tang_len = np.linalg.norm(tang)
        if tang_len < 1e-14:
            return
        tang /= tang_len
        head_len = 0.06 * radius
        self.ax.quiver(tip[0], tip[1], tip[2],
                       tang[0] * head_len, tang[1] * head_len, tang[2] * head_len,
                       color=color, linewidth=2.2, arrow_length_ratio=0.99)

    # ------------------------------------------------------------------
    # Public add_* methods
    # ------------------------------------------------------------------

    def add_bivector(self, B, position=(0., 0., 0.), *,
                     color="steelblue", label: str | None = None,
                     alpha: float = 0.35, radius: float | None = None,
                     draw_normal: bool = True,
                     draw_orientation: bool = True,
                     show_adjoint: bool = False,
                     adjoint_color="tomato",
                     adjoint_label: str | None = None):
        """
        Add a GA(3,0) bivector to the scene.

        Parameters
        ----------
        B : GA30.Even
            An even-grade element (only the grade-2 part is used).
        position : (3,) array-like
            Origin of the disk and arrows.
        color : str or colour spec
            Colour for the bivector disk, normal arrow, and orientation arc.
        label : str, optional
            Legend label for the normal arrow.
        alpha : float
            Disk opacity in [0, 1].
        radius : float, optional
            Disk radius. Defaults to  sqrt(||B||)  so area = π·||B||².
        draw_normal : bool
            Draw the dual-vector (normal) arrow.
        draw_orientation : bool
            Draw the curved orientation arc.
        show_adjoint : bool
            Also draw  B† = -B  (same plane, opposite orientation).
        adjoint_color : str
            Colour for the adjoint rendering.
        adjoint_label : str, optional
            Legend label for the adjoint normal arrow.

        Returns
        -------
        self  (for chaining)
        """
        bvec   = project(B, 2)
        center = np.asarray(position, dtype=float)
        n_vec  = _bivector_normal(bvec)
        mag    = float(norm(bvec))

        # Radius: defaults so the disk area = π·||B||²
        r = radius if radius is not None else (math.sqrt(mag) if mag > 1e-14 else 0.5)
        u, v = _plane_frame(n_vec)

        # ---- draw bivector B ----
        self._draw_disk(center, u, v, r, color, alpha)
        if draw_normal:
            self._draw_normal_arrow(center, n_vec, color, label=label)
        if draw_orientation and r > 1e-14:
            self._draw_orientation_arc(center, u, v, r, color, sign=+1.0)

        # ---- optionally draw adjoint B† ----
        if show_adjoint:
            Badj  = adjoint(bvec)           # = -B in GA(3,0)
            n_adj = _bivector_normal(Badj)   # = -n_vec
            albl  = adjoint_label if adjoint_label else (f"{label}†" if label else "B†")
            # Same disk (same plane), different colour
            self._draw_disk(center, u, v, r, adjoint_color, alpha)
            if draw_normal:
                self._draw_normal_arrow(center, n_adj, adjoint_color, label=albl)
            if draw_orientation and r > 1e-14:
                # sign=-1: opposite rotation sense
                self._draw_orientation_arc(center, u, v, r, adjoint_color, sign=-1.0)

        return self

    def add_vector(self, v, position=(0., 0., 0.), *,
                   color="black", label: str | None = None, lw: float = 2.5):
        """
        Draw a GA(3,0) grade-1 vector as an arrow.

        Parameters
        ----------
        v : GA30.Odd
            A grade-1 vector (only the grade-1 part is used).
        position : (3,) array-like
            Arrow base.
        """
        g1     = project(v, 1)
        center = np.asarray(position, dtype=float)
        comps  = np.array([g1.x, g1.y, g1.z], dtype=float)
        if np.linalg.norm(comps) < 1e-14:
            return self
        kw = dict(color=color, linewidth=lw, arrow_length_ratio=0.15)
        if label:
            kw["label"] = label
        self.ax.quiver(center[0], center[1], center[2],
                       comps[0], comps[1], comps[2], **kw)
        return self

    def add_vector_pair(self, v1, v2, position=(0., 0., 0.), *,
                        color_v1="royalblue", color_v2="seagreen",
                        color_bv="mediumpurple",
                        label_v1: str | None = None,
                        label_v2: str | None = None,
                        label_bv: str | None = None,
                        alpha: float = 0.35,
                        show_adjoint: bool = False):
        """
        Draw two vectors v1, v2 and their outer-product bivector v1 ∧ v2.

        The bivector is shown as a parallelogram-like disk whose area equals
        the magnitude of v1 ∧ v2.
        """
        self.add_vector(v1, position, color=color_v1, label=label_v1)
        self.add_vector(v2, position, color=color_v2, label=label_v2)
        bvec = v1 * v2   # geometric product (= outer product for grade-1)
        self.add_bivector(bvec, position, color=color_bv,
                          label=label_bv, alpha=alpha, show_adjoint=show_adjoint)
        return self

    def add_basis_bivectors(self, scale: float = 1.0, alpha: float = 0.30,
                            show_adjoints: bool = False):
        """
        Add all three coordinate bivectors: e1e2, e2e3, e3e1.

        Parameters
        ----------
        scale : float
            Multiply each unit bivector by this factor.
        alpha : float
            Disk opacity.
        show_adjoints : bool
            Also draw all three adjoints.
        """
        e1, e2, e3 = GA30.basis
        entries = [
            (scale * e1 * e2, "royalblue",  "$e_1 e_2$"),
            (scale * e2 * e3, "seagreen",   "$e_2 e_3$"),
            (scale * e3 * e1, "darkorange", "$e_3 e_1$"),
        ]
        for bv, col, lbl in entries:
            self.add_bivector(bv, color=col, label=lbl, alpha=alpha,
                              show_adjoint=show_adjoints)
        return self

    def add_rotation_sweep(self, B, v, n_steps: int = 10, *,
                           start_color="gold", end_color="orangered",
                           trail_alpha: float = 0.45,
                           label: str | None = None):
        """
        Illustrate the rotation  R(t) v R†(t)  generated by bivector B.

        Draws the vector v at  n_steps + 1  positions as t goes from 0 to 1,
        so the total rotation angle is  ||B||  radians.

        Parameters
        ----------
        B : GA30.Even  (grade-2 bivector)
            The bivector that generates the rotation.
        v : GA30.Odd   (grade-1 vector)
            The vector to be rotated.
        n_steps : int
            Number of intermediate frames (exclusive of start and end).
        start_color, end_color : str
            Arrow colours at t=0 and t=1 respectively.
        """
        from matplotlib.colors import to_rgb
        sc = np.array(to_rgb(start_color), dtype=float)
        ec = np.array(to_rgb(end_color),   dtype=float)

        for k in range(n_steps + 1):
            t     = k / n_steps
            R     = bivector_exp(t * B)
            Radj  = adjoint(R)
            rv    = R * v * Radj
            comps = np.array([project(rv, 1).x,
                              project(rv, 1).y,
                              project(rv, 1).z], dtype=float)
            col  = tuple(sc + t * (ec - sc))
            alph = trail_alpha + (1.0 - trail_alpha) * (k / n_steps)
            lbl  = label if k == n_steps else None
            kw   = dict(color=col, linewidth=1.8 + t * 0.7,
                        arrow_length_ratio=0.15, alpha=alph)
            if lbl:
                kw["label"] = lbl
            self.ax.quiver(0, 0, 0, comps[0], comps[1], comps[2], **kw)
        return self

    def add_grade_decomposition(self, B, position=(0., 0., 0.), *,
                                 scalar_color="slategray",
                                 bv_color="steelblue",
                                 alpha: float = 0.35):
        """
        Show the grade-0 (scalar) and grade-2 (bivector) components of B separately.
        The scalar part is annotated as text; the bivector part is drawn as a disk.
        """
        center = np.asarray(position, dtype=float)
        scl = float(project(B, 0).w)
        bvec = project(B, 2)
        if abs(scl) > 1e-14:
            self.ax.text(center[0], center[1], center[2] + 0.05,
                         f"scalar = {scl:.3g}",
                         color=scalar_color, fontsize=10, ha="center")
        if float(norm(bvec)) > 1e-14:
            self.add_bivector(bvec, position=position, color=bv_color,
                              label="grade-2 part", alpha=alpha)
        return self

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def draw_axes(self, length: float = 0.85, alpha: float = 0.45):
        """Draw reference coordinate frame arrows and labels."""
        for direction, color, lbl in [
            ([1, 0, 0], "crimson",    "$e_1$"),
            ([0, 1, 0], "forestgreen","$e_2$"),
            ([0, 0, 1], "royalblue",  "$e_3$"),
        ]:
            d = np.array(direction, dtype=float) * length
            self.ax.quiver(0, 0, 0, d[0], d[1], d[2],
                           color=color, alpha=alpha,
                           linewidth=1.6, arrow_length_ratio=0.14)
            off = np.array(direction, dtype=float) * (length + 0.12)
            self.ax.text(off[0], off[1], off[2], lbl,
                         color=color, fontsize=11, ha="center")
        return self

    def set_limits(self, lim: float = 1.5):
        """Set equal symmetric axis limits."""
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        return self

    def legend(self, **kwargs):
        """Add a legend if any labelled elements exist."""
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels, **kwargs)
        return self

    def set_view(self, elev: float, azim: float):
        """Adjust the 3D viewing angle."""
        self.ax.view_init(elev=elev, azim=azim)
        return self

    def show(self, tight: bool = True):
        """Display the figure (blocks until the window is closed)."""
        if tight:
            plt.tight_layout()
        plt.show()
        return self

    def save(self, path: str, dpi: int = 150, **kwargs):
        """Save the figure to *path* without displaying it."""
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        return self

    def close(self):
        """Close the matplotlib figure."""
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def visualize_bivector(B, *, show_adjoint: bool = True,
                       label: str = "B", **kwargs):
    """
    Quick single-bivector visualisation.

    >>> from simplega.visualization import visualize_bivector
    >>> import simplega.ga30 as GA30
    >>> e1, e2 = GA30.basis[:2]
    >>> visualize_bivector(e1 * e2 + 0.5 * e2 * e3, show_adjoint=True)
    """
    viz = BivectorVisualizer(**kwargs)
    viz.draw_axes().set_limits(1.4)
    viz.add_bivector(B, label=label, show_adjoint=show_adjoint,
                     adjoint_label=f"{label}†")
    viz.legend()
    viz.show()


def visualize_basis(*, show_adjoints: bool = True, **kwargs):
    """
    Show all three coordinate bivectors of GA(3,0).

    >>> from simplega.visualization import visualize_basis
    >>> visualize_basis(show_adjoints=True)
    """
    viz = BivectorVisualizer(**kwargs)
    viz.draw_axes().set_limits(1.4)
    viz.add_basis_bivectors(show_adjoints=show_adjoints)
    viz.legend()
    viz.show()


def visualize_rotation(B, v, n_steps: int = 12, **kwargs):
    """
    Visualise the rotation of vector *v* generated by bivector *B*.

    >>> from simplega.visualization import visualize_rotation
    >>> import simplega.ga30 as GA30
    >>> e1, e2, e3 = GA30.basis
    >>> import math
    >>> visualize_rotation(math.pi/2 * e1 * e2, e3)
    """
    viz = BivectorVisualizer(**kwargs)
    viz.draw_axes().set_limits(1.4)
    viz.add_bivector(B, alpha=0.20, color="steelblue", label="rotation plane")
    viz.add_rotation_sweep(B, v, n_steps=n_steps, label="rotated vector")
    viz.legend()
    viz.show()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    e1, e2, e3 = GA30.basis

    print("Opening 3 figures — close each window to exit.")
    print("  Figure 1 : Coordinate bivectors e1e2, e2e3, e3e1 with adjoints")
    print("  Figure 2 : A general bivector  B = e1e2 + 0.5·e2e3  and its adjoint B†")
    print("  Figure 3 : Rotation sweep  R(t)·e3·R†(t)  for B = (π/2)·e1e2")

    # ── Figure 1: all three coordinate bivectors + their adjoints ──────────
    viz1 = BivectorVisualizer(title="Coordinate Bivectors + Adjoints  (GA 3,0)")
    (viz1
     .draw_axes()
     .set_limits(1.4)
     .add_basis_bivectors(show_adjoints=True)
     .legend(loc="upper left", fontsize=9))

    # ── Figure 2: a general bivector and its adjoint ───────────────────────
    B = e1 * e2 + 0.5 * e2 * e3
    viz2 = BivectorVisualizer(
        title="Bivector  B = e₁e₂ + 0.5·e₂e₃  and its Adjoint B†",
        elev=30, azim=-50,
    )
    (viz2
     .draw_axes()
     .set_limits(1.4)
     .add_bivector(B, label="B", show_adjoint=True, adjoint_label="B†",
                   color="steelblue", adjoint_color="tomato", alpha=0.40)
     .legend())

    # ── Figure 3: rotation sweep R(t)·v·R†(t) ─────────────────────────────
    B_rot = (math.pi / 2) * e1 * e2        # bivector that generates the rotation
    v_init = e3                             # initial vector being rotated
    viz3 = BivectorVisualizer(
        title="Rotation Sweep  R(t)·e₃·R†(t),  B = (π/2)·e₁e₂",
        elev=28, azim=-65,
    )
    (viz3
     .draw_axes()
     .set_limits(1.3)
     .add_bivector(B_rot, color="steelblue", alpha=0.18,
                   label="rotation plane (e₁e₂)")
     .add_rotation_sweep(B_rot, v_init, n_steps=12, label="rotated e₃")
     .legend())

    plt.show()
