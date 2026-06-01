"""
3D visualisation for all GA(3,0) multivector objects using matplotlib.

GA(3,0) grade structure
-----------------------
Even(w, x, y, z)    grade-0 scalar  +  grade-2 bivector
    w               scalar coefficient
    x               coefficient of -e2e3  (bivector)
    y               coefficient of -e3e1  (bivector)
    z               coefficient of -e1e2  (bivector)

Odd(w, x, y, z)     grade-1 vector  +  grade-3 trivector
    x, y, z         vector:    x·e1 + y·e2 + z·e3
    w               trivector: w·e1e2e3  (pseudoscalar I3)

Visual encoding
---------------
grade-0  scalar    →  solid translucent sphere,   radius = |w|
grade-1  vector    →  arrow from origin            along (x,y,z)
grade-2  bivector  →  filled disk,  normal = (−x,−y,−z),  radius = ‖bivector‖
grade-3  trivector →  wireframe sphere,  radius = |w|^(1/3)

Available methods
-----------------
GA30Visualizer.add_odd_vector(v)           – grade-1 arrow + grade-3 sphere
GA30Visualizer.add_even_element(B)         – grade-2 disk  + grade-0 sphere
GA30Visualizer.add_geometric_product(v1,v2)– inner product projection + outer product disk
GA30Visualizer.add_sandwich_rotation(R, v) – v before/after R·v·R†, with sweep arc

plot_grade_decomposition(element)          – 2-panel figure showing each grade separately

Run as a script
---------------
    python simplega/ga30_visualization.py
"""

import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import simplega.ga30 as GA30
    from simplega import project, bivector_exp
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import simplega.ga30 as GA30
    from simplega import project, bivector_exp

Even = GA30.Even
Odd  = GA30.Odd


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _grade1_vec(v: Odd) -> np.ndarray:
    """Grade-1 (vector) part of an Odd element as a numpy array."""
    g = v.project(1)
    return np.array([g.x, g.y, g.z], dtype=float)


def _grade3_scalar(v: Odd) -> float:
    """Grade-3 (trivector / I3) coefficient of an Odd element."""
    return v.project(3).w


def _grade0_scalar(B: Even) -> float:
    """Grade-0 (scalar) coefficient of an Even element."""
    return B.project(0).w


def _grade2_normal(B: Even) -> np.ndarray:
    """
    Dual vector (normal) of the grade-2 (bivector) part of an Even element.
    For Even(0, x, y, z): normal = (−x, −y, −z).
    """
    g = B.project(2)
    return np.array([-g.x, -g.y, -g.z], dtype=float)


def _plane_frame(normal: np.ndarray):
    """Orthonormal (u, v) spanning the plane ⊥ to normal with u×v = n̂."""
    n = normal / (np.linalg.norm(normal) + 1e-15)
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    u = np.cross(n, ref);  u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _disk_verts(center: np.ndarray, u: np.ndarray, v: np.ndarray,
                radius: float, n_pts: int = 48) -> np.ndarray:
    """Vertices of a disk of given radius centred at center, in the u–v plane."""
    t = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    return (center[np.newaxis, :] +
            radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v)))


def _rotor_slerp(R: Even, t: float) -> Even:
    """
    Spherical linear interpolation from Even.one() to R at fraction t.
    R is assumed to be a unit rotor; negative-w representative is flipped.
    """
    # Take the positive-w representative (shorter arc from identity)
    sign = 1.0 if R.w >= 0.0 else -1.0
    w = min(1.0, max(-1.0, sign * R.w))
    alpha = math.acos(w)
    if alpha < 1e-10:
        return Even.one()
    s = math.sin(alpha)
    a = math.sin((1.0 - t) * alpha) / s
    b = math.sin(t * alpha) / s * sign
    return Even(a + b * R.w, b * R.x, b * R.y, b * R.z)


# ---------------------------------------------------------------------------
# Main visualiser class
# ---------------------------------------------------------------------------

class GA30Visualizer:
    """
    3D visualiser for GA(3,0) multivectors.

    Each grade is rendered with a distinct visual element:
    ● grade-0  scalar    → solid translucent sphere of radius |w|
    ● grade-1  vector    → arrow from origin of length ‖v‖
    ● grade-2  bivector  → filled disk, normal = (−x,−y,−z), radius = ‖B₂‖
    ● grade-3  trivector → wireframe sphere of radius |w|^(1/3)

    Methods return ``self`` for chaining.
    """

    def __init__(self, figsize=(11, 9),
                 title: str = "GA(3,0) Visualisation",
                 elev: float = 22.0, azim: float = -55.0):
        self.fig = plt.figure(figsize=figsize)
        self.ax  = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=elev, azim=azim)
        self.ax.set_xlabel("$x$", fontsize=12, labelpad=8)
        self.ax.set_ylabel("$y$", fontsize=12, labelpad=8)
        self.ax.set_zlabel("$z$", fontsize=12, labelpad=8)
        self.ax.set_title(title, fontsize=13, pad=12)

    # ------------------------------------------------------------------
    # Low-level drawing helpers
    # ------------------------------------------------------------------

    def _arrow(self, origin, direction, color, label=None,
               lw: float = 2.2, ratio: float = 0.15, length=None):
        d = np.asarray(direction, dtype=float)
        if length is not None:
            n = np.linalg.norm(d)
            if n > 1e-14:
                d = d / n * length
        kw = dict(color=color, linewidth=lw, arrow_length_ratio=ratio)
        if label:
            kw["label"] = label
        o = np.asarray(origin, dtype=float)
        self.ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2], **kw)

    def _sphere(self, center: np.ndarray, radius: float, color,
                alpha: float = 0.22, wireframe: bool = False, n: int = 22):
        if radius < 1e-12:
            return
        u = np.linspace(0.0, 2.0 * math.pi, n)
        v = np.linspace(0.0, math.pi, n)
        xs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        zs = center[2] + radius * np.outer(np.ones(n), np.cos(v))
        if wireframe:
            self.ax.plot_wireframe(xs, ys, zs, color=color, alpha=alpha,
                                   linewidth=0.6, rstride=4, cstride=4)
        else:
            self.ax.plot_surface(xs, ys, zs, color=color, alpha=alpha,
                                 linewidth=0)

    def _disk(self, center: np.ndarray, normal: np.ndarray, radius: float,
              color, alpha: float = 0.35, n_pts: int = 48):
        if radius < 1e-12:
            return
        u, v = _plane_frame(normal)
        verts = _disk_verts(center, u, v, radius, n_pts)
        self.ax.add_collection3d(
            Poly3DCollection([verts], alpha=alpha,
                             facecolor=color, edgecolor="none")
        )
        rim = np.vstack([verts, verts[0]])
        self.ax.plot(rim[:, 0], rim[:, 1], rim[:, 2],
                     color=color, lw=0.8, alpha=0.55)

    def _right_angle_mark(self, foot: np.ndarray, dir_a: np.ndarray,
                           dir_b: np.ndarray, size: float = 0.06, color="gray"):
        """Small square at `foot` in the plane of dir_a × dir_b."""
        a = dir_a / (np.linalg.norm(dir_a) + 1e-15)
        b = dir_b / (np.linalg.norm(dir_b) + 1e-15)
        p0 = foot + size * a
        p1 = foot + size * (a + b)
        p2 = foot + size * b
        pts = np.array([foot, p0, p1, p2])
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                     color=color, lw=1.0, alpha=0.7)

    # ------------------------------------------------------------------
    # Public add_* methods
    # ------------------------------------------------------------------

    def add_odd_vector(self, v: Odd, *,
                       color: str = "steelblue",
                       label: str | None = None,
                       show_trivector: bool = True,
                       trivector_color: str | None = None,
                       arrow_lw: float = 2.4):
        """
        Draw an Odd element  v = (grade-1 vector) + (grade-3 trivector I3).

        • **Arrow**   along the grade-1 part (x, y, z), length = ‖v₁‖.
        • **Sphere**  for the grade-3 coefficient w·I3:
          solid, radius = |w|; blue = positive, red = negative.

        Parameters
        ----------
        v               : Odd multivector
        color           : colour for the grade-1 arrow
        label           : legend label
        show_trivector  : also draw the grade-3 sphere
        trivector_color : override sphere colour (default: blue/red by sign)
        """
        # ── grade-1 arrow ──────────────────────────────────────────────────
        g1 = _grade1_vec(v)
        if np.linalg.norm(g1) > 1e-12:
            self._arrow((0, 0, 0), g1, color, label=label,
                        lw=arrow_lw, ratio=0.14)
        elif label:
            self.ax.plot([], [], [], color=color, label=label)

        # ── grade-3 sphere ─────────────────────────────────────────────────
        if show_trivector:
            w3 = _grade3_scalar(v)
            if abs(w3) > 1e-12:
                sc = trivector_color or ("royalblue" if w3 > 0 else "tomato")
                self._sphere(np.zeros(3), abs(w3), sc, alpha=0.25)
                tip = np.array([0.0, 0.0, abs(w3) + 0.08])
                self.ax.text(*tip, f"I₃·{w3:+.2f}",
                             color=sc, fontsize=8, ha="center")
        return self

    def add_even_element(self, B: Even, *,
                         color: str = "darkorange",
                         label: str | None = None,
                         show_scalar: bool = True,
                         show_normal: bool = True,
                         show_adjoint: bool = False,
                         adjoint_color: str = "tomato",
                         adjoint_label: str | None = None,
                         disk_alpha: float = 0.38,
                         scalar_alpha: float = 0.22):
        """
        Draw an Even element  B = (grade-0 scalar) + (grade-2 bivector).

        • **Disk**    for the grade-2 part: normal = (−x,−y,−z), radius = ‖B₂‖.
        • **Arrow**   from origin along the normal (orientation indicator).
        • **Sphere**  for the grade-0 scalar w: wireframe, radius = |w|.
        • Optionally draw the adjoint B† (negated bivector, same scalar).

        Parameters
        ----------
        B              : Even multivector
        color          : colour for the bivector disk and normal arrow
        label          : legend label (on the normal arrow)
        show_scalar    : draw the grade-0 wireframe sphere
        show_normal    : draw the orientation arrow through the disk
        show_adjoint   : also draw B† (flipped bivector, same scalar)
        """
        normal   = _grade2_normal(B)
        biv_norm = np.linalg.norm(normal)

        # ── grade-2 disk ───────────────────────────────────────────────────
        if biv_norm > 1e-12:
            self._disk(np.zeros(3), normal, biv_norm, color, alpha=disk_alpha)
            if show_normal:
                n_hat = normal / biv_norm
                self._arrow((0, 0, 0), n_hat, color, label=label,
                            lw=2.0, ratio=0.18, length=biv_norm * 1.15)
            elif label:
                self.ax.plot([], [], [], color=color, label=label)

        # ── grade-0 scalar sphere ──────────────────────────────────────────
        if show_scalar:
            w0 = _grade0_scalar(B)
            if abs(w0) > 1e-12:
                sc = "royalblue" if w0 > 0 else "tomato"
                self._sphere(np.zeros(3), abs(w0), sc,
                             alpha=scalar_alpha, wireframe=True)
                self.ax.text(0.0, 0.0, abs(w0) + 0.08,
                             f"w={w0:+.2f}", color=sc, fontsize=8, ha="center")

        # ── adjoint B† ─────────────────────────────────────────────────────
        if show_adjoint:
            Badj     = B.adjoint
            adj_n    = _grade2_normal(Badj)
            adj_norm = np.linalg.norm(adj_n)
            clbl     = adjoint_label or (f"{label}†" if label else "B†")
            if adj_norm > 1e-12:
                self._disk(np.zeros(3), adj_n, adj_norm * 0.88,
                           adjoint_color, alpha=disk_alpha * 0.75)
                if show_normal:
                    self._arrow((0, 0, 0), adj_n / adj_norm, adjoint_color,
                                label=clbl, lw=1.6, ratio=0.18,
                                length=adj_norm * 0.95)
        return self

    def add_geometric_product(self, v1: Odd, v2: Odd, *,
                               color_v1:    str = "steelblue",
                               color_v2:    str = "seagreen",
                               color_inner: str = "gold",
                               color_outer: str = "darkorange",
                               label_v1:    str | None = "v₁",
                               label_v2:    str | None = "v₂",
                               show_inner:  bool = True,
                               show_outer:  bool = True,
                               disk_alpha:  float = 0.32):
        """
        Visualise the geometric product  v1 · v2 = ⟨v1,v2⟩ + v1∧v2.

        In GA(3,0) the geometric product of two grade-1 vectors decomposes as:
            v1 · v2  =  ⟨v1, v2⟩  +  v1 ∧ v2
                     =  (scalar)   +  (bivector)

        Drawn elements
        --------------
        • **v₁** and **v₂** as arrows.
        • **Inner product** ⟨v1,v2⟩ (grade-0):  dashed projection of v₁ onto
          the v₂ direction, right-angle mark at the foot, scalar label.
        • **Outer product** v1∧v2 (grade-2):  filled disk (normal ∝ v1×v2,
          radius = ‖v1∧v2‖) plus the spanning parallelogram.

        Parameters
        ----------
        v1, v2      : Odd elements (grade-1 parts are used; trivectors ignored)
        show_inner  : draw the inner-product projection
        show_outer  : draw the outer-product disk and parallelogram
        """
        a = _grade1_vec(v1)
        b = _grade1_vec(v2)

        # ── vector arrows ──────────────────────────────────────────────────
        self._arrow((0, 0, 0), a, color_v1, label=label_v1, lw=2.4, ratio=0.14)
        self._arrow((0, 0, 0), b, color_v2, label=label_v2, lw=2.4, ratio=0.14)

        # ── inner product  ⟨v1, v2⟩ ───────────────────────────────────────
        if show_inner:
            result    = v1 * v2
            inner_val = result.project(0).w
            b_norm    = np.linalg.norm(b)
            if b_norm > 1e-12:
                # vector projection of a onto b
                proj   = (inner_val / (b_norm ** 2)) * b
                perp   = a - proj
                # dashed projection line
                self.ax.plot([0, proj[0]], [0, proj[1]], [0, proj[2]],
                             color=color_inner, lw=2.0, linestyle="--",
                             alpha=0.9, label=f"⟨v₁,v₂⟩ = {inner_val:.2f}")
                # right-angle mark at foot of perpendicular
                if np.linalg.norm(perp) > 1e-10:
                    size = min(np.linalg.norm(a), b_norm) * 0.08
                    self._right_angle_mark(proj, perp, b, size=size,
                                           color=color_inner)

        # ── outer product  v1∧v2 ──────────────────────────────────────────
        if show_outer:
            result   = v1 * v2
            biv_part = result.project(2)
            normal   = np.array([-biv_part.x, -biv_part.y, -biv_part.z])
            biv_norm = np.linalg.norm(normal)
            if biv_norm > 1e-12:
                # filled disk
                self._disk(np.zeros(3), normal, biv_norm,
                           color_outer, alpha=disk_alpha)
                # spanning parallelogram
                verts = np.array([[0, 0, 0], a, a + b, b])
                self.ax.add_collection3d(
                    Poly3DCollection([verts], alpha=disk_alpha * 0.55,
                                     facecolor=color_outer,
                                     edgecolor=color_outer, linewidth=0.8)
                )
                # normal arrow
                n_hat = normal / biv_norm
                self._arrow((0, 0, 0), n_hat, color_outer,
                            label=f"v₁∧v₂  ‖‖={biv_norm:.2f}",
                            lw=1.8, ratio=0.20, length=biv_norm * 1.15)
        return self

    def add_sandwich_rotation(self, R: Even, v: Odd, *,
                               color_before: str = "lightgray",
                               color_after:  str = "steelblue",
                               color_axis:   str = "darkorange",
                               label_before: str | None = "v",
                               label_after:  str | None = "R·v·R†",
                               show_plane:   bool = True,
                               show_arc:     bool = True,
                               arc_steps:    int = 48):
        """
        Visualise the sandwich product  R · v · R†  for a grade-1 vector v.

        R is normalised internally so non-unit rotors are accepted.

        Drawn elements
        --------------
        • **Before arrow**: grade-1 part of v (light gray).
        • **After arrow**:  grade-1 part of R·v·R† (coloured).
        • **Rotation axis**: normal of R's bivector plane.
        • **Rotation-plane disk**: the plane in which v rotates (optional).
        • **Sweep arc**: curved path from v to R·v·R† (dashed).

        Parameters
        ----------
        R           : Even rotor (normalised internally)
        v           : Odd element (grade-1 part is rotated)
        show_plane  : draw the rotation-plane disk
        show_arc    : draw the swept arc
        arc_steps   : number of arc segments
        """
        # normalise R
        rn = math.sqrt(R.w**2 + R.x**2 + R.y**2 + R.z**2)
        if rn < 1e-14:
            raise ValueError("Rotor R must not be zero.")
        Rn = Even(R.w / rn, R.x / rn, R.y / rn, R.z / rn)

        # result of sandwich
        result = Rn * v * Rn.adjoint
        v1  = _grade1_vec(v)
        rv1 = _grade1_vec(result)

        # ── before / after arrows ──────────────────────────────────────────
        n1 = np.linalg.norm(v1)
        if n1 > 1e-12:
            self._arrow((0, 0, 0), v1, color_before,
                        label=label_before, lw=1.8, ratio=0.14)
        rn1 = np.linalg.norm(rv1)
        if rn1 > 1e-12:
            self._arrow((0, 0, 0), rv1, color_after,
                        label=label_after, lw=2.4, ratio=0.14)

        # ── rotation axis + plane ──────────────────────────────────────────
        normal   = _grade2_normal(Rn)
        biv_norm = np.linalg.norm(normal)
        ax_len   = max(n1, 0.8) * 1.2

        if biv_norm > 1e-12:
            n_hat = normal / biv_norm
            self._arrow((0, 0, 0), n_hat, color_axis,
                        label="rotation axis", lw=1.8, ratio=0.18,
                        length=ax_len)
            if show_plane:
                self._disk(np.zeros(3), normal, max(n1, biv_norm) * 1.05,
                           color_axis, alpha=0.10)

        # ── sweep arc ─────────────────────────────────────────────────────
        if show_arc and biv_norm > 1e-12:
            pts = []
            for k in range(arc_steps + 1):
                R_t = _rotor_slerp(Rn, k / arc_steps)
                pts.append(_grade1_vec(R_t * v * R_t.adjoint))
            pts = np.array(pts)
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                         color=color_after, lw=1.8,
                         linestyle="--", alpha=0.75)
            # arrowhead at arc end
            tang = pts[-1] - pts[-3]
            tn   = np.linalg.norm(tang)
            if tn > 1e-14:
                tang /= tn
                self.ax.quiver(*pts[-2], *(tang * 0.07),
                               color=color_after, lw=1.6,
                               arrow_length_ratio=0.99)
        return self

    # ------------------------------------------------------------------
    # Scene helpers
    # ------------------------------------------------------------------

    def draw_axes(self, length: float = 0.8, alpha: float = 0.45):
        """Draw the reference x, y, z frame arrows."""
        for direction, color, lbl in [
            ([1, 0, 0], "crimson",     "$x$"),
            ([0, 1, 0], "forestgreen", "$y$"),
            ([0, 0, 1], "royalblue",   "$z$"),
        ]:
            d = np.array(direction, dtype=float) * length
            self.ax.quiver(0, 0, 0, d[0], d[1], d[2],
                           color=color, alpha=alpha,
                           linewidth=1.5, arrow_length_ratio=0.14)
            off = np.array(direction, dtype=float) * (length + 0.10)
            self.ax.text(*off, lbl, color=color, fontsize=11, ha="center")
        return self

    def set_limits(self, lim: float = 1.5):
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        return self

    def set_view(self, elev: float, azim: float):
        self.ax.view_init(elev=elev, azim=azim)
        return self

    def legend(self, **kwargs):
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels, **kwargs)
        return self

    def show(self, tight: bool = True):
        if tight:
            plt.tight_layout()
        plt.show()
        return self

    def save(self, path: str, dpi: int = 150, **kwargs):
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
        return self

    def close(self):
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# Grade decomposition — 2-panel standalone figure
# ---------------------------------------------------------------------------

def _panel_axes(fig, i: int, n: int, title: str, elev: float, azim: float):
    """Create a 3-D subplot panel, draw reference axes, return the axes."""
    ax = fig.add_subplot(1, n, i + 1, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("$x$", labelpad=4, fontsize=9)
    ax.set_ylabel("$y$", labelpad=4, fontsize=9)
    ax.set_zlabel("$z$", labelpad=4, fontsize=9)
    for d, c in [([1,0,0],"crimson"),([0,1,0],"forestgreen"),([0,0,1],"royalblue")]:
        dd = np.array(d, dtype=float) * 0.65
        ax.quiver(0,0,0, dd[0], dd[1], dd[2], color=c, alpha=0.40,
                  linewidth=1.2, arrow_length_ratio=0.16)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_zlim(-1.4, 1.4)
    return ax


def _draw_sphere_on_ax(ax, radius: float, color, alpha: float,
                        wireframe: bool = False, n: int = 22):
    if radius < 1e-12:
        return
    u = np.linspace(0, 2 * math.pi, n)
    v = np.linspace(0, math.pi, n)
    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones(n), np.cos(v))
    if wireframe:
        ax.plot_wireframe(xs, ys, zs, color=color, alpha=alpha,
                          linewidth=0.6, rstride=4, cstride=4)
    else:
        ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0)


def plot_grade_decomposition(element, *,
                              title: str | None = None,
                              figsize=(13, 5),
                              elev: float = 22.0,
                              azim: float = -55.0):
    """
    Two-panel grade decomposition of an Even or Odd multivector.

    For **Even(w, x, y, z)**:
      Left panel  — grade-0 scalar: solid sphere, radius = |w|.
      Right panel — grade-2 bivector: filled disk, normal = (−x,−y,−z).

    For **Odd(w, x, y, z)**:
      Left panel  — grade-1 vector: arrow along (x, y, z).
      Right panel — grade-3 trivector: wireframe sphere, radius = |w|^(1/3).

    Returns the matplotlib Figure.
    """
    is_even = isinstance(element, Even)

    fig = plt.figure(figsize=figsize)
    sup = title or (repr(element))
    fig.suptitle(f"Grade decomposition:  {sup}", fontsize=12, y=1.01)

    if is_even:
        panel_titles = ["grade-0  scalar  w",
                        "grade-2  bivector  (−x)e₂e₃+(−y)e₃e₁+(−z)e₁e₂"]
    else:
        panel_titles = ["grade-1  vector  x·e₁+y·e₂+z·e₃",
                        "grade-3  trivector  w·I₃"]

    for i, ptitle in enumerate(panel_titles):
        ax = _panel_axes(fig, i, 2, ptitle, elev, azim)

        if is_even:
            if i == 0:                          # grade-0 scalar
                w = _grade0_scalar(element)
                if abs(w) > 1e-12:
                    sc = "royalblue" if w > 0 else "tomato"
                    _draw_sphere_on_ax(ax, abs(w), sc, alpha=0.30)
                    ax.text(0, 0, abs(w) + 0.12, f"w = {w:+.3f}",
                            fontsize=9, color=sc, ha="center")
                else:
                    ax.text(0, 0, 0, "0", fontsize=16, ha="center",
                            va="center", color="gray")

            else:                               # grade-2 bivector
                normal   = _grade2_normal(element)
                biv_norm = np.linalg.norm(normal)
                if biv_norm > 1e-12:
                    u, v = _plane_frame(normal)
                    verts = _disk_verts(np.zeros(3), u, v, biv_norm)
                    ax.add_collection3d(
                        Poly3DCollection([verts], alpha=0.42,
                                         facecolor="darkorange", edgecolor="none")
                    )
                    rim = np.vstack([verts, verts[0]])
                    ax.plot(rim[:,0], rim[:,1], rim[:,2],
                            color="darkorange", lw=0.8, alpha=0.55)
                    n_hat = normal / biv_norm
                    ax.quiver(0, 0, 0,
                              n_hat[0]*biv_norm*1.2,
                              n_hat[1]*biv_norm*1.2,
                              n_hat[2]*biv_norm*1.2,
                              color="darkorange", lw=2.0,
                              arrow_length_ratio=0.18)
                    ax.text(*(n_hat * (biv_norm + 0.18)),
                            f"‖B₂‖={biv_norm:.3f}",
                            fontsize=9, color="darkorange", ha="center")
                else:
                    ax.text(0, 0, 0, "0", fontsize=16, ha="center",
                            va="center", color="gray")
        else:
            if i == 0:                          # grade-1 vector
                vec = _grade1_vec(element)
                vn  = np.linalg.norm(vec)
                if vn > 1e-12:
                    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
                              color="steelblue", lw=2.4,
                              arrow_length_ratio=0.14)
                    tip = vec / vn * (vn + 0.15)
                    ax.text(*tip, f"‖v₁‖={vn:.3f}",
                            fontsize=9, color="steelblue", ha="center")
                else:
                    ax.text(0, 0, 0, "0", fontsize=16, ha="center",
                            va="center", color="gray")

            else:                               # grade-3 trivector
                w3 = _grade3_scalar(element)
                if abs(w3) > 1e-12:
                    sc = "royalblue" if w3 > 0 else "tomato"
                    r  = abs(w3) ** (1.0 / 3.0)
                    _draw_sphere_on_ax(ax, r, sc, alpha=0.35, wireframe=True)
                    ax.text(0, 0, r + 0.12, f"I₃·{w3:+.3f}",
                            fontsize=9, color=sc, ha="center")
                else:
                    ax.text(0, 0, 0, "0", fontsize=16, ha="center",
                            va="center", color="gray")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Script entry point  –  python simplega/ga30_visualization.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    e1, e2, e3, I3 = GA30.e1, GA30.e2, GA30.e3, GA30.I3

    print("Opening 5 figures — close them to exit.")
    print("  Figure 1 : Odd element  — grade-1 vector + grade-3 trivector")
    print("  Figure 2 : Even element — grade-0 scalar + grade-2 bivector")
    print("  Figure 3 : Geometric product  v1·v2 = ⟨v1,v2⟩ + v1∧v2")
    print("  Figure 4 : Sandwich rotation  R·v·R†")
    print("  Figure 5 : Grade decomposition panels (2 figures)")

    # ── Figure 1: Odd element — vector + trivector ────────────────────────
    # v = 0.8·e1 + 0.5·e2 − 0.6·e3 + 0.4·I3
    v_odd = 0.8 * e1 + 0.5 * e2 + (-0.6) * e3 + 0.4 * I3
    viz1 = GA30Visualizer(
        title="Odd element   v = 0.8·e₁ + 0.5·e₂ − 0.6·e₃ + 0.4·I₃\n"
              "arrow = grade-1 vector  ·  sphere = grade-3 trivector",
        elev=25, azim=-50,
    )
    (viz1
     .draw_axes()
     .set_limits(1.4)
     .add_odd_vector(v_odd, color="steelblue", label="v")
     .legend(fontsize=9))

    # ── Figure 2: Even element — scalar + bivector ────────────────────────
    # B = 0.4 + 0.7·e1e2 + 0.5·e2e3
    # e1e2 = Even(0,0,0,-1), so 0.7·e1e2 → z = -0.7
    # e2e3 = Even(0,-1,0,0), so 0.5·e2e3 → x = -0.5
    B_even = Even(0.4, -0.5, 0.0, -0.7)
    viz2 = GA30Visualizer(
        title="Even element   B = 0.4 + 0.5·e₂e₃ + 0.7·e₁e₂\n"
              "disk = grade-2 bivector  ·  wireframe sphere = grade-0 scalar",
        elev=28, azim=-45,
    )
    (viz2
     .draw_axes()
     .set_limits(1.4)
     .add_even_element(B_even, color="darkorange", label="B",
                       show_adjoint=True, adjoint_label="B†")
     .legend(fontsize=9))

    # ── Figure 3: Geometric product v1·v2 ─────────────────────────────────
    v1 = 0.9 * e1 + 0.4 * e2
    v2 = 0.4 * e1 + 0.8 * e3
    viz3 = GA30Visualizer(
        title="Geometric product   v₁·v₂ = ⟨v₁,v₂⟩ + v₁∧v₂\n"
              "dashed line = inner product  ·  disk + parallelogram = outer product",
        elev=28, azim=-45,
    )
    (viz3
     .draw_axes()
     .set_limits(1.5)
     .add_geometric_product(v1, v2)
     .legend(fontsize=9))

    # ── Figure 4: Sandwich rotation R·v·R† ───────────────────────────────
    # 90° CCW rotation around z: use B = -(pi/4)·e1e2
    # e1e2 = Even(0,0,0,-1), so -(pi/4)·e1e2 = Even(0,0,0,pi/4)
    B_rot = Even(0.0, 0.0, 0.0, math.pi / 4)
    R = B_rot.bivector_exp()                       # unit rotor, 90° around z
    v_rot = 0.8 * e1 + 0.2 * e2 + 0.5 * e3       # general vector
    viz4 = GA30Visualizer(
        title="Sandwich rotation   R·v·R†\n"
              "R = bivector_exp(−(π/4)·e₁e₂)  →  90° CCW around ẑ",
        elev=28, azim=-50,
    )
    (viz4
     .draw_axes()
     .set_limits(1.3)
     .add_sandwich_rotation(R, v_rot)
     .legend(fontsize=9))

    # ── Figure 5a/5b: grade decomposition panels ──────────────────────────
    fig5a = plot_grade_decomposition(
        v_odd,
        title="Odd(0.4, 0.8, 0.5, −0.6)",
        elev=25, azim=-50,
    )
    fig5b = plot_grade_decomposition(
        B_even,
        title="Even(0.4, −0.5, 0.0, −0.7)",
        elev=28, azim=-45,
    )

    plt.show()
