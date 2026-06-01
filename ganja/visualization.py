"""
Matplotlib-based visualization for `ganja.Algebra` multivectors.

API inspired by ganja.js's `Algebra.graph(...)` function, but written
from scratch for Python / matplotlib. Currently supports 2-D PGA
(signature (2, 0, 1)) and 3-D PGA (signature (3, 0, 1)).

Usage
-----
    from ganja import Algebra, graph

    PGA3 = Algebra(3, 0, 1)
    e0, e1, e2, e3 = PGA3.basis_vectors()
    p1 = e0 + 1*e1 + 0*e2 + 0*e3        # a point
    p2 = e0 + 0*e1 + 1*e2 + 0*e3
    p3 = e0 + 0*e1 + 0*e2 + 1*e3
    line = p1 & p2                       # not provided here; use dual/wedge
    graph([p1, "p1", p2, p3], title="Three points")

The `graph` function accepts a list of items. Each item is either:
    * a `Multivector` recognised as a point, line, or plane,
    * a Python `str` — label attached to the previous item,
    * a 3-tuple `(r, g, b)` or matplotlib colour string — colour for the
      next item,
    * a numpy-array `(N, 2)` or `(N, 3)` — polyline drawn directly,
    * a callable — invoked with no args; its return value is rendered.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import math
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

from .algebra import Algebra, Multivector


# ---------------------------------------------------------------------------
# Element recognition for PGA(2,0,1) and PGA(3,0,1)
# ---------------------------------------------------------------------------

def _classify_pga(mv: Multivector) -> Optional[str]:
    """
    Identify a PGA multivector as 'point', 'line', or 'plane', or return
    None if it is not a pure k-blade of an interesting grade.
    """
    spec = mv._spec
    grades = mv.grades()
    if len(grades) != 1:
        return None
    g = grades[0]
    if (spec.p, spec.q, spec.r) == (2, 0, 1):
        if g == 2:
            return "point"      # bivectors are points in 2D PGA
        if g == 1:
            return "line"       # vectors are lines in 2D PGA
    if (spec.p, spec.q, spec.r) == (3, 0, 1):
        if g == 3:
            return "point"      # trivectors are points in 3D PGA
        if g == 2:
            return "line"
        if g == 1:
            return "plane"
    return None


# ---------------------------------------------------------------------------
# Coordinate extraction for PGA(3,0,1)
#
# Basis vectors are e1, e2, e3, e0 with e0² = 0. We use the bit layout
# bit 0 -> e1, bit 1 -> e2, bit 2 -> e3, bit 3 -> e0  (matches the order
# returned by Algebra.basis_vectors()).
#
# Point as trivector: P = w (e1 e2 e3) + x (e0 e2 e3) - y (e0 e1 e3)
#                                       + z (e0 e1 e2)
# We extract Euclidean coordinates by dividing the e0-containing parts
# by the e123 coefficient (the "homogeneous w").
# ---------------------------------------------------------------------------

def _bits(*idx: int) -> int:
    b = 0
    for i in idx:
        b |= 1 << i
    return b


def _pga3_point_xyz(mv: Multivector) -> Optional[Tuple[float, float, float]]:
    """Return (x, y, z) for a 3D PGA point (grade-3 element), or None."""
    v = mv._v
    # blades, with our bit-layout (e1=bit0, e2=bit1, e3=bit2, e0=bit3):
    b_e123 = _bits(0, 1, 2)              # 0b0111
    b_e023 = _bits(3, 1, 2)              # 0b1110  (e0 e2 e3 after reorder)
    b_e013 = _bits(3, 0, 2)              # 0b1101
    b_e012 = _bits(3, 0, 1)              # 0b1011

    w = float(v[b_e123])
    if abs(w) < 1e-15:
        return None
    # Signs come from the canonical reordering relative to e0 e1 e2 e3.
    # The point formula above uses e0 e2 e3, -(e0 e1 e3), e0 e1 e2 with
    # those signs. Our canonical order is increasing bit, so:
    #   v[e0e2e3] corresponds to +x
    #   v[e0e1e3] corresponds to -y    (note the minus)
    #   v[e0e1e2] corresponds to +z
    x = float(v[b_e023]) / w
    y = -float(v[b_e013]) / w
    z = float(v[b_e012]) / w
    return (x, y, z)


def _pga3_line_segment(mv: Multivector,
                       limit: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Approximate a 3D PGA line (grade-2 element) by two endpoints inside
    the cube [-limit, limit]^3.

    A line in 3D PGA is a bivector with 6 components: a direction
    bivector (Plücker direction) and a moment.

    Direction:  d = (v[e23], v[e31], v[e12])  with sign conventions matching
                the e1, e2, e3 vector basis.
    Moment:     m = (v[e0e1], v[e0e2], v[e0e3])  (positions on the line).

    A point on the line is `closest = d × m / |d|²`. The segment is then
    `closest ± t * d` clipped to the cube.
    """
    v = mv._v
    b_e23 = _bits(1, 2)
    b_e31 = _bits(0, 2)
    b_e12 = _bits(0, 1)
    b_e01 = _bits(3, 0)
    b_e02 = _bits(3, 1)
    b_e03 = _bits(3, 2)

    # Direction: note the −e31 sign convention (e3∧e1 = −e1∧e3).
    d = np.array([float(v[b_e23]),
                  -float(v[b_e31]),
                  float(v[b_e12])])
    m = np.array([float(v[b_e01]),
                  float(v[b_e02]),
                  float(v[b_e03])])

    d_norm2 = float(np.dot(d, d))
    if d_norm2 < 1e-20:
        return None
    closest = np.cross(d, m) / d_norm2
    d_hat = d / math.sqrt(d_norm2)
    # Long-enough segment to span the viewing cube.
    t = 2.5 * limit
    p1 = closest - t * d_hat
    p2 = closest + t * d_hat
    return p1, p2


def _pga3_plane_triangle(mv: Multivector,
                         limit: float) -> Optional[np.ndarray]:
    """
    Sample a triangle on a 3D PGA plane (grade-1 element).

    Plane:  a e1 + b e2 + c e3 + d e0   represents   a x + b y + c z + d = 0.
    """
    v = mv._v
    a = float(v[_bits(0)])
    b = float(v[_bits(1)])
    c = float(v[_bits(2)])
    d = float(v[_bits(3)])
    nvec = np.array([a, b, c])
    nlen = float(np.linalg.norm(nvec))
    if nlen < 1e-15:
        return None
    n_hat = nvec / nlen
    # closest point on the plane to the origin
    origin_on = -d * n_hat / nlen
    # any pair of in-plane orthonormal vectors
    ref = np.array([1.0, 0.0, 0.0]) if abs(n_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n_hat, ref)
    u /= max(np.linalg.norm(u), 1e-15)
    w = np.cross(n_hat, u)
    R = 1.5 * limit
    # A square (two triangles) centred on origin_on.
    p1 = origin_on + R * (+u + w)
    p2 = origin_on + R * (-u + w)
    p3 = origin_on + R * (-u - w)
    p4 = origin_on + R * (+u - w)
    return np.array([p1, p2, p3, p4])


# ---------------------------------------------------------------------------
# Coordinate extraction for PGA(2,0,1)
#
# Basis: e1, e2, e0 (with e0² = 0).
# Points are bivectors:    P = w e12 + x e02 - y e01
# Lines  are vectors:      L = a e1 + b e2 + c e0    (a x + b y + c = 0)
# ---------------------------------------------------------------------------

def _pga2_point_xy(mv: Multivector) -> Optional[Tuple[float, float]]:
    v = mv._v
    b_e12 = _bits(0, 1)
    b_e01 = _bits(2, 0)
    b_e02 = _bits(2, 1)
    w = float(v[b_e12])
    if abs(w) < 1e-15:
        return None
    x = float(v[b_e02]) / w
    y = -float(v[b_e01]) / w
    return (x, y)


def _pga2_line_segment(mv: Multivector,
                       limit: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    v = mv._v
    a = float(v[_bits(0)])
    b = float(v[_bits(1)])
    c = float(v[_bits(2)])
    nrm2 = a * a + b * b
    if nrm2 < 1e-20:
        return None
    # closest point on the line to the origin
    p0 = -c * np.array([a, b]) / nrm2
    # direction along the line
    d = np.array([-b, a]) / math.sqrt(nrm2)
    t = 2.5 * limit
    return p0 - t * d, p0 + t * d


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

Item = Union[Multivector, str, tuple, list, np.ndarray, callable]


def graph(
    items: Iterable[Item],
    *,
    algebra: Optional[Algebra] = None,
    title: str = "",
    lim: float = 2.0,
    ax: Optional[plt.Axes] = None,
    point_size: float = 60.0,
    line_width: float = 2.0,
    plane_alpha: float = 0.30,
    figsize: Tuple[float, float] = (8, 7),
    show: bool = False,
):
    """
    Render a list of geometric objects on a matplotlib axes.

    Each item in `items` is one of:

    * A `Multivector` — drawn as a point / line / plane based on its
      grade, given the algebra's signature.
    * A string — used as a label for the previously drawn item.
    * A colour (matplotlib colour string or RGB tuple of floats) —
      applied to subsequent items until the next colour.
    * An array-like of shape `(N, 2)` or `(N, 3)` — polyline.
    * A zero-argument callable — invoked, its return value is processed
      using the same rules.

    Parameters
    ----------
    algebra      explicit algebra; if omitted, inferred from the first
                 Multivector in `items`.
    lim          half-width of the viewing cube.
    ax           target axes; if None, a new figure/axes is created
                 (3-D for PGA(3,0,1), 2-D otherwise).
    show         call `plt.show()` after drawing.

    Returns
    -------
    The matplotlib `Axes` used.
    """
    items_list: List[Item] = list(items)

    # Infer algebra from the first Multivector.
    if algebra is None:
        for it in items_list:
            if isinstance(it, Multivector):
                algebra = it.algebra
                break
    if algebra is None:
        raise ValueError(
            "graph(): no Multivector found in items and no algebra given"
        )

    sig = algebra.signature
    is_3d = sig == (3, 0, 1)
    is_2d = sig == (2, 0, 1)
    if not (is_2d or is_3d):
        raise NotImplementedError(
            f"graph() currently supports PGA(2,0,1) and PGA(3,0,1); "
            f"got Algebra{sig}"
        )

    # Create axes if needed.
    own_fig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        else:
            ax = fig.add_subplot(111)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("x"); ax.set_ylabel("y")
        if title:
            ax.set_title(title)
        own_fig = True

    # Walk items in order, drawing as we go.
    current_color = None
    pending_label: Optional[str] = None
    last_anchor: Optional[Tuple[float, ...]] = None  # for label placement

    def _is_color(x) -> bool:
        if isinstance(x, str):
            try:
                from matplotlib.colors import to_rgba
                to_rgba(x)
                return True
            except (ValueError, TypeError):
                return False
        if isinstance(x, (tuple, list)) and len(x) in (3, 4):
            return all(isinstance(v, (int, float)) for v in x)
        return False

    def _draw_point(anchor):
        nonlocal last_anchor, pending_label
        if anchor is None:
            return
        if is_3d:
            ax.scatter(*anchor, color=current_color or "tab:blue",
                       s=point_size, depthshade=True, zorder=5)
        else:
            ax.scatter(*anchor, color=current_color or "tab:blue",
                       s=point_size, zorder=5)
        last_anchor = tuple(anchor)
        if pending_label is not None:
            offset = 0.05 * lim
            if is_3d:
                ax.text(anchor[0] + offset, anchor[1] + offset,
                        anchor[2] + offset,
                        pending_label, color=current_color or "black")
            else:
                ax.text(anchor[0] + offset, anchor[1] + offset,
                        pending_label, color=current_color or "black")
            pending_label = None

    def _draw_segment(p1, p2):
        nonlocal last_anchor
        col = current_color or "tab:orange"
        if is_3d:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=col, linewidth=line_width)
        else:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    color=col, linewidth=line_width)
        last_anchor = tuple((np.asarray(p1) + np.asarray(p2)) * 0.5)
        _maybe_place_label()

    def _draw_polygon(verts):
        nonlocal last_anchor
        col = current_color or "tab:green"
        if is_3d:
            poly = Poly3DCollection([verts], alpha=plane_alpha,
                                    facecolor=col, edgecolor=col)
            ax.add_collection3d(poly)
        else:
            ax.fill([p[0] for p in verts], [p[1] for p in verts],
                    color=col, alpha=plane_alpha,
                    edgecolor=col)
        last_anchor = tuple(np.mean(verts, axis=0))
        _maybe_place_label()

    def _maybe_place_label():
        nonlocal pending_label
        if pending_label is None or last_anchor is None:
            return
        col = current_color or "black"
        if is_3d:
            ax.text(last_anchor[0], last_anchor[1], last_anchor[2],
                    pending_label, color=col)
        else:
            ax.text(last_anchor[0], last_anchor[1],
                    pending_label, color=col)
        pending_label = None

    def _draw_polyline(arr):
        arr = np.asarray(arr, dtype=float)
        col = current_color or "tab:gray"
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            return
        if is_3d and arr.shape[1] == 3:
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2],
                    color=col, linewidth=line_width)
        elif not is_3d and arr.shape[1] == 2:
            ax.plot(arr[:, 0], arr[:, 1], color=col, linewidth=line_width)

    for item in items_list:
        # Unwrap callables (ganja's animation-style entries).
        if callable(item) and not isinstance(item, Multivector):
            try:
                item = item()
            except Exception:
                continue

        # Strings: either a colour or a label for the previous item.
        if isinstance(item, str):
            if _is_color(item):
                current_color = item
            else:
                pending_label = item
                # If we already drew something and have an anchor, place now.
                _maybe_place_label()
            continue

        # Tuples / lists: RGB colour or polyline data.
        if isinstance(item, (tuple, list)):
            if _is_color(item):
                current_color = item
                continue
            try:
                arr = np.asarray(item, dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.ndim == 2 and arr.shape[1] in (2, 3):
                _draw_polyline(arr)
                last_anchor = tuple(arr[-1])
                _maybe_place_label()
            continue

        if isinstance(item, np.ndarray):
            _draw_polyline(item)
            if item.size:
                last_anchor = tuple(item.reshape(-1, item.shape[-1])[-1])
                _maybe_place_label()
            continue

        if isinstance(item, Multivector):
            kind = _classify_pga(item)
            if kind is None:
                # Try point-only fallback (e.g. user passed a vector
                # intended to be a Euclidean point as numbers).
                continue
            if kind == "point":
                if is_3d:
                    _draw_point(_pga3_point_xyz(item))
                else:
                    _draw_point(_pga2_point_xy(item))
            elif kind == "line":
                seg = (_pga3_line_segment(item, lim)
                       if is_3d else _pga2_line_segment(item, lim))
                if seg is not None:
                    _draw_segment(*seg)
            elif kind == "plane":
                verts = _pga3_plane_triangle(item, lim)
                if verts is not None:
                    _draw_polygon(verts)
            continue

    if own_fig and title:
        ax.set_title(title)

    if show:
        plt.show()

    return ax
