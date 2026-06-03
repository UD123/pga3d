
import math
import os
import sys

import pytest
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

# Allow running this file directly as a script
# (python pga3d/pga3d_display.py) by making the repo root importable.
if __package__ in (None, ""):
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    from pga3d import Point, Line, Plane, Translator, Rotor
else:
    from . import Point, Line, Plane, Translator, Rotor

def test_point_creation():
    p = Point(1, 2, 3)
    assert isinstance(p, Point)
    assert hasattr(p, 'x') and hasattr(p, 'y') and hasattr(p, 'z')

def test_line_from_points():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 1, 1)
    line = Line.from_points(p1, p2)
    assert isinstance(line, Line)

def test_plane_from_points():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 0, 0)
    p3 = Point(0, 1, 0)
    plane = Plane.from_points(p1, p2, p3)
    assert isinstance(plane, Plane)

def test_translator_and_rotor():
    t = Translator.from_xyz(1, 2, 3)
    r = Rotor.from_angle_and_line(math.pi/2, Line.from_xyz(0, 0, 1))
    assert isinstance(t, Translator)
    assert isinstance(r, Rotor)

def test_transformations():
    p = Point(1, 0, 0)
    t = Translator.from_xyz(1, 0, 0)
    r = Rotor.from_angle_and_line(math.pi/2, Line.from_xyz(0, 0, 1))
    m = t * r
    p_t = t.project(p)
    p_tr = m.project(p)
    assert isinstance(p_t, Point)
    assert isinstance(p_tr, Point)

def test_projection_onto_line_and_plane():
    p1 = Point(2, 3, 4)
    p2 = Point(20, 3, 7)
    p3 = Point(9, 12, 17)
    line = Line.from_points(p3, p1)
    plane = Plane.from_points(p1, p2, p3)
    p_proj = p1.project_onto(line)
    plane_proj = plane.project_onto(Point(0, 0, 0))
    assert isinstance(p_proj, Point)
    assert isinstance(plane_proj, Plane)

def test_plot_points_lines_planes():
    # Points
    p0 = Point(0, 0, 0)
    p1 = Point(2, 3, 4)
    p2 = Point(20, 3, 7)
    p3 = Point(9, 12, 17)
    # Lines
    line1 = Line.from_points(p1, p2)
    line2 = Line.from_points(p2, p3)
    line3 = Line.from_points(p3, p1)
    # Plane
    plane1 = Plane.from_points(p1, p2, p3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
     # Plot points
    for p in [p0, p1, p2, p3]:
        # Use .value or .coords or whatever your Point class uses
        x, y, z = p._value[0], p._value[1], p._value[2]
        ax.scatter(x, y, z, label=f"Point({x},{y},{z})")
    # Plot lines (as segments between points)
    for a, b in [(p1, p2), (p2, p3), (p3, p1)]:
        ax.plot([a._value[0], b._value[0]], [a._value[1], b._value[1]], [a._value[2], b._value[2]], 'k-')
    # Plot plane (as a patch)
    pts = np.array([[p1._value[0], p1._value[1], p1._value[2]],
                    [p2._value[0], p2._value[1], p2._value[2]],
                    [p3._value[0], p3._value[1], p3._value[2]]])
    poly = Poly3DCollection([pts], alpha=0.2, color='cyan')
    ax.add_collection3d(poly)

    ax.legend()
    plt.title("PGA3D Example Objects")
    #plt.savefig("test_pga3d_plot.png")
    plt.show()
    #plt.close(fig)


def plot_points(points, ax=None, color='C0', label_prefix='Point'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for i, p in enumerate(points):
        x, y, z = p._value[0], p._value[1], p._value[2]
        ax.scatter(x, y, z, color=color, label=f"{label_prefix}{i}({x},{y},{z})")
    return ax

def plot_lines(lines, ax=None, color='k', label_prefix='Line'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    for i, (a, b) in enumerate(lines):
        ax.plot([a._value[0], b._value[0]], [a._value[1], b._value[1]], [a._value[2], b._value[2]], color=color, label=f"{label_prefix}{i}")
    return ax

def plot_plane(pts, ax=None, color='cyan', alpha=0.2, label='Plane'):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pts_np = np.array([[p._value[0], p._value[1], p._value[2]] for p in pts])
    poly = Poly3DCollection([pts_np], alpha=alpha, color=color, label=label)
    ax.add_collection3d(poly)
    return ax


# ---------------------------------------------------------------------------
# Helpers to visualize Line / Plane multivectors directly
# (useful for displaying the result of meet/join operations).
# ---------------------------------------------------------------------------

def _point_xyz(p):
    return np.array([p._value[0], p._value[1], p._value[2]], dtype=float)


def line_to_segment(line, length=10.0, refs=((0, 0, 0), (1, 1, 1), (1, -1, 2))):
    """Return two endpoints (np.ndarray) on a Line algebra element.

    The line is sampled by projecting reference points onto it. Returns
    ``None`` if the line is degenerate / ideal (cannot be visualized in 3D).
    """
    proj = []
    for r in refs:
        try:
            q = Point(*r).project_onto(line)
        except ZeroDivisionError:
            continue
        v = _point_xyz(q)
        if np.all(np.isfinite(v)):
            proj.append(v)
        if len(proj) >= 2 and np.linalg.norm(proj[-1] - proj[0]) > 1e-9:
            break
    if len(proj) < 2:
        return None
    a = proj[0]
    b = proj[-1]
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    d = d / n
    mid = 0.5 * (a + b)
    return mid - d * length, mid + d * length


def plot_line_object(line, ax=None, color='C3', length=10.0, label='Line',
                     center=(0, 0, 0)):
    """Plot a Line algebra element as a line segment in 3D."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    refs = (center,
            (center[0] + 1, center[1] + 1, center[2] + 1),
            (center[0] + 1, center[1] - 1, center[2] + 2))
    seg = line_to_segment(line, length=length, refs=refs)
    if seg is None:
        return ax
    a, b = seg
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
            color=color, linewidth=2, label=label)
    return ax


def plane_to_quad(plane, center=(0, 0, 0), size=8.0):
    """Return 4 corner points (list of np.ndarray) of a quad on the plane."""
    val = plane.value
    # PGA 3,0,1 plane: index 1 = d, 2 = a, 3 = b, 4 = c
    a, b, c = val[2], val[3], val[4]
    normal = np.array([a, b, c], dtype=float)
    nn = np.linalg.norm(normal)
    if nn < 1e-12:
        return None
    normal = normal / nn
    # foot of perpendicular from `center` onto the plane.
    try:
        p_on = Point(*center).project_onto(plane)
    except ZeroDivisionError:
        return None
    p0 = _point_xyz(p_on)
    if not np.all(np.isfinite(p0)):
        return None
    # build an orthonormal in-plane basis
    helper = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return [
        p0 + size * u + size * v,
        p0 - size * u + size * v,
        p0 - size * u - size * v,
        p0 + size * u - size * v,
    ]


def plot_plane_object(plane, ax=None, color='cyan', alpha=0.25,
                      center=(0, 0, 0), size=8.0, label='Plane'):
    """Plot a Plane algebra element as a finite quad in 3D."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    quad = plane_to_quad(plane, center=center, size=size)
    if quad is None:
        return ax
    pts = np.array(quad)
    poly = Poly3DCollection([pts], alpha=alpha, facecolor=color,
                            edgecolor=color, label=label)
    ax.add_collection3d(poly)
    return ax


# ---------------------------------------------------------------------------
# Meet (^ outer product) and Join (& regressive product) example plots
# ---------------------------------------------------------------------------

def plot_join_two_points(ax=None):
    """Join of two points (p1 & p2) is the line through them."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    p1 = Point(0, 0, 0)
    p2 = Point(4, 3, 2)
    line = p1 & p2  # join
    plot_points([p1, p2], ax=ax, color='C0')
    plot_line_object(line, ax=ax, color='C3', length=6.0,
                     label='Join: p1 & p2', center=(2, 1.5, 1))
    ax.legend()
    ax.set_title('Join of two points -> line')
    return ax


def plot_join_three_points(ax=None):
    """Join of three points (p1 & p2 & p3) is the plane through them."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    p1 = Point(3, 0, 0)
    p2 = Point(0, 3, 0)
    p3 = Point(0, 0, 3)
    plane = p1 & p2 & p3  # join
    plot_points([p1, p2, p3], ax=ax, color='C0')
    plot_lines([(p1, p2), (p2, p3), (p3, p1)], ax=ax, color='C0')
    plot_plane_object(plane, ax=ax, color='C3', alpha=0.3,
                      center=(1, 1, 1), size=4.0,
                      label='Join: p1 & p2 & p3')
    ax.legend()
    ax.set_title('Join of three points -> plane')
    return ax


def plot_join_line_and_point(ax=None):
    """Join of a line and a point (line & p) is the plane containing both."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    a = Point(0, 0, 0)
    b = Point(4, 0, 0)
    line = a & b
    p = Point(2, 4, 3)
    plane = line & p  # join
    plot_points([a, b, p], ax=ax, color='C0')
    plot_line_object(line, ax=ax, color='C2', length=4.0,
                     label='input line', center=(2, 0, 0))
    plot_plane_object(plane, ax=ax, color='C3', alpha=0.3,
                      center=(2, 1.5, 1), size=4.0,
                      label='Join: line & p')
    ax.legend()
    ax.set_title('Join of line and point -> plane')
    return ax


def plot_meet_two_planes(ax=None):
    """Meet of two planes (pi1 ^ pi2) is the line of intersection."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi1 = Plane.from_abcd(1, 0, 0, 0)   # x = 0 (yz-plane)
    pi2 = Plane.from_abcd(0, 1, 0, 0)   # y = 0 (xz-plane)
    line = pi1 ^ pi2  # meet
    plot_plane_object(pi1, ax=ax, color='C0', alpha=0.2,
                      center=(0, 0, 0), size=4.0, label='plane 1')
    plot_plane_object(pi2, ax=ax, color='C1', alpha=0.2,
                      center=(0, 0, 0), size=4.0, label='plane 2')
    plot_line_object(line, ax=ax, color='C3', length=5.0,
                     label='Meet: pi1 ^ pi2', center=(0, 0, 0))
    ax.legend()
    ax.set_title('Meet of two planes -> line')
    return ax


def plot_meet_three_planes(ax=None):
    """Meet of three planes (pi1 ^ pi2 ^ pi3) is their common point."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi1 = Plane.from_abcd(1, 0, 0, -2)  # x = 2
    pi2 = Plane.from_abcd(0, 1, 0, -3)  # y = 3
    pi3 = Plane.from_abcd(0, 0, 1, -1)  # z = 1
    p = pi1 ^ pi2 ^ pi3  # meet -> Algebra encoding a point
    # Convert algebra-point to a Point with x,y,z
    z = p.value[11]
    y = p.value[12]
    x = p.value[13]
    w = p.value[14]
    pt = Point(x / w, y / w, z / w) if abs(w) > 1e-12 else None
    plot_plane_object(pi1, ax=ax, color='C0', alpha=0.2,
                      center=(2, 3, 1), size=3.0, label='plane 1 (x=2)')
    plot_plane_object(pi2, ax=ax, color='C1', alpha=0.2,
                      center=(2, 3, 1), size=3.0, label='plane 2 (y=3)')
    plot_plane_object(pi3, ax=ax, color='C2', alpha=0.2,
                      center=(2, 3, 1), size=3.0, label='plane 3 (z=1)')
    if pt is not None:
        plot_points([pt], ax=ax, color='C3',
                    label_prefix='Meet: pi1^pi2^pi3 = ')
    ax.legend()
    ax.set_title('Meet of three planes -> point')
    return ax


def plot_meet_plane_and_line(ax=None):
    """Meet of a plane and a line (pi ^ line) is their intersection point."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi = Plane.from_abcd(0, 0, 1, -2)  # z = 2
    a = Point(-1, -1, 0)
    b = Point(3, 4, 5)
    line = a & b
    p = pi ^ line  # meet -> Algebra encoding a point
    z = p.value[11]
    y = p.value[12]
    x = p.value[13]
    w = p.value[14]
    pt = Point(x / w, y / w, z / w) if abs(w) > 1e-12 else None
    plot_plane_object(pi, ax=ax, color='C0', alpha=0.2,
                      center=(1, 1, 2), size=4.0, label='plane (z=2)')
    plot_line_object(line, ax=ax, color='C2', length=5.0,
                     label='input line', center=(1, 1, 2))
    if pt is not None:
        plot_points([pt], ax=ax, color='C3',
                    label_prefix='Meet: pi ^ line = ')
    ax.legend()
    ax.set_title('Meet of plane and line -> point')
    return ax


def plot_meet_and_join_examples():
    """Show all meet/join visualizations."""
    plot_join_two_points()
    plt.show()
    plot_join_three_points()
    plt.show()
    plot_join_line_and_point()
    plt.show()
    plot_meet_two_planes()
    plt.show()
    plot_meet_three_planes()
    plt.show()
    plot_meet_plane_and_line()
    plt.show()

def plot_point_creation():
    p = Point(1, 2, 3)
    ax = plot_points([p])
    ax.legend()
    plt.title("Single Point")
    plt.show()

def plot_line_from_points():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 1, 1)
    ax = plot_points([p1, p2])
    plot_lines([(p1, p2)], ax=ax)
    ax.legend()
    plt.title("Line from Two Points")
    plt.show()

def plot_plane_from_points():
    p1 = Point(0, 0, 0)
    p2 = Point(1, 0, 0)
    p3 = Point(0, 1, 0)
    ax = plot_points([p1, p2, p3])
    plot_plane([p1, p2, p3], ax=ax)
    ax.legend()
    plt.title("Plane from Three Points")
    plt.show()

def plot_translator_and_rotor():
    p = Point(1, 0, 0)
    t = Translator.from_xyz(1, 2, 3)
    r = Rotor.from_angle_and_line(math.pi/2, Line.from_xyz(0, 0, 1))
    p_t = t.project(p)
    p_r = r.project(p)
    ax = plot_points([p, p_t, p_r], color='C0')
    ax.legend()
    plt.title("Point, Translated, and Rotated")
    plt.show()

def plot_transformations():
    p = Point(1, 0, 0)
    t = Translator.from_xyz(1, 0, 0)
    r = Rotor.from_angle_and_line(math.pi/2, Line.from_xyz(0, 0, 1))
    m = t * r
    p_t = t.project(p)
    p_tr = m.project(p)
    ax = plot_points([p, p_t, p_tr], color='C1')
    ax.legend()
    plt.title("Point, Translated, and Translated+Rotated")
    plt.show()

def plot_projection_onto_line_and_plane():
    p1 = Point(2, 3, 4)
    p2 = Point(20, 3, 7)
    p3 = Point(9, 12, 17)
    line = Line.from_points(p3, p1)
    plane = Plane.from_points(p1, p2, p3)
    p_proj = p1.project_onto(line)
    plane_proj = plane.project_onto(Point(0, 0, 0))
    ax = plot_points([p1, p_proj], color='C2')
    plot_lines([(p3, p1)], ax=ax)
    plot_plane([p1, p2, p3], ax=ax)
    ax.legend()
    plt.title("Projection of Point onto Line and Plane")
    plt.show()

def plot_plot_objects():
    p0 = Point(0, 0, 0)
    p1 = Point(2, 3, 4)
    p2 = Point(20, 3, 7)
    p3 = Point(9, 12, 17)
    ax = plot_points([p0, p1, p2, p3])
    plot_lines([(p1, p2), (p2, p3), (p3, p1)], ax=ax)
    plot_plane([p1, p2, p3], ax=ax)
    ax.legend()
    plt.title("PGA3D Example Objects")
    plt.show()

# You can call these plot functions after each test or in __main__ for visual inspection.
if __name__ == '__main__':
    plot_point_creation()
    plot_line_from_points()
    plot_plane_from_points()
    plot_translator_and_rotor()
    plot_transformations()
    plot_projection_onto_line_and_plane()
    plot_plot_objects()
    plot_meet_and_join_examples()
    test_plot_points_lines_planes()