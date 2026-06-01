
import math
import pytest
from pga3d import Point, Line, Plane, Translator, Rotor
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
# test_examples.py

import matplotlib.pyplot as plt

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
    test_plot_points_lines_planes()