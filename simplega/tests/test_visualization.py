"""
Tests for BivectorVisualizer.

All tests use the 'Agg' backend so no display is required.
"""

import math
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt

import simplega.ga30 as GA30
from simplega import project, norm, adjoint, bivector_exp
from simplega.bivector_visualization import (
    BivectorVisualizer,
    _bivector_normal,
    _plane_frame,
    _disk_verts,
    _arc_pts,
)

e1, e2, e3 = GA30.basis


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestBivectorNormal:
    """_bivector_normal maps GA30 bivectors to their dual (normal) vectors."""

    def test_e1e2_normal_is_plus_z(self):
        n = _bivector_normal(e1 * e2)
        np.testing.assert_allclose(n, [0, 0, 1], atol=1e-12)

    def test_e2e3_normal_is_plus_x(self):
        n = _bivector_normal(e2 * e3)
        np.testing.assert_allclose(n, [1, 0, 0], atol=1e-12)

    def test_e3e1_normal_is_plus_y(self):
        n = _bivector_normal(e3 * e1)
        np.testing.assert_allclose(n, [0, 1, 0], atol=1e-12)

    def test_adjoint_flips_normal(self):
        B = e1 * e2
        n_B    = _bivector_normal(B)
        n_Badj = _bivector_normal(adjoint(B))
        np.testing.assert_allclose(n_Badj, -n_B, atol=1e-12)

    def test_linear_combination(self):
        B = e1 * e2 + e2 * e3  # Even(0, -1, 0, -1)
        n = _bivector_normal(B)
        # e2e3 coeff=1 -> e1=1; e1e2 coeff=1 -> e3=1
        np.testing.assert_allclose(n, [1, 0, 1], atol=1e-12)

    def test_scaled_bivector(self):
        B = 3.0 * e1 * e2
        n = _bivector_normal(B)
        np.testing.assert_allclose(n, [0, 0, 3], atol=1e-12)


class TestPlaneFrame:
    """_plane_frame builds an orthonormal frame satisfying u × v = n̂."""

    @pytest.mark.parametrize("normal", [
        [0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0],
        [1, 1, 0], [1, 1, 1],
    ])
    def test_right_handed(self, normal):
        n = np.array(normal, dtype=float)
        u, v = _plane_frame(n)
        cross = np.cross(u, v)
        n_hat = n / np.linalg.norm(n)
        np.testing.assert_allclose(cross, n_hat, atol=1e-12,
            err_msg=f"u × v should equal n̂ for normal={normal}")

    @pytest.mark.parametrize("normal", [
        [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1],
    ])
    def test_orthonormal(self, normal):
        n = np.array(normal, dtype=float)
        u, v = _plane_frame(n)
        assert math.isclose(np.linalg.norm(u), 1.0, rel_tol=1e-12), "u not unit"
        assert math.isclose(np.linalg.norm(v), 1.0, rel_tol=1e-12), "v not unit"
        assert abs(np.dot(u, v)) < 1e-12, "u and v not orthogonal"

    @pytest.mark.parametrize("normal", [
        [0, 0, 1], [1, 0, 0], [0, 1, 0],
    ])
    def test_orthogonal_to_normal(self, normal):
        n = np.array(normal, dtype=float)
        n_hat = n / np.linalg.norm(n)
        u, v  = _plane_frame(n)
        assert abs(np.dot(u, n_hat)) < 1e-12, "u not perp to normal"
        assert abs(np.dot(v, n_hat)) < 1e-12, "v not perp to normal"

    def test_zero_normal_returns_xy_basis(self):
        u, v = _plane_frame(np.zeros(3))
        np.testing.assert_allclose(u, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(v, [0, 1, 0], atol=1e-12)


class TestDiskVerts:
    def test_shape(self):
        c = np.zeros(3)
        u, v = np.array([1.,0.,0.]), np.array([0.,1.,0.])
        verts = _disk_verts(c, u, v, radius=2.0, n_pts=32)
        assert verts.shape == (32, 3)

    def test_radius(self):
        c = np.zeros(3)
        u, v = np.array([1.,0.,0.]), np.array([0.,1.,0.])
        r = 1.5
        verts = _disk_verts(c, u, v, r, n_pts=64)
        dists = np.linalg.norm(verts - c, axis=1)
        np.testing.assert_allclose(dists, r, atol=1e-12)

    def test_center_offset(self):
        c = np.array([1., 2., 3.])
        u, v = np.array([1.,0.,0.]), np.array([0.,1.,0.])
        verts = _disk_verts(c, u, v, 1.0, n_pts=16)
        dists = np.linalg.norm(verts - c, axis=1)
        np.testing.assert_allclose(dists, 1.0, atol=1e-12)

    def test_lies_in_uv_plane(self):
        c = np.zeros(3)
        n = np.array([0., 0., 1.])
        u, v = _plane_frame(n)
        verts = _disk_verts(c, u, v, 1.0, n_pts=32)
        # z-component should be zero (in the xy-plane)
        np.testing.assert_allclose(verts[:, 2], 0.0, atol=1e-12)


class TestArcPts:
    def test_shape(self):
        c = np.zeros(3)
        u, v = np.array([1.,0.,0.]), np.array([0.,1.,0.])
        pts = _arc_pts(c, u, v, radius=1.0, n_pts=48)
        assert pts.shape == (48, 3)

    def test_radius_approx(self):
        c = np.zeros(3)
        u, v = np.array([1.,0.,0.]), np.array([0.,1.,0.])
        pts = _arc_pts(c, u, v, radius=1.0, n_pts=48)
        dists = np.linalg.norm(pts - c, axis=1)
        # arc is at radius * 1.10
        np.testing.assert_allclose(dists, 1.10, atol=1e-12)

    def test_sign_reversal_different_pts(self):
        c = np.zeros(3)
        u, v = np.array([1.,0.,0.]), np.array([0.,1.,0.])
        pts_pos = _arc_pts(c, u, v, 1.0, sign=+1)
        pts_neg = _arc_pts(c, u, v, 1.0, sign=-1)
        # The two arcs should differ (opposite orientation means different path)
        assert not np.allclose(pts_pos, pts_neg)


# ---------------------------------------------------------------------------
# BivectorVisualizer construction and methods
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_figures():
    """Ensure all figures are closed after each test."""
    yield
    plt.close("all")


class TestBivectorVisualizerConstruction:
    def test_creates_figure(self):
        viz = BivectorVisualizer()
        assert viz.fig is not None
        assert viz.ax is not None

    def test_custom_figsize(self):
        viz = BivectorVisualizer(figsize=(6, 5))
        w, h = viz.fig.get_size_inches()
        assert math.isclose(w, 6.0) and math.isclose(h, 5.0)

    def test_is_3d_axes(self):
        viz = BivectorVisualizer()
        assert hasattr(viz.ax, "get_zlim"), "Axes is not 3D"


class TestAddBivector:
    def test_add_e1e2(self):
        viz = BivectorVisualizer()
        result = viz.add_bivector(e1 * e2)
        assert result is viz, "add_bivector should return self for chaining"

    def test_add_with_label(self):
        viz = BivectorVisualizer()
        viz.add_bivector(e1 * e2, label="e1e2")
        _, labels = viz.ax.get_legend_handles_labels()
        assert "e1e2" in labels

    def test_add_with_adjoint(self):
        viz = BivectorVisualizer()
        viz.add_bivector(e1 * e2, label="B", show_adjoint=True, adjoint_label="B†")
        _, labels = viz.ax.get_legend_handles_labels()
        assert "B" in labels
        assert "B†" in labels

    def test_chaining(self):
        viz = BivectorVisualizer()
        result = (viz
                  .add_bivector(e1 * e2, label="B12")
                  .add_bivector(e2 * e3, label="B23"))
        assert result is viz

    def test_zero_bivector_no_crash(self):
        viz = BivectorVisualizer()
        viz.add_bivector(GA30.Even.zero())   # should not raise

    def test_grade_extraction(self):
        """add_bivector should use only the grade-2 part."""
        viz = BivectorVisualizer()
        B_with_scalar = 5.0 + e1 * e2   # Even with scalar + bivector
        viz.add_bivector(B_with_scalar)   # should not raise


class TestAddVector:
    def test_add_vector_e1(self):
        viz = BivectorVisualizer()
        result = viz.add_vector(e1, label="e1")
        assert result is viz

    def test_add_zero_vector_no_crash(self):
        viz = BivectorVisualizer()
        viz.add_vector(GA30.Odd.zero())

    def test_add_grade1_extraction(self):
        v = e1 + 2.0 * e2 + 3.0 * GA30.I3   # grade-1 + grade-3
        viz = BivectorVisualizer()
        viz.add_vector(v)   # should not raise


class TestAddVectorPair:
    def test_vector_pair(self):
        viz = BivectorVisualizer()
        result = viz.add_vector_pair(
            e1, e2,
            label_v1="e1", label_v2="e2", label_bv="e1∧e2"
        )
        assert result is viz

    def test_vector_pair_with_adjoint(self):
        viz = BivectorVisualizer()
        viz.add_vector_pair(e1, e2, show_adjoint=True)  # no crash


class TestAddBasisBivectors:
    def test_adds_three_bivectors(self):
        viz = BivectorVisualizer()
        result = viz.add_basis_bivectors()
        assert result is viz
        _, labels = viz.ax.get_legend_handles_labels()
        assert len(labels) == 3

    def test_with_adjoints(self):
        viz = BivectorVisualizer()
        viz.add_basis_bivectors(show_adjoints=True)
        _, labels = viz.ax.get_legend_handles_labels()
        assert len(labels) == 6   # 3 bivectors + 3 adjoints


class TestAddRotationSweep:
    def test_sweep_e1e2_on_e3(self):
        viz = BivectorVisualizer()
        B = (math.pi / 2) * e1 * e2  # 90° rotation in xy-plane
        result = viz.add_rotation_sweep(B, e3, n_steps=6, label="v(t)")
        assert result is viz

    def test_rotation_end_point(self):
        """
        π/2 CCW rotation of e1 in the e1e2 plane should yield e2.

        In GA the sandwich formula  R * v * R†  with  R = bivector_exp(θ·B̂)
        produces a rotation by angle 2θ, so to get 90° we need θ = π/4.
        The sign convention bivector_exp(-(π/4)·e1e2) gives CCW rotation
        (e1 → e2 direction).
        """
        B = -(math.pi / 4) * e1 * e2   # half-angle with CCW sign
        R = bivector_exp(B)
        rotated = R * e1 * adjoint(R)
        g1 = project(rotated, 1)
        np.testing.assert_allclose(
            [g1.x, g1.y, g1.z], [0.0, 1.0, 0.0], atol=1e-10,
            err_msg="π/2 CCW rotation of e1: bivector_exp(-(π/4)·e1e2)*e1*R† should give e2"
        )


class TestAddGradeDecomposition:
    def test_no_crash(self):
        viz = BivectorVisualizer()
        B = 2.0 + 0.7 * e1 * e2 + 0.4 * e2 * e3
        viz.add_grade_decomposition(B)   # should not raise


class TestSceneHelpers:
    def test_draw_axes(self):
        viz = BivectorVisualizer()
        result = viz.draw_axes()
        assert result is viz

    def test_set_limits(self):
        viz = BivectorVisualizer()
        result = viz.set_limits(2.0)
        assert result is viz
        lo, hi = viz.ax.get_xlim()
        assert math.isclose(lo, -2.0) and math.isclose(hi, 2.0)

    def test_set_view(self):
        viz = BivectorVisualizer()
        viz.set_view(30, 45)   # should not raise

    def test_legend_no_crash_empty(self):
        viz = BivectorVisualizer()
        viz.legend()   # no labels yet — should not raise

    def test_legend_with_labels(self):
        viz = BivectorVisualizer()
        viz.add_bivector(e1 * e2, label="B")
        viz.legend()   # should not raise


class TestSaveAndClose:
    def test_save_png(self, tmp_path):
        viz = BivectorVisualizer()
        viz.draw_axes()
        viz.add_bivector(e1 * e2, label="B")
        path = str(tmp_path / "test.png")
        viz.save(path)
        import os
        assert os.path.exists(path)

    def test_close_no_error(self):
        viz = BivectorVisualizer()
        viz.close()   # should not raise


# ---------------------------------------------------------------------------
# Geometric correctness round-trips
# ---------------------------------------------------------------------------

class TestGeometricCorrectness:
    def test_normal_perpendicular_to_disk_verts(self):
        """Disk vertices must all be orthogonal to the normal."""
        B = e1 * e2 + 0.5 * e2 * e3
        n = _bivector_normal(B)
        u, v = _plane_frame(n)
        verts = _disk_verts(np.zeros(3), u, v, radius=1.0)
        n_hat = n / np.linalg.norm(n)
        dots = verts @ n_hat
        np.testing.assert_allclose(dots, 0.0, atol=1e-12,
            err_msg="Disk vertices should be perpendicular to the normal")

    def test_adjoint_normal_is_negation(self):
        """The adjoint normal should be exactly the negation of the original."""
        for B in [e1 * e2, e2 * e3, e3 * e1, e1 * e2 + e3 * e1]:
            n  = _bivector_normal(B)
            na = _bivector_normal(adjoint(B))
            np.testing.assert_allclose(na, -n, atol=1e-12)

    def test_rotation_preserves_norm(self):
        """bivector_exp should generate norm-preserving rotations."""
        B = (math.pi / 3) * (e1 * e2 + 0.3 * e2 * e3)
        v = 1.2 * e1 + 0.5 * e2 - 0.8 * e3
        R    = bivector_exp(B)
        Radj = adjoint(R)
        rv   = R * v * Radj
        assert math.isclose(norm(v), norm(rv), rel_tol=1e-10), (
            f"Rotation changed norm: {norm(v):.6f} → {norm(rv):.6f}"
        )

    def test_orientation_arc_sign_relationship(self):
        """sign=+1 and sign=-1 arcs should trace in opposite angular directions."""
        c = np.zeros(3)
        u = np.array([1., 0., 0.])
        v = np.array([0., 1., 0.])
        pos = _arc_pts(c, u, v, 1.0, sign=+1, n_pts=64)
        neg = _arc_pts(c, u, v, 1.0, sign=-1, n_pts=64)
        # The angular displacement of the positive arc should oppose the negative
        angle_pos = np.arctan2((pos[-1] - c) @ v, (pos[-1] - c) @ u)
        angle_neg = np.arctan2((neg[-1] - c) @ v, (neg[-1] - c) @ u)
        # One arc goes clockwise, the other counter-clockwise: they diverge
        assert not math.isclose(angle_pos, angle_neg, abs_tol=0.1), (
            "Positive and negative arcs should end at different angular positions"
        )
