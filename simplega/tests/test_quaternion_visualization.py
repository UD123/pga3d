"""
Tests for QuaternionVisualizer and its geometry helpers.

All tests use the 'Agg' backend so no display is required.
"""

import math
import numpy as np
import pytest
import sys, os
path_to_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, path_to_root)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")   # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt

from simplega.quaternions import Quaternion
from simplega.quaternion_visualization import (
    QuaternionVisualizer,
    axis_angle,
    rotate_vec,
    slerp,
    _plane_frame,
    _unit,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _from_axis_angle(axis, angle_deg: float) -> Quaternion:
    """Build a unit quaternion from an axis (array-like) and angle in degrees."""
    angle = math.radians(angle_deg)
    ax = np.asarray(axis, dtype=float)
    ax /= np.linalg.norm(ax)
    s = math.sin(angle / 2)
    return Quaternion(math.cos(angle / 2), s * ax[0], s * ax[1], s * ax[2])


def _q_close(q1: Quaternion, q2: Quaternion, atol: float = 1e-10) -> bool:
    """True when q1 ≈ q2 or q1 ≈ -q2 (double-cover equivalence)."""
    d  = abs(q1.w - q2.w) + abs(q1.x - q2.x) + abs(q1.y - q2.y) + abs(q1.z - q2.z)
    dn = abs(q1.w + q2.w) + abs(q1.x + q2.x) + abs(q1.y + q2.y) + abs(q1.z + q2.z)
    return d < atol or dn < atol


# ---------------------------------------------------------------------------
# _unit
# ---------------------------------------------------------------------------

class TestUnit:
    def test_unit_is_unit(self):
        q = Quaternion(2, 1, 1, 1)
        u = _unit(q)
        assert math.isclose(u.norm(), 1.0, rel_tol=1e-12)

    def test_already_unit_unchanged(self):
        q = _from_axis_angle([0, 0, 1], 45)
        u = _unit(q)
        assert math.isclose(u.norm(), 1.0, rel_tol=1e-12)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            _unit(Quaternion(0, 0, 0, 0))


# ---------------------------------------------------------------------------
# axis_angle
# ---------------------------------------------------------------------------

class TestAxisAngle:
    def test_identity_gives_zero_angle(self):
        q = Quaternion.one()
        _, angle = axis_angle(q)
        assert math.isclose(angle, 0.0, abs_tol=1e-10)

    def test_90_deg_z(self):
        q = _from_axis_angle([0, 0, 1], 90)
        ax, angle = axis_angle(q)
        assert math.isclose(angle, math.pi / 2, rel_tol=1e-10)
        np.testing.assert_allclose(ax, [0, 0, 1], atol=1e-10)

    def test_180_deg_x(self):
        q = _from_axis_angle([1, 0, 0], 180)
        ax, angle = axis_angle(q)
        assert math.isclose(angle, math.pi, rel_tol=1e-10)
        np.testing.assert_allclose(np.abs(ax), [1, 0, 0], atol=1e-10)

    def test_60_deg_diagonal(self):
        axis_in = np.array([1., 1., 0.]) / math.sqrt(2)
        q = _from_axis_angle(axis_in, 60)
        ax, angle = axis_angle(q)
        assert math.isclose(angle, math.pi / 3, rel_tol=1e-10)
        np.testing.assert_allclose(ax, axis_in, atol=1e-10)

    def test_returns_unit_axis(self):
        q = _from_axis_angle([1, 2, 3], 75)
        ax, _ = axis_angle(q)
        assert math.isclose(np.linalg.norm(ax), 1.0, rel_tol=1e-10)

    def test_negative_w_same_rotation(self):
        """q and -q must give the same axis and angle."""
        q = _from_axis_angle([0, 1, 0], 120)
        qn = Quaternion(-q.w, -q.x, -q.y, -q.z)
        ax1, ang1 = axis_angle(q)
        ax2, ang2 = axis_angle(qn)
        assert math.isclose(ang1, ang2, rel_tol=1e-10)
        np.testing.assert_allclose(ax1, ax2, atol=1e-10)

    def test_angle_in_0_to_2pi(self):
        for deg in [0, 30, 90, 135, 180]:
            q = _from_axis_angle([0, 0, 1], deg)
            _, angle = axis_angle(q)
            assert 0.0 <= angle <= 2 * math.pi + 1e-10


# ---------------------------------------------------------------------------
# rotate_vec
# ---------------------------------------------------------------------------

class TestRotateVec:
    def test_identity_no_rotation(self):
        v = np.array([1., 2., 3.])
        q = Quaternion.one()
        rv = rotate_vec(q, v)
        np.testing.assert_allclose(rv, v, atol=1e-12)

    def test_90_deg_z_rotates_x_to_y(self):
        q = _from_axis_angle([0, 0, 1], 90)
        v = np.array([1., 0., 0.])
        rv = rotate_vec(q, v)
        np.testing.assert_allclose(rv, [0, 1, 0], atol=1e-10)

    def test_90_deg_z_rotates_y_to_minus_x(self):
        q = _from_axis_angle([0, 0, 1], 90)
        v = np.array([0., 1., 0.])
        rv = rotate_vec(q, v)
        np.testing.assert_allclose(rv, [-1, 0, 0], atol=1e-10)

    def test_180_deg_x_flips_y(self):
        q = _from_axis_angle([1, 0, 0], 180)
        v = np.array([0., 1., 0.])
        rv = rotate_vec(q, v)
        np.testing.assert_allclose(rv, [0, -1, 0], atol=1e-10)

    def test_rotation_axis_is_fixed(self):
        """Rotating the axis vector should return the same axis vector."""
        axis = np.array([0., 0., 1.])
        q = _from_axis_angle(axis, 90)
        rv = rotate_vec(q, axis)
        np.testing.assert_allclose(rv, axis, atol=1e-10)

    def test_preserves_norm(self):
        q = _from_axis_angle([1, 1, 1], 73)
        v = np.array([1.5, -2.3, 0.7])
        rv = rotate_vec(q, v)
        assert math.isclose(np.linalg.norm(rv), np.linalg.norm(v), rel_tol=1e-10)

    def test_preserves_dot_product(self):
        """Rotation should preserve the dot product between two vectors."""
        q = _from_axis_angle([0, 1, 0], 55)
        v1 = np.array([1., 0., 0.])
        v2 = np.array([0., 1., 0.])
        rv1 = rotate_vec(q, v1)
        rv2 = rotate_vec(q, v2)
        assert math.isclose(np.dot(v1, v2), np.dot(rv1, rv2), abs_tol=1e-10)

    def test_360_deg_identity(self):
        """A 360° rotation should return the original vector."""
        q = _from_axis_angle([0, 0, 1], 360)
        v = np.array([1., 2., 3.])
        rv = rotate_vec(q, v)
        np.testing.assert_allclose(rv, v, atol=1e-10)


# ---------------------------------------------------------------------------
# slerp
# ---------------------------------------------------------------------------

class TestSlerp:
    def test_t0_returns_q1(self):
        q1 = _from_axis_angle([0, 0, 1], 45)
        q2 = _from_axis_angle([1, 0, 0], 90)
        result = slerp(q1, q2, 0.0)
        assert _q_close(result, q1)

    def test_t1_returns_q2(self):
        q1 = _from_axis_angle([0, 0, 1], 45)
        q2 = _from_axis_angle([1, 0, 0], 90)
        result = slerp(q1, q2, 1.0)
        assert _q_close(result, q2)

    def test_t_half_is_unit(self):
        q1 = _from_axis_angle([0, 0, 1], 45)
        q2 = _from_axis_angle([1, 0, 0], 90)
        result = slerp(q1, q2, 0.5)
        assert math.isclose(result.norm(), 1.0, rel_tol=1e-10)

    def test_same_quaternion_gives_self(self):
        q = _from_axis_angle([0, 1, 0], 60)
        result = slerp(q, q, 0.5)
        assert _q_close(result, q)

    def test_result_is_unit(self):
        q1 = _from_axis_angle([1, 0, 0], 30)
        q2 = _from_axis_angle([0, 1, 1], 150)
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r = slerp(q1, q2, t)
            assert math.isclose(r.norm(), 1.0, rel_tol=1e-10), f"not unit at t={t}"

    def test_shorter_arc(self):
        """slerp should take the shorter arc even when q2 needs negating."""
        q1 = Quaternion.one()
        q2 = _from_axis_angle([0, 0, 1], 350)   # 350° ≡ −10°; shorter arc is 10°
        result = slerp(q1, q2, 0.5)
        # The half-way point should have a small rotation angle
        _, half_angle = axis_angle(result)
        assert half_angle < math.pi / 2, (
            f"slerp did not take the shorter arc; half-angle = {math.degrees(half_angle):.1f}°"
        )

    def test_monotone_along_path(self):
        """The angle from q1 should increase monotonically as t → 1."""
        q1 = _from_axis_angle([0, 0, 1], 0)
        q2 = _from_axis_angle([0, 0, 1], 90)
        angles = []
        for k in range(11):
            qi = slerp(q1, q2, k / 10)
            ax, ang = axis_angle(qi)
            angles.append(ang)
        for i in range(len(angles) - 1):
            assert angles[i] <= angles[i+1] + 1e-10, (
                f"angle not monotone at step {i}: {angles[i]:.4f} > {angles[i+1]:.4f}"
            )


# ---------------------------------------------------------------------------
# _plane_frame
# ---------------------------------------------------------------------------

class TestPlaneFrame:
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
        u, v = _plane_frame(n)
        assert abs(np.dot(u, n_hat)) < 1e-12, "u not perp to normal"
        assert abs(np.dot(v, n_hat)) < 1e-12, "v not perp to normal"


# ---------------------------------------------------------------------------
# Fixture: close all figures after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# QuaternionVisualizer construction
# ---------------------------------------------------------------------------

class TestQuaternionVisualizerConstruction:
    def test_creates_figure(self):
        viz = QuaternionVisualizer()
        assert viz.fig is not None
        assert viz.ax is not None

    def test_custom_figsize(self):
        viz = QuaternionVisualizer(figsize=(6, 5))
        w, h = viz.fig.get_size_inches()
        assert math.isclose(w, 6.0) and math.isclose(h, 5.0)

    def test_is_3d_axes(self):
        viz = QuaternionVisualizer()
        assert hasattr(viz.ax, "get_zlim"), "Axes is not 3D"

    def test_custom_title(self):
        viz = QuaternionVisualizer(title="My Title")
        assert viz.ax.get_title() == "My Title"


# ---------------------------------------------------------------------------
# add_quaternion
# ---------------------------------------------------------------------------

class TestAddQuaternion:
    def test_basic_no_crash(self):
        viz = QuaternionVisualizer()
        result = viz.add_quaternion(_from_axis_angle([0, 0, 1], 90))
        assert result is viz

    def test_chaining(self):
        viz = QuaternionVisualizer()
        q1 = _from_axis_angle([0, 0, 1], 60)
        q2 = _from_axis_angle([1, 0, 0], 45)
        result = (viz
                  .add_quaternion(q1, color="steelblue", label="q1")
                  .add_quaternion(q2, color="tomato",    label="q2"))
        assert result is viz

    def test_label_appears_in_legend(self):
        viz = QuaternionVisualizer()
        viz.add_quaternion(_from_axis_angle([0, 0, 1], 90), label="q90z")
        _, labels = viz.ax.get_legend_handles_labels()
        assert "q90z" in labels

    def test_show_conjugate(self):
        viz = QuaternionVisualizer()
        result = viz.add_quaternion(
            _from_axis_angle([0, 0, 1], 90),
            label="q", show_conjugate=True, conjugate_label="q†"
        )
        assert result is viz
        _, labels = viz.ax.get_legend_handles_labels()
        assert "q" in labels
        assert "q†" in labels

    def test_no_frame(self):
        viz = QuaternionVisualizer()
        result = viz.add_quaternion(
            _from_axis_angle([1, 1, 0], 45), show_frame=False
        )
        assert result is viz

    def test_no_sector(self):
        viz = QuaternionVisualizer()
        result = viz.add_quaternion(
            _from_axis_angle([0, 1, 0], 120), show_sector=False
        )
        assert result is viz

    def test_identity_no_crash(self):
        viz = QuaternionVisualizer()
        viz.add_quaternion(Quaternion.one())   # near-zero angle — should not raise

    def test_180_deg_no_crash(self):
        viz = QuaternionVisualizer()
        viz.add_quaternion(_from_axis_angle([1, 0, 0], 180))


# ---------------------------------------------------------------------------
# add_rotation_effect
# ---------------------------------------------------------------------------

class TestAddRotationEffect:
    def test_basic_no_crash(self):
        q = _from_axis_angle([0, 0, 1], 90)
        viz = QuaternionVisualizer()
        result = viz.add_rotation_effect(q, [np.array([1., 0., 0.])])
        assert result is viz

    def test_multiple_vectors(self):
        q = _from_axis_angle([1, 1, 0], 120)
        vecs = [np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])]
        viz = QuaternionVisualizer()
        result = viz.add_rotation_effect(q, vecs, labels=["R(x)", "R(y)", "R(z)"])
        assert result is viz

    def test_no_arc(self):
        q = _from_axis_angle([0, 0, 1], 45)
        viz = QuaternionVisualizer()
        result = viz.add_rotation_effect(q, [np.array([1., 0., 0.])], show_arc=False)
        assert result is viz

    def test_rotation_effect_correctness(self):
        """x-vector rotated 90° around z should be close to y-vector."""
        q = _from_axis_angle([0, 0, 1], 90)
        rv = rotate_vec(q, np.array([1., 0., 0.]))
        np.testing.assert_allclose(rv, [0., 1., 0.], atol=1e-10)


# ---------------------------------------------------------------------------
# add_slerp
# ---------------------------------------------------------------------------

class TestAddSlerp:
    def test_basic_no_crash(self):
        q1 = _from_axis_angle([0, 0, 1],  45)
        q2 = _from_axis_angle([1, 0, 0], 135)
        viz = QuaternionVisualizer()
        result = viz.add_slerp(q1, q2)
        assert result is viz

    def test_with_reference_vec(self):
        q1 = _from_axis_angle([0, 0, 1], 30)
        q2 = _from_axis_angle([0, 1, 0], 90)
        viz = QuaternionVisualizer()
        result = viz.add_slerp(q1, q2, reference_vec=np.array([0., 0., 1.]))
        assert result is viz

    def test_with_labels(self):
        q1 = _from_axis_angle([0, 0, 1], 30)
        q2 = _from_axis_angle([1, 0, 0], 60)
        viz = QuaternionVisualizer()
        viz.add_slerp(q1, q2, label_start="start", label_end="end")
        _, labels = viz.ax.get_legend_handles_labels()
        assert "start" in labels
        assert "end" in labels

    def test_custom_n_steps(self):
        q1 = _from_axis_angle([0, 0, 1], 10)
        q2 = _from_axis_angle([0, 0, 1], 80)
        viz = QuaternionVisualizer()
        viz.add_slerp(q1, q2, n_steps=5)   # very coarse — should not raise

    def test_identical_quaternions(self):
        q = _from_axis_angle([0, 0, 1], 45)
        viz = QuaternionVisualizer()
        viz.add_slerp(q, q)   # should not raise


# ---------------------------------------------------------------------------
# add_composition
# ---------------------------------------------------------------------------

class TestAddComposition:
    def test_basic_no_crash(self):
        q1 = _from_axis_angle([1, 0, 0],  60)
        q2 = _from_axis_angle([0, 0, 1],  90)
        viz = QuaternionVisualizer()
        result = viz.add_composition(q1, q2)
        assert result is viz

    def test_with_custom_labels(self):
        q1 = _from_axis_angle([1, 0, 0],  60)
        q2 = _from_axis_angle([0, 0, 1],  90)
        viz = QuaternionVisualizer()
        viz.add_composition(q1, q2, label1="A", label2="B", label_prod="AB")
        _, labels = viz.ax.get_legend_handles_labels()
        assert "A" in labels
        assert "B" in labels
        assert "AB" in labels

    def test_composition_is_product(self):
        """The rotation produced by q1*q2 should equal applying q2 then q1."""
        q1 = _from_axis_angle([0, 0, 1], 90)
        q2 = _from_axis_angle([1, 0, 0], 90)
        v  = np.array([0., 1., 0.])
        # Apply q2 then q1
        v2 = rotate_vec(q2, v)
        v12 = rotate_vec(q1, v2)
        # Apply product q1*q2 directly
        qp = _unit(q1 * q2)
        vp = rotate_vec(qp, v)
        np.testing.assert_allclose(v12, vp, atol=1e-10,
            err_msg="Composition q1*q2 should equal applying q2 then q1")


# ---------------------------------------------------------------------------
# add_unit_sphere
# ---------------------------------------------------------------------------

class TestAddUnitSphere:
    def test_no_crash(self):
        viz = QuaternionVisualizer()
        result = viz.add_unit_sphere()
        assert result is viz

    def test_custom_alpha(self):
        viz = QuaternionVisualizer()
        result = viz.add_unit_sphere(alpha=0.1, n=12)
        assert result is viz


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

class TestSceneHelpers:
    def test_draw_axes(self):
        viz = QuaternionVisualizer()
        result = viz.draw_axes()
        assert result is viz

    def test_set_limits(self):
        viz = QuaternionVisualizer()
        result = viz.set_limits(2.0)
        assert result is viz
        lo, hi = viz.ax.get_xlim()
        assert math.isclose(lo, -2.0) and math.isclose(hi, 2.0)

    def test_set_view(self):
        viz = QuaternionVisualizer()
        viz.set_view(30, 45)   # should not raise

    def test_legend_no_crash_empty(self):
        viz = QuaternionVisualizer()
        viz.legend()   # no labels yet — should not raise

    def test_legend_with_labels(self):
        viz = QuaternionVisualizer()
        viz.add_quaternion(_from_axis_angle([0, 0, 1], 90), label="q")
        viz.legend()   # should not raise


# ---------------------------------------------------------------------------
# Save and close
# ---------------------------------------------------------------------------

class TestSaveAndClose:
    def test_save_png(self, tmp_path):
        import os
        viz = QuaternionVisualizer()
        viz.draw_axes()
        viz.add_quaternion(_from_axis_angle([0, 0, 1], 90), label="q")
        path = str(tmp_path / "test.png")
        viz.save(path)
        assert os.path.exists(path)

    def test_close_no_error(self):
        viz = QuaternionVisualizer()
        viz.close()   # should not raise


# ---------------------------------------------------------------------------
# Geometric correctness
# ---------------------------------------------------------------------------

class TestGeometricCorrectness:
    def test_rotate_vec_preserves_norm(self):
        q = _from_axis_angle([1, 2, 3], 73)
        v = np.array([1.5, -2.3, 0.7])
        rv = rotate_vec(q, v)
        assert math.isclose(np.linalg.norm(rv), np.linalg.norm(v), rel_tol=1e-10)

    def test_axis_angle_round_trip(self):
        """axis_angle should recover the original axis and angle."""
        for (ax, deg) in [([0,0,1], 90), ([1,0,0], 60), ([1,1,1], 120)]:
            q = _from_axis_angle(ax, deg)
            ax_out, ang_out = axis_angle(q)
            expected_ax = np.asarray(ax, dtype=float)
            expected_ax /= np.linalg.norm(expected_ax)
            np.testing.assert_allclose(ax_out, expected_ax, atol=1e-10,
                err_msg=f"axis mismatch for axis={ax}, deg={deg}")
            assert math.isclose(ang_out, math.radians(deg), rel_tol=1e-10), (
                f"angle mismatch: got {math.degrees(ang_out):.4f}°, expected {deg}°"
            )

    def test_slerp_endpoints_are_input_quaternions(self):
        q1 = _from_axis_angle([0, 0, 1],  30)
        q2 = _from_axis_angle([1, 0, 0], 120)
        r0 = slerp(q1, q2, 0.0)
        r1 = slerp(q1, q2, 1.0)
        assert _q_close(r0, q1, atol=1e-10)
        assert _q_close(r1, q2, atol=1e-10)

    def test_slerp_preserves_unit_norm(self):
        q1 = _from_axis_angle([1, 0, 0], 45)
        q2 = _from_axis_angle([0, 1, 0], 135)
        for t in np.linspace(0, 1, 21):
            r = slerp(q1, q2, float(t))
            assert math.isclose(r.norm(), 1.0, rel_tol=1e-10), (
                f"slerp result not unit at t={t:.2f}: norm={r.norm()}"
            )

    def test_conjugate_is_inverse_rotation(self):
        """Applying q then q† (conjugate) should recover the original vector."""
        q = _from_axis_angle([1, 1, 0], 80)
        v = np.array([1., 2., -1.])
        rv  = rotate_vec(q, v)
        rrv = rotate_vec(q.conj(), rv)
        np.testing.assert_allclose(rrv, v, atol=1e-10,
            err_msg="q† should be the inverse of q")

    def test_composition_associativity(self):
        """(q1·q2)·q3 and q1·(q2·q3) should produce the same rotation."""
        q1 = _from_axis_angle([0, 0, 1], 60)
        q2 = _from_axis_angle([1, 0, 0], 90)
        q3 = _from_axis_angle([0, 1, 0], 45)
        v  = np.array([1., 0., 0.])

        qab = _unit(q1 * q2)
        v_abc = rotate_vec(_unit(qab * q3), v)

        qbc = _unit(q2 * q3)
        v_abc2 = rotate_vec(_unit(q1 * qbc), v)

        np.testing.assert_allclose(v_abc, v_abc2, atol=1e-10,
            err_msg="Rotation composition should be associative")

    def test_90_deg_z_frame_rotation(self):
        """x̂ frame vector rotated 90° around ẑ should land on ŷ."""
        q = _from_axis_angle([0, 0, 1], 90)
        rv = rotate_vec(q, np.array([1., 0., 0.]))
        np.testing.assert_allclose(rv, [0., 1., 0.], atol=1e-10)

    def test_slerp_midpoint_axis_alignment(self):
        """
        Halfway between the identity and a 90° z-rotation should be a 45° z-rotation.
        """
        q1 = Quaternion.one()
        q2 = _from_axis_angle([0, 0, 1], 90)
        mid = slerp(q1, q2, 0.5)
        ax, ang = axis_angle(mid)
        assert math.isclose(ang, math.pi / 4, rel_tol=1e-9), (
            f"Midpoint angle should be 45°, got {math.degrees(ang):.4f}°"
        )
        np.testing.assert_allclose(ax, [0., 0., 1.], atol=1e-9)


# ---------------------------------------------------------------------------
# add_flag
# ---------------------------------------------------------------------------

class TestAddFlag:
    def test_basic_no_crash(self):
        q = _from_axis_angle([0, 0, 1], 90)
        viz = QuaternionVisualizer()
        result = viz.add_flag(q)
        assert result is viz

    def test_chaining(self):
        viz = QuaternionVisualizer()
        result = (viz
                  .add_flag(_from_axis_angle([0, 0, 1], 90), color="steelblue", label="q1")
                  .add_flag(_from_axis_angle([1, 0, 0], 60),  color="tomato",    label="q2"))
        assert result is viz

    def test_label_in_legend(self):
        viz = QuaternionVisualizer()
        viz.add_flag(_from_axis_angle([0, 0, 1], 90), label="flagq")
        _, labels = viz.ax.get_legend_handles_labels()
        assert "flagq" in labels

    def test_identity_no_crash(self):
        viz = QuaternionVisualizer()
        viz.add_flag(Quaternion.one())   # angle ≈ 0 — arc and ref line suppressed

    def test_180_deg_no_crash(self):
        viz = QuaternionVisualizer()
        viz.add_flag(_from_axis_angle([1, 0, 0], 180))

    def test_no_arc(self):
        viz = QuaternionVisualizer()
        result = viz.add_flag(_from_axis_angle([0, 1, 0], 75), show_arc=False)
        assert result is viz

    def test_no_reference(self):
        viz = QuaternionVisualizer()
        result = viz.add_flag(_from_axis_angle([0, 1, 0], 75), show_reference=False)
        assert result is viz

    def test_custom_pole_and_flag_size(self):
        viz = QuaternionVisualizer()
        result = viz.add_flag(
            _from_axis_angle([1, 1, 0], 120),
            pole_length=0.8, flag_len=0.25, flag_width=0.15,
        )
        assert result is viz

    def test_pennant_tip_direction(self):
        """
        For a 90° z-rotation, axis = ẑ.  In _plane_frame(ẑ) the reference
        direction u lies in the xy-plane.  The pennant tip should be at
        cos(90°)·u + sin(90°)·v = v (perpendicular to u in the xy-plane).
        """
        q = _from_axis_angle([0, 0, 1], 90)
        ax_vec, angle = axis_angle(q)
        u, v = _plane_frame(ax_vec)
        cos_t, sin_t = math.cos(angle), math.sin(angle)
        fd = cos_t * u + sin_t * v
        # fd should be a unit vector in the ⊥ plane
        assert math.isclose(np.linalg.norm(fd), 1.0, rel_tol=1e-10)
        # For angle = π/2: fd = v exactly
        np.testing.assert_allclose(fd, v, atol=1e-10)

    def test_pennant_tip_is_unit_for_various_angles(self):
        """The pennant direction fd = cos(θ)·u + sin(θ)·v should always be unit."""
        for deg in [0, 30, 60, 90, 120, 150, 180]:
            q = _from_axis_angle([1, 1, 0], deg)
            ax_vec, angle = axis_angle(q)
            u, v = _plane_frame(ax_vec)
            fd = math.cos(angle) * u + math.sin(angle) * v
            assert math.isclose(np.linalg.norm(fd), 1.0, rel_tol=1e-10), (
                f"fd not unit at {deg}°: |fd|={np.linalg.norm(fd)}"
            )

    def test_pennant_perp_in_plane(self):
        """fd_perp = cos(θ)·v − sin(θ)·u should be ⊥ to fd and unit."""
        q = _from_axis_angle([0, 0, 1], 60)
        ax_vec, angle = axis_angle(q)
        u, v = _plane_frame(ax_vec)
        cos_t, sin_t = math.cos(angle), math.sin(angle)
        fd      =  cos_t * u + sin_t * v
        fd_perp =  cos_t * v - sin_t * u
        assert abs(np.dot(fd, fd_perp)) < 1e-12, "fd and fd_perp are not orthogonal"
        assert math.isclose(np.linalg.norm(fd_perp), 1.0, rel_tol=1e-10)

    def test_save_with_flag(self, tmp_path):
        import os
        viz = QuaternionVisualizer()
        viz.draw_axes()
        viz.add_flag(_from_axis_angle([0, 0, 1], 90), label="q")
        path = str(tmp_path / "flag.png")
        viz.save(path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# add_unnormalized_flag
# ---------------------------------------------------------------------------

class TestAddUnnormalizedFlag:
    def test_basic_no_crash(self):
        q = Quaternion(1.5, 0.6, 0.0, 0.6)
        viz = QuaternionVisualizer()
        result = viz.add_unnormalized_flag(q)
        assert result is viz

    def test_chaining(self):
        viz = QuaternionVisualizer()
        result = (viz
                  .add_unnormalized_flag(Quaternion(0.6, 0.0, 0.0, 0.6), label="q1")
                  .add_unnormalized_flag(Quaternion(1.2, 0.5, 0.3, 0.0), label="q2"))
        assert result is viz

    def test_label_in_legend(self):
        viz = QuaternionVisualizer()
        viz.add_unnormalized_flag(Quaternion(1.0, 0.5, 0.5, 0.0), label="unorm")
        _, labels = viz.ax.get_legend_handles_labels()
        assert "unorm" in labels

    def test_zero_raises(self):
        viz = QuaternionVisualizer()
        with pytest.raises(ValueError):
            viz.add_unnormalized_flag(Quaternion(0, 0, 0, 0))

    def test_unit_quaternion_same_as_add_flag(self):
        """For a unit quaternion the pole length should be ≈ 1 (same as add_flag default)."""
        q = _from_axis_angle([0, 0, 1], 90)
        assert math.isclose(q.norm(), 1.0, rel_tol=1e-10)
        viz = QuaternionVisualizer()
        viz.add_unnormalized_flag(q)   # pole_length = 1.0 — should not crash

    def test_pole_length_equals_norm(self):
        """
        The pole tip is at ax_vec * ‖q‖, so its distance from the origin
        should equal the quaternion norm.
        """
        q = Quaternion(1.5, 0.6, 0.0, 0.6)
        n = q.norm()
        ax_vec, _ = axis_angle(q)
        tip = ax_vec * n
        assert math.isclose(np.linalg.norm(tip), n, rel_tol=1e-10)

    def test_norm_less_than_one(self):
        """A sub-unit quaternion should produce a short pole without error."""
        q = Quaternion(0.3, 0.0, 0.2, 0.0)
        assert q.norm() < 1.0
        viz = QuaternionVisualizer()
        viz.add_unnormalized_flag(q)

    def test_norm_greater_than_one(self):
        """A super-unit quaternion should produce a long pole without error."""
        q = Quaternion(2.0, 1.0, 1.0, 1.0)
        assert q.norm() > 1.0
        viz = QuaternionVisualizer()
        viz.add_unnormalized_flag(q)

    def test_no_norm_label(self):
        q = Quaternion(1.0, 0.5, 0.0, 0.5)
        viz = QuaternionVisualizer()
        result = viz.add_unnormalized_flag(q, show_norm_label=False)
        assert result is viz

    def test_axis_angle_matches_normalized(self):
        """Axis and angle extracted from q and from q/‖q‖ must agree."""
        q = Quaternion(1.2, 0.4, 0.6, 0.2)
        q_unit = _unit(q)
        ax1, ang1 = axis_angle(q)
        ax2, ang2 = axis_angle(q_unit)
        np.testing.assert_allclose(ax1, ax2, atol=1e-10)
        assert math.isclose(ang1, ang2, rel_tol=1e-10)

    def test_save_with_unnormalized_flag(self, tmp_path):
        import os
        viz = QuaternionVisualizer()
        viz.draw_axes()
        viz.add_unnormalized_flag(Quaternion(1.5, 0.6, 0.0, 0.6), label="q")
        path = str(tmp_path / "unorm_flag.png")
        viz.save(path)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# add_sandwich_rotation
# ---------------------------------------------------------------------------

class TestAddSandwichRotation:
    # Helpers shared by several tests
    _p = Quaternion(0.0, 1.0, 0.0, 0.0)         # pure vector along x, ‖p‖=1
    _q = Quaternion(math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4))  # 90° z, unit

    def test_basic_no_crash(self):
        viz = QuaternionVisualizer()
        result = viz.add_sandwich_rotation(self._p, self._q)
        assert result is viz

    def test_chaining(self):
        viz = QuaternionVisualizer()
        p2 = Quaternion(0.0, 0.0, 1.0, 0.0)
        q2 = Quaternion(1.2, 0.3, 0.3, 0.0)
        result = (viz
                  .add_sandwich_rotation(self._p, self._q)
                  .add_sandwich_rotation(p2, q2,
                                         color_p="purple", color_q="brown",
                                         color_result="gray"))
        assert result is viz

    def test_labels_in_legend(self):
        viz = QuaternionVisualizer()
        viz.add_sandwich_rotation(self._p, self._q,
                                  label_p="my_p", label_q="my_q",
                                  label_result="my_r")
        _, labels = viz.ax.get_legend_handles_labels()
        assert "my_p" in labels
        assert "my_q" in labels
        assert "my_r" in labels

    def test_no_trajectory(self):
        viz = QuaternionVisualizer()
        result = viz.add_sandwich_rotation(self._p, self._q,
                                            show_trajectory=False)
        assert result is viz

    def test_result_norm_equals_q_norm_sq_times_p_norm(self):
        """‖q·p·q†‖ must equal ‖q‖² · ‖p‖ for arbitrary non-unit q, p."""
        p = Quaternion(0.5, 1.2, -0.3, 0.8)
        q = Quaternion(1.1, 0.4,  0.6, 0.2)
        r = q * p * q.conj()
        expected = q.norm() ** 2 * p.norm()
        assert math.isclose(r.norm(), expected, rel_tol=1e-10), (
            f"‖r‖={r.norm():.6f}, expected ‖q‖²·‖p‖={expected:.6f}"
        )

    def test_unit_q_preserves_p_norm(self):
        """When q is unit, ‖q·p·q†‖ == ‖p‖ (pure rotation, no scaling)."""
        p = Quaternion(0.5, 1.2, -0.3, 0.8)
        q = _from_axis_angle([1, 1, 0], 75)
        r = q * p * q.conj()
        assert math.isclose(r.norm(), p.norm(), rel_tol=1e-10)

    def test_rotation_of_pure_vector_x_by_90_deg_z(self):
        """q·(0,1,0,0)·q† with q=90°-z should give (0,0,1,0) (x→y rotation)."""
        p = Quaternion(0.0, 1.0, 0.0, 0.0)       # pure x-vector
        q = _from_axis_angle([0, 0, 1], 90)        # unit, 90° around z
        r = q * p * q.conj()
        np.testing.assert_allclose([r.w, r.x, r.y, r.z],
                                    [0.0, 0.0, 1.0, 0.0], atol=1e-10)

    def test_scaled_q_rotates_and_scales(self):
        """Scaling q by s scales the result by s², without changing the rotation axis."""
        p = Quaternion(0.0, 1.0, 0.0, 0.0)
        q_unit = _from_axis_angle([0, 0, 1], 90)
        s = 1.5
        q_scaled = Quaternion(q_unit.w * s, q_unit.x * s,
                               q_unit.y * s, q_unit.z * s)
        r_unit   = q_unit   * p * q_unit.conj()
        r_scaled = q_scaled * p * q_scaled.conj()
        # Axis directions must agree
        ax_u, _ = axis_angle(r_unit)
        ax_s, _ = axis_angle(r_scaled)
        np.testing.assert_allclose(ax_u, ax_s, atol=1e-10)
        # Norms: r_scaled = s² · r_unit
        assert math.isclose(r_scaled.norm(), s**2 * r_unit.norm(), rel_tol=1e-10)

    def test_trajectory_start_and_end(self):
        """
        The trajectory arc should start near p's pole tip and end near r's pole tip.
        Verified geometrically: at t=0 the rotated axis is p's axis, at t=1 it's r's.
        """
        p = Quaternion(0.0, 1.0, 0.0, 0.0)
        q = _from_axis_angle([0, 0, 1], 90)
        r = q * p * q.conj()
        # t=0: slerp(one, q, 0) = one; rotate_vec(one, p_ax) = p_ax
        p_ax, _ = axis_angle(p)
        start_pt = rotate_vec(Quaternion.one(), p_ax) * p.norm()
        np.testing.assert_allclose(start_pt, p_ax * p.norm(), atol=1e-10)
        # t=1: slerp(one, q, 1) = q; rotate_vec(q, p_ax) = r's axis
        r_ax, _ = axis_angle(r)
        end_pt = rotate_vec(q, p_ax) * r.norm()
        np.testing.assert_allclose(end_pt, r_ax * r.norm(), atol=1e-9)

    def test_non_unit_p_and_q(self):
        """Both p and q non-unit — should draw without error."""
        p = Quaternion(0.8, 0.6, 0.0, 0.0)
        q = Quaternion(1.2, 0.0, 0.0, 0.9)
        viz = QuaternionVisualizer()
        viz.add_sandwich_rotation(p, q)

    def test_save_sandwich(self, tmp_path):
        import os
        viz = QuaternionVisualizer()
        viz.draw_axes()
        viz.add_sandwich_rotation(self._p, self._q)
        path = str(tmp_path / "sandwich.png")
        viz.save(path)
        assert os.path.exists(path)
