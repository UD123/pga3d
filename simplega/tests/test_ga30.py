"""
Tests for GA(3,0).
Python port of SimpleGA.jl/test/test30.jl
"""

import math
import random
import pytest

import simplega.ga30 as GA30
from simplega import dot, norm, project, bivector_exp, inject, adjoint, isapprox
from simplega.tests.test_helpers import (
    run_type_tests,
    run_basis_tests,
    run_test_positive_norm,
    run_common_tests,
)

# ------------------------------------------------------------------
# Basis and type construction
# ------------------------------------------------------------------

def test_type_construction():
    run_type_tests(GA30.Even, GA30.Odd)


def test_basis_dots():
    bas = GA30.basis
    dots = [dot(e, e) for e in bas]
    assert dots == [1, 1, 1], f"Expected [1,1,1], got {dots}"


def test_basis_anticommutation():
    run_basis_tests(GA30.basis)


# ------------------------------------------------------------------
# Norm and algebraic identities
# ------------------------------------------------------------------

def test_positive_norms():
    e1, e2, e3 = GA30.basis
    run_test_positive_norm(e1, e2)


def test_basis_squares():
    e1, e2, e3 = GA30.basis
    one = GA30.Even.one()
    assert e1 * e1 == one
    assert e2 * e2 == one
    assert e3 * e3 == one


def test_bivector_squares():
    e1, e2, e3 = GA30.basis
    # In GA(3,0), every unit bivector squares to -1
    for B in [e1 * e2, e2 * e3, e1 * e3]:
        BsqNorm = norm(B * B)
        assert math.isclose(BsqNorm, 1.0, rel_tol=1e-12), (
            f"norm(B^2) = {BsqNorm}, expected 1"
        )
        minus_one = GA30.Even(-1.0, 0.0, 0.0, 0.0)
        assert (B * B).isapprox(minus_one), f"B^2 = {B*B}, expected -1"


def test_pseudoscalar():
    e1, e2, e3 = GA30.basis
    I3 = GA30.I3
    # I3^2 = -1 in GA(3,0)
    I3sq = I3 * I3
    expected = GA30.Even(-1.0, 0.0, 0.0, 0.0)
    assert I3sq.isapprox(expected), f"I3^2 = {I3sq}"


# ------------------------------------------------------------------
# Exponential / rotation
# ------------------------------------------------------------------

def test_bivector_exp_unit_rotation():
    e1, e2 = GA30.basis[:2]
    B = e1 * e2  # unit bivector
    t = math.pi / 3
    R = bivector_exp(t * B)
    # exp(t*B) = cos(t) + sin(t)*B  since B^2 = -1
    expected = math.cos(t) + math.sin(t) * B
    assert R.isapprox(expected, rtol=1e-12), f"bivector_exp mismatch: {R} vs {expected}"


def test_rotation_preserves_norm():
    random.seed(7)
    e1, e2, e3 = GA30.basis
    v = random.random() * e1 + random.random() * e2 + random.random() * e3
    B = e1 * e2
    R = bivector_exp(0.5 * B)
    rotated = R * v * adjoint(R)
    assert math.isclose(norm(v), norm(rotated), rel_tol=1e-12), (
        f"Rotation changed norm: {norm(v)} -> {norm(rotated)}"
    )


# ------------------------------------------------------------------
# Adjoint / reverse
# ------------------------------------------------------------------

def test_adjoint_even():
    e1, e2 = GA30.basis[:2]
    B = e1 * e2   # Even with z=-1
    Badj = adjoint(B)
    # reverse of bivector negates it
    assert Badj.isapprox(GA30.Even(0.0, 0.0, 0.0, 1.0)), (
        f"adjoint(e1e2) = {Badj}"
    )


def test_adjoint_odd():
    e1, e2, e3 = GA30.basis
    v = 1.0 * e1 + 2.0 * e2 + 3.0 * e3
    # reverse of a grade-1 element is itself
    assert adjoint(v).isapprox(v), f"adjoint(v) = {adjoint(v)}, expected {v}"


# ------------------------------------------------------------------
# Grade projection
# ------------------------------------------------------------------

def test_projection_even():
    e1, e2 = GA30.basis[:2]
    B = e1 * e2
    me = 3.0 + 2.0 * B  # Even(3, 0, 0, -2) since e1e2 = Even(0,0,0,-1)
    assert project(me, 0).isapprox(GA30.Even(3.0, 0.0, 0.0, 0.0))
    assert project(me, 2).isapprox(GA30.Even(0.0, 0.0, 0.0, me.z))
    assert project(me, 4).isapprox(GA30.Even.zero())


def test_projection_odd():
    e1, e2, e3 = GA30.basis
    I3 = GA30.I3
    v = 1.0 * e1 + 2.0 * e2 + 3.0 * e3 + 4.0 * I3
    assert project(v, 1).isapprox(1.0 * e1 + 2.0 * e2 + 3.0 * e3)
    assert project(v, 3).isapprox(4.0 * I3)
    assert project(v, 5).isapprox(GA30.Odd.zero())


# ------------------------------------------------------------------
# Dot product
# ------------------------------------------------------------------

def test_dot_basis():
    e1, e2, e3 = GA30.basis
    assert dot(e1, e1) == 1.0
    assert dot(e2, e2) == 1.0
    assert dot(e3, e3) == 1.0
    assert dot(e1, e2) == 0.0
    assert dot(e1, e3) == 0.0


def test_dot_cross_grade_zero():
    e1, e2, e3 = GA30.basis
    B = e1 * e2
    assert dot(B, e1) == 0.0


# ------------------------------------------------------------------
# Comprehensive random tests
# ------------------------------------------------------------------

def test_common_properties():
    random.seed(13)
    e1, e2, e3 = GA30.basis
    I3 = GA30.I3

    me1 = (random.random() + random.random() * e1 * e2
           + e1 * e3 * random.random() + e3 * random.random() * e2)
    me2 = (random.random() + random.random() * e1 * e2
           + e1 * e3 * random.random() + e3 * random.random() * e2)
    me3 = (random.random() + random.random() * e1 * e2
           + e1 * e3 * random.random() + e3 * random.random() * e2)
    mo1 = (random.random() * e1 + random.random() * e2 + e3 * random.random()
           + e3 * random.random() * e2 * e1)
    mo2 = (random.random() * e1 + random.random() * e2 + e3 * random.random()
           + e3 * random.random() * e2 * e1)
    mo3 = (random.random() * e1 + random.random() * e2 + e3 * random.random()
           + e3 * random.random() * e2 * e1)

    arr1 = [random.random() for _ in range(3)]
    v1 = inject(arr1, GA30.basis)
    arr2 = [random.random() for _ in range(3)]
    v2 = inject(arr2, GA30.basis)

    run_common_tests(me1, me2, me3, mo1, mo2, mo3, v1, v2)
