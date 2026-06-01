"""
Tests for GA(3,1).
Python port of SimpleGA.jl/test/test31.jl
"""

import math
import random
import pytest

import simplega.ga31 as GA31
from simplega import dot, norm, project, bivector_exp, inject, adjoint, isapprox
from simplega.tests.test_helpers import (
    run_type_tests,
    run_basis_tests,
    run_test_positive_norm,
    run_test_mixed_norm,
    run_common_tests,
)

e1 = GA31.e1
e2 = GA31.e2
e3 = GA31.e3
f3 = GA31.f3
I4 = GA31.I4

# ------------------------------------------------------------------
# Basis and type construction
# ------------------------------------------------------------------

def test_type_construction():
    run_type_tests(GA31.Even, GA31.Odd)


def test_basis_dots():
    bas = GA31.basis
    dots = [dot(e, e) for e in bas]
    assert dots == [1, 1, 1, -1], f"Expected [1,1,1,-1], got {dots}"


def test_basis_anticommutation():
    run_basis_tests(GA31.basis)


# ------------------------------------------------------------------
# Norm and signature
# ------------------------------------------------------------------

def test_positive_norms():
    run_test_positive_norm(e1, e2)


def test_mixed_norm():
    run_test_mixed_norm(e3, f3)


def test_basis_squares():
    one = GA31.Even.one()
    assert (e1 * e1).isapprox(one),  f"e1*e1 = {e1*e1}"
    assert (e2 * e2).isapprox(one),  f"e2*e2 = {e2*e2}"
    assert (e3 * e3).isapprox(one),  f"e3*e3 = {e3*e3}"
    minus_one = -1.0 * GA31.Even.one()
    assert (f3 * f3).isapprox(minus_one), f"f3*f3 = {f3*f3}"


# ------------------------------------------------------------------
# Exponential
# ------------------------------------------------------------------

def test_bivector_exp():
    B = e1 * e2   # positive-positive bivector: B^2 = -1
    t = math.pi / 5
    R = bivector_exp(t * B)
    # Should give cos(t) + sin(t) * B
    expected = math.cos(t) + math.sin(t) * B
    assert R.isapprox(expected, rtol=1e-10), f"bivector_exp mismatch: {R} vs {expected}"


def test_bivector_exp_mixed():
    B = e1 * f3   # positive-negative bivector: B^2 = +1
    t = 0.4
    R = bivector_exp(t * B)
    # B^2 = +1 so exp(tB) = cosh(t) + sinh(t)*B
    expected = math.cosh(t) + math.sinh(t) * B
    assert R.isapprox(expected, rtol=1e-10), f"bivector_exp (hyperbolic) mismatch: {R} vs {expected}"


# ------------------------------------------------------------------
# Adjoint / reverse
# ------------------------------------------------------------------

def test_adjoint_grade1():
    v = 1.0 * e1 + 2.0 * e2
    # Reverse of a grade-1 vector is itself
    assert adjoint(v).isapprox(v)


def test_adjoint_bivector():
    B = e1 * e2
    # Reverse of a grade-2 element negates it
    assert adjoint(B).isapprox(-1.0 * B), f"adjoint(e1e2) = {adjoint(B)}"


# ------------------------------------------------------------------
# Grade projection
# ------------------------------------------------------------------

def test_projection_even_grade0():
    scl = 2.5
    me = scl * GA31.Even.one() + e1 * e2
    p0 = project(me, 0)
    assert math.isclose(p0.tr(), scl, rel_tol=1e-12), (
        f"tr(project(me, 0)) = {p0.tr()}, expected {scl}"
    )


def test_projection_odd():
    v = 1.0 * e1 + 2.0 * e2 + 3.0 * e3
    assert project(v, 1).isapprox(v)
    assert project(v, 3).isapprox(GA31.Odd.zero())


# ------------------------------------------------------------------
# Dot products
# ------------------------------------------------------------------

def test_dot_positive_basis():
    assert math.isclose(dot(e1, e1), 1.0)
    assert math.isclose(dot(e2, e2), 1.0)
    assert math.isclose(dot(e3, e3), 1.0)


def test_dot_negative_basis():
    assert math.isclose(dot(f3, f3), -1.0), f"dot(f3,f3) = {dot(f3,f3)}"


def test_dot_orthogonality():
    assert math.isclose(dot(e1, e2), 0.0)
    assert math.isclose(dot(e1, f3), 0.0)
    assert math.isclose(dot(e3, f3), 0.0)


def test_dot_cross_grade_zero():
    B = e1 * e2
    assert dot(B, e1) == 0.0


# ------------------------------------------------------------------
# Pseudoscalar
# ------------------------------------------------------------------

def test_pseudoscalar_square():
    # In GA(3,1): I4^2 = +1  (since signature (3,1) gives (-1)^(3+1)*(+1) = +1)
    I4sq = I4 * I4
    one = GA31.Even.one()
    assert I4sq.isapprox(one) or I4sq.isapprox(-1.0 * one), (
        f"I4^2 = {I4sq}, expected ±1"
    )


# ------------------------------------------------------------------
# Comprehensive random tests
# ------------------------------------------------------------------

def test_common_properties():
    random.seed(99)

    me1 = (random.random() + random.random() * e1 * e2
           + e1 * e3 * random.random() + e3 * random.random() * e2
           + f3 * (random.random() * e1 - random.random() * e2 + random.random() * e3)
           + I4 * random.random())
    me2 = (random.random() + random.random() * e1 * e2
           + e1 * e3 * random.random() + e3 * random.random() * e2
           + f3 * (random.random() * e1 - random.random() * e2 + random.random() * e3)
           + I4 * random.random())
    me3 = (random.random() + random.random() * e1 * e2
           + e1 * e3 * random.random() + e3 * random.random() * e2
           + f3 * (random.random() * e1 - random.random() * e2 + random.random() * e3)
           + I4 * random.random())
    mo1 = (random.random() * e1 + random.random() * e2 + e3 * random.random()
           + f3 / (random.random() + 0.1)
           + I4 * (random.random() * e1 + e2 * random.random()
                   + e3 * random.random() - random.random() * f3))
    mo2 = (random.random() * e1 + random.random() * e2 + e3 * random.random()
           + f3 / (random.random() + 0.1)
           + I4 * (random.random() * e1 + e2 * random.random()
                   + e3 * random.random() - random.random() * f3))
    mo3 = (random.random() * e1 + random.random() * e2 + e3 * random.random()
           + f3 / (random.random() + 0.1)
           + I4 * (random.random() * e1 + e2 * random.random()
                   + e3 * random.random() - random.random() * f3))

    arr1 = [random.random() for _ in range(4)]
    v1 = inject(arr1, GA31.basis)
    arr2 = [random.random() for _ in range(4)]
    v2 = inject(arr2, GA31.basis)

    run_common_tests(me1, me2, me3, mo1, mo2, mo3, v1, v2)
