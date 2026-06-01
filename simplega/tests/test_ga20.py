"""
Tests for GA(2,0).
Python port of SimpleGA.jl/test/test20.jl
"""

import math
import random
import pytest

import simplega.ga20 as GA20
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
    run_type_tests(GA20.Even, GA20.Odd)


def test_basis():
    bas = GA20.basis
    e1, e2 = bas

    # Both basis vectors should have dot product == 1 with themselves
    dots = [dot(e, e) for e in bas]
    assert dots == [1, 1], f"Expected [1,1], got {dots}"

    run_basis_tests(bas)


# ------------------------------------------------------------------
# Norm identities
# ------------------------------------------------------------------

def test_positive_norms():
    e1, e2 = GA20.basis
    run_test_positive_norm(e1, e2)


def test_scalar_one():
    e1, e2 = GA20.basis
    B = e1 * e2
    one = GA20.Even.one()
    assert e1 * e1 == one, f"e1*e1 = {e1*e1}, expected {one}"
    assert e2 * e2 == one, f"e2*e2 = {e2*e2}, expected {one}"


# ------------------------------------------------------------------
# Arithmetic / geometric product
# ------------------------------------------------------------------

def test_anticommutation():
    e1, e2 = GA20.basis
    zero_e = GA20.Even.zero()
    assert (e1 * e2 + e2 * e1) == zero_e, "e1e2 + e2e1 should vanish"


def test_bivector_squares_negative():
    e1, e2 = GA20.basis
    B = e1 * e2
    # In GA(2,0): (e1e2)^2 = -1
    expected = GA20.Even(-1 + 0j)
    assert B * B == expected, f"(e1e2)^2 = {B*B}, expected {expected}"


def test_scalar_mul_commutativity():
    e1, e2 = GA20.basis
    B = e1 * e2
    assert (3.0 * B).isapprox(B * 3.0)
    assert (5.0 * e1).isapprox(e1 * 5.0)


# ------------------------------------------------------------------
# Exponential
# ------------------------------------------------------------------

def test_bivector_exp():
    e1, e2 = GA20.basis
    B = e1 * e2   # B^2 = -1, so exp(t*B) = cos(t) + sin(t)*B
    t = math.pi / 4
    R = bivector_exp(t * B)
    # Should be cos(pi/4) + sin(pi/4) * I2
    expected = GA20.Even(complex(math.cos(t), math.sin(t)))
    assert R.isapprox(expected, rtol=1e-12), f"bivector_exp mismatch: {R} vs {expected}"


def test_exp_full():
    e1, e2 = GA20.basis
    me = 0.3 + 0.7 * e1 * e2   # scalar + bivector
    R = me.exp()
    assert isinstance(R, GA20.Even)


# ------------------------------------------------------------------
# Adjoint / reverse
# ------------------------------------------------------------------

def test_adjoint():
    e1, e2 = GA20.basis
    # adjoint of e1 is e1 (Odd adjoint = identity in GA20)
    assert adjoint(e1) == e1
    # adjoint of e1e2 conjugates c1: Even(0+1j) -> Even(0-1j)
    B = e1 * e2
    Badj = adjoint(B)
    assert Badj.isapprox(GA20.Even(0 - 1j)), f"adjoint(e1e2) = {Badj}"


# ------------------------------------------------------------------
# Grade projection
# ------------------------------------------------------------------

def test_projection():
    e1, e2 = GA20.basis
    me = GA20.Even(2 + 3j)  # scalar=2, bivector=3
    assert project(me, 0).isapprox(GA20.Even(2 + 0j))
    assert project(me, 2).isapprox(GA20.Even(0 + 3j))
    assert project(me, 4).isapprox(GA20.Even.zero())

    mo = 4 * e1 + 5 * e2
    assert project(mo, 1).isapprox(mo)
    assert project(mo, 3).isapprox(GA20.Odd.zero())


# ------------------------------------------------------------------
# Comprehensive random tests
# ------------------------------------------------------------------

def test_common_properties():
    random.seed(42)
    e1, e2 = GA20.basis

    me1 = GA20.Even(complex(random.random(), random.random()))
    me2 = GA20.Even(complex(random.random(), random.random()))
    me3 = GA20.Even(complex(random.random(), random.random()))
    mo1 = GA20.Odd(complex(random.random(), random.random()))
    mo2 = GA20.Odd(complex(random.random(), random.random()))
    mo3 = GA20.Odd(complex(random.random(), random.random()))

    arr1 = [random.random(), random.random()]
    v1 = inject(arr1, GA20.basis)
    arr2 = [random.random(), random.random()]
    v2 = inject(arr2, GA20.basis)

    run_common_tests(me1, me2, me3, mo1, mo2, mo3, v1, v2)


# ------------------------------------------------------------------
# dot product
# ------------------------------------------------------------------

def test_dot_basis():
    e1, e2 = GA20.basis
    assert dot(e1, e1) == 1.0
    assert dot(e2, e2) == 1.0
    assert dot(e1, e2) == 0.0


def test_dot_even_odd_zero():
    e1, e2 = GA20.basis
    B = e1 * e2
    assert dot(B, e1) == 0.0
    assert dot(e1, B) == 0.0


# ------------------------------------------------------------------
# Trace
# ------------------------------------------------------------------

def test_trace():
    e1, e2 = GA20.basis
    me = GA20.Even(3 + 7j)
    assert math.isclose(me.tr(), 3.0)
    assert e1.tr() == 0.0
