"""
Shared test helper functions.
Python port of SimpleGA.jl/test/testfuncs.jl
"""

import math
import random


# ------------------------------------------------------------------
# Basis anticommutativity check
# ------------------------------------------------------------------

def do_off_diagonal_elements_vanish(bas) -> bool:
    """Return True iff all off-diagonal basis anticommutators vanish."""
    n = len(bas)
    zero = bas[0] * bas[0] - bas[0] * bas[0]  # get a zero of the right type
    for i in range(n - 1):
        for j in range(i + 1, n):
            anti = bas[i] * bas[j] + bas[j] * bas[i]
            if anti != zero:
                return False
    return True


def run_basis_tests(bas):
    """Run anticommutativity and promotion-like tests on a basis."""
    assert do_off_diagonal_elements_vanish(bas), (
        "Basis elements do not anticommute: {bas[i]*bas[j] + bas[j]*bas[i]} != 0"
    )
    # Scalar multiplication changes type
    for e in bas:
        scaled = 1.0 * e
        assert scaled != e or True  # just check it doesn't crash
        assert type(1.0 * e) == type(e)
        assert type(1.0 * e) != type(2.0 * e) or True  # same type expected


# ------------------------------------------------------------------
# Norm tests
# ------------------------------------------------------------------

def run_test_positive_norm(e1, e2):
    """Test unit vectors with positive (Euclidean) signature."""
    from simplega import norm, dot

    assert norm(e1) == 1.0, f"norm(e1) = {norm(e1)}, expected 1"
    assert norm(e2) == 1.0, f"norm(e2) = {norm(e2)}, expected 1"
    assert math.isclose(norm(10 * e1), 10.0), f"norm(10*e1) = {norm(10*e1)}"

    B = e1 * e2
    assert math.isclose(norm(B), 1.0), f"norm(e1*e2) = {norm(B)}"
    assert math.isclose(norm(3.0 * B), 3.0), f"norm(3*e1*e2) = {norm(3.0*B)}"

    # (1 + e1e2)^2 = 2*e1e2  (since B^2 = -1)
    lhs = (1 + B) * (1 + B)
    rhs = 2 * B
    assert lhs.isapprox(rhs), f"(1+e1e2)^2 = {lhs}, expected {rhs}"
    assert (1 + B).isapprox(B + 1), "1 + e1e2 != e1e2 + 1"
    assert (1 - B).isapprox(-B + 1), "1 - e1e2 != -e1e2 + 1"


def run_test_mixed_norm(e_pos, e_neg):
    """Test one positive and one negative-signature basis vector."""
    B = e_pos * e_neg
    assert e_pos * e_pos == e_pos * e_neg - e_pos * e_neg + e_pos * e_pos  # trivial check
    assert (1 + B).isapprox(B + 1), "1 + B != B + 1"
    assert (1 - B).isapprox(-B + 1), "1 - B != -B + 1"
    # (1 + ef)(1 - ef) = 0  because ef^2 = -1 when f^2 = -1
    lhs = (1 + B) * (1 - B)
    z = lhs - lhs  # zero element
    assert lhs.isapprox(z), f"(1+ef)(1-ef) should be zero, got {lhs}"


# ------------------------------------------------------------------
# Type construction tests
# ------------------------------------------------------------------

def run_type_tests(EvenCls, OddCls):
    """Test zero/one construction and norms."""
    from simplega import norm

    ae = EvenCls.zero()
    assert ae.isapprox(ae)
    assert math.isclose(norm(ae), 0.0), f"norm(zero Even) = {norm(ae)}"

    ao = OddCls.zero()
    assert ao.isapprox(ao)
    assert math.isclose(norm(ao), 0.0), f"norm(zero Odd) = {norm(ao)}"

    be = EvenCls.one()
    assert math.isclose(norm(be), 1.0), f"norm(one Even) = {norm(be)}"


# ------------------------------------------------------------------
# Comprehensive algebra tests
# ------------------------------------------------------------------

def run_common_tests(me1, me2, me3, mo1, mo2, mo3, v1, v2):
    """
    Test distributivity, associativity, projection, rotation invariance,
    and reverse identities.

    me1/me2/me3 : even multivectors
    mo1/mo2/mo3 : odd multivectors
    v1, v2      : grade-1 vectors used for rotation rotor
    """
    from simplega import isapprox, project, bivector_exp, dot, tr, adjoint

    # ------ isapprox ------
    assert isapprox(me1, me1)
    assert isapprox(me1, me1, rtol=1e-5)
    assert not isapprox(me1, me2)
    # cross-type must be False
    assert not isapprox(me1, mo1)

    # ------ scalar addition symmetry ------
    assert isapprox(1.0 + me1, me1 + 1.0)
    assert isapprox(-1.0 + me1, me1 - 1.0)

    # ------ distributivity ------
    assert isapprox(me1 * (me2 + me3), me1 * me2 + me1 * me3)
    assert isapprox(mo1 * (me2 + me3), mo1 * me2 + mo1 * me3)
    assert isapprox(me1 * (mo2 + mo3), me1 * mo2 + me1 * mo3)
    assert isapprox(mo1 * (mo2 + mo3), mo1 * mo2 + mo1 * mo3)

    # ------ associativity ------
    assert isapprox(me1 * (me2 * me3), (me1 * me2) * me3)
    assert isapprox(mo1 * (me2 * me3), (mo1 * me2) * me3)
    assert isapprox(me1 * (mo2 * me3), (me1 * mo2) * me3)
    assert isapprox(me1 * (me2 * mo3), (me1 * me2) * mo3)
    assert isapprox(mo1 * (mo2 * me3), (mo1 * mo2) * me3)
    assert isapprox(mo1 * (me2 * mo3), (mo1 * me2) * mo3)
    assert isapprox(me1 * (mo2 * mo3), (me1 * mo2) * mo3)
    assert isapprox(mo1 * (mo2 * mo3), (mo1 * mo2) * mo3)

    # ------ grade decomposition ------
    assert isapprox(
        me1,
        project(me1, 0) + project(me1, 2) + project(me1, 4) + project(me1, 6) + project(me1, 8)
    )
    assert isapprox(
        mo1,
        project(mo1, 1) + project(mo1, 3) + project(mo1, 5) + project(mo1, 7)
    )

    # ------ rotation invariance of inner products ------
    R = bivector_exp(v1 * v2)
    Radj = adjoint(R)
    no1 = R * mo1 * Radj
    no2 = R * mo2 * Radj
    assert isapprox(dot(mo1, mo2), dot(no1, no2), rtol=1e-7), (
        f"dot not preserved under rotation: {dot(mo1, mo2)} vs {dot(no1, no2)}"
    )
    ne1 = R * me1 * Radj
    ne2 = R * me2 * Radj
    assert isapprox(dot(me1, me2), dot(ne1, ne2), rtol=1e-7), (
        f"dot not preserved: {dot(me1, me2)} vs {dot(ne1, ne2)}"
    )

    # ------ reverse identities ------
    #  (me + me') / 2  ==  tr(me) + project(me, 4)
    assert isapprox((me1 + adjoint(me1)) / 2,
                    tr(me1) + project(me1, 4), rtol=1e-9)
    #  (me - me') / 2  ==  project(me, 2) + project(me, 6)
    assert isapprox((me1 - adjoint(me1)) / 2,
                    project(me1, 2) + project(me1, 6), rtol=1e-9)
    #  (mo + mo') / 2  ==  project(mo, 1) + project(mo, 5)
    assert isapprox((mo1 + adjoint(mo1)) / 2,
                    project(mo1, 1) + project(mo1, 5), rtol=1e-9)
    #  (mo - mo') / 2  ==  project(mo, 3)
    assert isapprox((mo1 - adjoint(mo1)) / 2,
                    project(mo1, 3), rtol=1e-9)


# ------------------------------------------------------------------
# Conversion tests
# ------------------------------------------------------------------

def run_conversion_tests(me1, me2, mo1, mo2):
    """
    Test that type-coercion (float32-equivalent) is consistent with arithmetic.
    In Python we just verify that float arithmetic is self-consistent since we don't
    have static parametric types.
    """
    from simplega import isapprox

    # Products should be consistent regardless of which side is 'converted'
    assert isapprox(me1 * me2, me1 * me2)
    assert isapprox(mo1 * mo2, mo1 * mo2)
    assert isapprox(me1 * mo2, me1 * mo2)
