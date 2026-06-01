"""
GA(3,1) implementation.
Python port of SimpleGA.jl/src/core31.jl and src/ga31.jl

Underlying representation: 2x2 complex matrices stored as four complex scalars.
    Even(c1, c2, c3, c4)  <->  [[c1, c2], [c3, c4]]
    Odd(c1, c2, c3, c4)   <->  [[c1, c2], [c3, c4]]  (different transformation rules)

Signature: e1^2 = e2^2 = e3^2 = +1,  f3^2 = -1.

Basis elements:
    s1 = Even(0, 1, 1, 0)           Pauli sigma_1
    s2 = Even(0, -j, j, 0)          Pauli sigma_2
    s3 = Even(1, 0, 0, -1)          Pauli sigma_3
    f3 = Odd(1, 0, 0, 1)            Time-like vector
    e1 = s1 * f3
    e2 = s2 * f3
    e3 = s3 * f3
    I4 = e1 * e2 * e3 * f3          Pseudoscalar
"""

import cmath
import math


class Even:
    """Even-grade element of GA(3,1) stored as 2x2 complex matrix entries."""

    __slots__ = ("c1", "c2", "c3", "c4")

    def __init__(self, c1=0 + 0j, c2=0 + 0j, c3=0 + 0j, c4=0 + 0j):
        self.c1 = complex(c1)
        self.c2 = complex(c2)
        self.c3 = complex(c3)
        self.c4 = complex(c4)

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------
    @classmethod
    def zero(cls):
        z = 0 + 0j
        return cls(z, z, z, z)

    @classmethod
    def one(cls):
        """Identity matrix."""
        o, z = 1 + 0j, 0 + 0j
        return cls(o, z, z, o)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __neg__(self):
        return Even(-self.c1, -self.c2, -self.c3, -self.c4)

    def __add__(self, other):
        if isinstance(other, Even):
            return Even(self.c1 + other.c1, self.c2 + other.c2,
                        self.c3 + other.c3, self.c4 + other.c4)
        if isinstance(other, (int, float)):
            # scalar adds to diagonal
            return Even(self.c1 + other, self.c2, self.c3, self.c4 + other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Even(self.c1 + other, self.c2, self.c3, self.c4 + other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Even):
            return Even(self.c1 - other.c1, self.c2 - other.c2,
                        self.c3 - other.c3, self.c4 - other.c4)
        if isinstance(other, (int, float)):
            return Even(self.c1 - other, self.c2, self.c3, self.c4 - other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Even(-self.c1 + other, -self.c2, -self.c3, -self.c4 + other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Even):
            a, b = self, other
            return Even(
                a.c1 * b.c1 + a.c2 * b.c3,
                a.c1 * b.c2 + a.c2 * b.c4,
                a.c3 * b.c1 + a.c4 * b.c3,
                a.c4 * b.c4 + a.c3 * b.c2,
            )
        if isinstance(other, Odd):
            a, b = self, other
            return Odd(
                a.c1 * b.c1 + a.c2 * b.c3,
                a.c1 * b.c2 + a.c2 * b.c4,
                a.c3 * b.c1 + a.c4 * b.c3,
                a.c4 * b.c4 + a.c3 * b.c2,
            )
        if isinstance(other, (int, float)):
            return Even(other * self.c1, other * self.c2,
                        other * self.c3, other * self.c4)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Even(other * self.c1, other * self.c2,
                        other * self.c3, other * self.c4)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            inv = 1.0 / other
            return Even(inv * self.c1, inv * self.c2, inv * self.c3, inv * self.c4)
        return NotImplemented

    # ------------------------------------------------------------------
    # Reverse / adjoint
    # ------------------------------------------------------------------
    @property
    def adjoint(self):
        """Reverse: swap off-diagonal, negate them."""
        return Even(self.c4, -self.c2, -self.c3, self.c1)

    # ------------------------------------------------------------------
    # Grade projection
    # ------------------------------------------------------------------
    def project(self, n: int):
        tra = (self.c1 + self.c4) / 2
        if n == 0:
            return tra.real * Even.one()
        if n == 2:
            return (self - self.adjoint) / 2
        if n == 4:
            v = tra.imag * 1j
            return Even(v, 0 + 0j, 0 + 0j, v)
        return Even.zero()

    # ------------------------------------------------------------------
    # Trace / dot / norm
    # ------------------------------------------------------------------
    def tr(self) -> float:
        return (self.c1 + self.c4).real / 2

    def dot(self, other) -> float:
        if isinstance(other, Even):
            tmp = (self.c1 * other.c1 + self.c2 * other.c3
                   + self.c4 * other.c4 + self.c3 * other.c2)
            return tmp.real / 2
        if isinstance(other, Odd):
            return 0.0
        return NotImplemented

    def norm(self) -> float:
        return math.sqrt(abs(self.dot(self)))

    # ------------------------------------------------------------------
    # Exponential
    # ------------------------------------------------------------------
    def bivector_exp(self):
        """Exponential of grade-2 part."""
        a = self.project(2)
        aa = a * a
        fct = cmath.sqrt((aa.c1 + aa.c4) / 2)
        if fct == 0:
            return Even.one() + a
        ch = cmath.cosh(fct)
        sh = cmath.sinh(fct) / fct
        return _as_even(ch) + _as_even(sh) * a

    def exp(self):
        """Full even-grade exponential."""
        R = self.bivector_exp()
        fct = (self.c1 + self.c4) / 2
        if fct == 0:
            return R
        return _as_even(cmath.exp(fct)) * R

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def isapprox(self, other, rtol: float = 1e-9, atol: float = 0.0) -> bool:
        if not isinstance(other, Even):
            return False
        return (cmath.isclose(self.c1, other.c1, rel_tol=rtol, abs_tol=atol) and
                cmath.isclose(self.c2, other.c2, rel_tol=rtol, abs_tol=atol) and
                cmath.isclose(self.c3, other.c3, rel_tol=rtol, abs_tol=atol) and
                cmath.isclose(self.c4, other.c4, rel_tol=rtol, abs_tol=atol))

    def __eq__(self, other):
        if isinstance(other, Even):
            return (self.c1 == other.c1 and self.c2 == other.c2
                    and self.c3 == other.c3 and self.c4 == other.c4)
        return NotImplemented

    def __hash__(self):
        return hash((self.c1, self.c2, self.c3, self.c4))

    def __repr__(self):
        # Display as: scalar + bivector terms + pseudoscalar
        parts = []
        scl = self.tr()
        if scl != 0.0:
            parts.append(str(scl))
        comps = [
            (self.dot(-e1 * e2), "e1e2"),
            (self.dot(-e1 * e3), "e1e3"),
            (self.dot(-e2 * e3), "e2e3"),
            (self.dot(e1 * f3),  "e1f3"),
            (self.dot(e2 * f3),  "e2f3"),
            (self.dot(e3 * f3),  "e3f3"),
            (self.dot(-I4),      "I4"),
        ]
        for val, label in comps:
            if val != 0.0:
                parts.append(f"{val}{label}")
        return " + ".join(parts) if parts else "0"


class Odd:
    """Odd-grade element of GA(3,1) stored as 2x2 complex matrix entries."""

    __slots__ = ("c1", "c2", "c3", "c4")

    def __init__(self, c1=0 + 0j, c2=0 + 0j, c3=0 + 0j, c4=0 + 0j):
        self.c1 = complex(c1)
        self.c2 = complex(c2)
        self.c3 = complex(c3)
        self.c4 = complex(c4)

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------
    @classmethod
    def zero(cls):
        z = 0 + 0j
        return cls(z, z, z, z)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __neg__(self):
        return Odd(-self.c1, -self.c2, -self.c3, -self.c4)

    def __add__(self, other):
        if isinstance(other, Odd):
            return Odd(self.c1 + other.c1, self.c2 + other.c2,
                       self.c3 + other.c3, self.c4 + other.c4)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Odd):
            return Odd(self.c1 - other.c1, self.c2 - other.c2,
                       self.c3 - other.c3, self.c4 - other.c4)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Even):
            a, b = self, other
            return Odd(
                a.c1 * b.c4.conjugate() - a.c2 * b.c2.conjugate(),
                -a.c1 * b.c3.conjugate() + a.c2 * b.c1.conjugate(),
                a.c3 * b.c4.conjugate() - a.c4 * b.c2.conjugate(),
                a.c4 * b.c1.conjugate() - a.c3 * b.c3.conjugate(),
            )
        if isinstance(other, Odd):
            a, b = self, other
            return Even(
                -a.c1 * b.c4.conjugate() + a.c2 * b.c2.conjugate(),
                a.c1 * b.c3.conjugate() - a.c2 * b.c1.conjugate(),
                -a.c3 * b.c4.conjugate() + a.c4 * b.c2.conjugate(),
                -a.c4 * b.c1.conjugate() + a.c3 * b.c3.conjugate(),
            )
        if isinstance(other, (int, float)):
            return Odd(other * self.c1, other * self.c2,
                       other * self.c3, other * self.c4)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Odd(other * self.c1, other * self.c2,
                       other * self.c3, other * self.c4)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            inv = 1.0 / other
            return Odd(inv * self.c1, inv * self.c2, inv * self.c3, inv * self.c4)
        return NotImplemented

    # ------------------------------------------------------------------
    # Reverse / adjoint
    # ------------------------------------------------------------------
    @property
    def adjoint(self):
        return Odd(self.c1.conjugate(), self.c3.conjugate(),
                   self.c2.conjugate(), self.c4.conjugate())

    # ------------------------------------------------------------------
    # Grade projection
    # ------------------------------------------------------------------
    def project(self, n: int):
        if n == 1:
            return (self + self.adjoint) / 2
        if n == 3:
            return (self - self.adjoint) / 2
        return Odd.zero()

    # ------------------------------------------------------------------
    # Trace / dot / norm
    # ------------------------------------------------------------------
    def tr(self) -> float:
        return 0.0

    def dot(self, other) -> float:
        if isinstance(other, Odd):
            tmp = (-self.c1 * other.c4.conjugate() + self.c2 * other.c2.conjugate()
                   - self.c4 * other.c1.conjugate() + self.c3 * other.c3.conjugate())
            return tmp.real / 2
        if isinstance(other, Even):
            return 0.0
        return NotImplemented

    def norm(self) -> float:
        return math.sqrt(abs(self.dot(self)))

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def isapprox(self, other, rtol: float = 1e-9, atol: float = 0.0) -> bool:
        if not isinstance(other, Odd):
            return False
        return (cmath.isclose(self.c1, other.c1, rel_tol=rtol, abs_tol=atol) and
                cmath.isclose(self.c2, other.c2, rel_tol=rtol, abs_tol=atol) and
                cmath.isclose(self.c3, other.c3, rel_tol=rtol, abs_tol=atol) and
                cmath.isclose(self.c4, other.c4, rel_tol=rtol, abs_tol=atol))

    def __eq__(self, other):
        if isinstance(other, Odd):
            return (self.c1 == other.c1 and self.c2 == other.c2
                    and self.c3 == other.c3 and self.c4 == other.c4)
        return NotImplemented

    def __hash__(self):
        return hash((self.c1, self.c2, self.c3, self.c4))

    def __repr__(self):
        parts = []
        comps = [
            (self.dot(e1),       "e1"),
            (self.dot(e2),       "e2"),
            (self.dot(e3),       "e3"),
            (self.dot(-f3),      "f3"),
            (self.dot(I4 * e1),  "I4e1"),
            (self.dot(I4 * e2),  "I4e2"),
            (self.dot(I4 * e3),  "I4e3"),
            (self.dot(-I4 * f3), "I4f3"),
        ]
        for val, label in comps:
            if val != 0.0:
                parts.append(f"{val}{label}")
        return " + ".join(parts) if parts else "0"


# ------------------------------------------------------------------
# Helper: wrap a complex scalar as the diagonal Even matrix
# ------------------------------------------------------------------
def _as_even(x: complex) -> Even:
    """Even(x, 0, 0, x) – scalar multiple of the identity matrix."""
    return Even(x, 0 + 0j, 0 + 0j, x)


# ------------------------------------------------------------------
# Basis elements for GA(3,1)
# ------------------------------------------------------------------
# Pauli-matrix even generators
_s1 = Even(0 + 0j, 1 + 0j, 1 + 0j, 0 + 0j)
_s2 = Even(0 + 0j, -1j, 1j, 0 + 0j)
_s3 = Even(1 + 0j, 0 + 0j, 0 + 0j, -1 + 0j)

# Time-like (negative-signature) vector
f3 = Odd(1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j)

# Positive-signature vectors
e1 = _s1 * f3
e2 = _s2 * f3
e3 = _s3 * f3

# Pseudoscalar
I4 = e1 * e2 * e3 * f3

basis = [e1, e2, e3, f3]
