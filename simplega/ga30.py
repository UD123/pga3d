"""
GA(3,0) implementation.
Python port of SimpleGA.jl/src/core30.jl and src/ga30.jl

Underlying representation: quaternion-like (w, x, y, z) for both grades.

Even(w, x, y, z):
    w = scalar (grade-0)
    x = -e2e3  bivector coefficient
    y = -e3e1  bivector coefficient
    z = -e1e2  bivector coefficient

Odd(w, x, y, z):
    x = e1  vector coefficient
    y = e2  vector coefficient
    z = e3  vector coefficient
    w = e1e2e3 (I3)  trivector coefficient

Basis:
    e1 = Odd(0, 1, 0, 0)
    e2 = Odd(0, 0, 1, 0)
    e3 = Odd(0, 0, 0, 1)
    I3 = Odd(1, 0, 0, 0)   pseudoscalar
"""

import math


class Even:
    """Even-grade element of GA(3,0): scalar + bivectors."""

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------
    @classmethod
    def zero(cls):
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def one(cls):
        return cls(1.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __neg__(self):
        return Even(-self.w, -self.x, -self.y, -self.z)

    def __add__(self, other):
        if isinstance(other, Even):
            return Even(self.w + other.w, self.x + other.x,
                        self.y + other.y, self.z + other.z)
        if isinstance(other, (int, float)):
            return Even(self.w + other, self.x, self.y, self.z)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Even(self.w + other, self.x, self.y, self.z)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Even):
            return Even(self.w - other.w, self.x - other.x,
                        self.y - other.y, self.z - other.z)
        if isinstance(other, (int, float)):
            return Even(self.w - other, self.x, self.y, self.z)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Even(-self.w + other, -self.x, -self.y, -self.z)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Even):
            a, b = self, other
            return Even(
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
                a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
            )
        if isinstance(other, Odd):
            a, b = self, other
            return Odd(
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
                a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
            )
        if isinstance(other, (int, float)):
            return Even(other * self.w, other * self.x,
                        other * self.y, other * self.z)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Even(other * self.w, other * self.x,
                        other * self.y, other * self.z)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            inv = 1.0 / other
            return Even(inv * self.w, inv * self.x, inv * self.y, inv * self.z)
        return NotImplemented

    # ------------------------------------------------------------------
    # Reverse / adjoint  (negate bivector components)
    # ------------------------------------------------------------------
    @property
    def adjoint(self):
        return Even(self.w, -self.x, -self.y, -self.z)

    # ------------------------------------------------------------------
    # Grade projection
    # ------------------------------------------------------------------
    def project(self, n: int):
        if n == 0:
            return Even(self.w, 0.0, 0.0, 0.0)
        if n == 2:
            return Even(0.0, self.x, self.y, self.z)
        return Even.zero()

    # ------------------------------------------------------------------
    # Trace / dot / norm
    # ------------------------------------------------------------------
    def tr(self) -> float:
        return self.w

    def dot(self, other) -> float:
        if isinstance(other, Even):
            return (self.w * other.w - self.x * other.x
                    - self.y * other.y - self.z * other.z)
        if isinstance(other, Odd):
            return 0.0
        return NotImplemented

    def norm(self) -> float:
        return math.sqrt(abs(self.dot(self)))

    # ------------------------------------------------------------------
    # Exponential
    # ------------------------------------------------------------------
    def bivector_exp(self):
        """exp of the grade-2 (bivector) part."""
        a = self.project(2)
        # nrm = sqrt(dot(a, -a))  where dot(biv, biv) = -(x^2+y^2+z^2)
        nrm = math.sqrt(a.x ** 2 + a.y ** 2 + a.z ** 2)
        if nrm == 0.0:
            return Even.one()
        c, s = math.cos(nrm), math.sin(nrm) / nrm
        return Even(c, s * a.x, s * a.y, s * a.z)

    def exp(self):
        """Full even-grade exponential."""
        R = self.bivector_exp()
        return R if self.w == 0.0 else math.exp(self.w) * R

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def isapprox(self, other, rtol: float = 1e-9, atol: float = 0.0) -> bool:
        if not isinstance(other, Even):
            return False
        return (math.isclose(self.w, other.w, rel_tol=rtol, abs_tol=atol) and
                math.isclose(self.x, other.x, rel_tol=rtol, abs_tol=atol) and
                math.isclose(self.y, other.y, rel_tol=rtol, abs_tol=atol) and
                math.isclose(self.z, other.z, rel_tol=rtol, abs_tol=atol))

    def __eq__(self, other):
        if isinstance(other, Even):
            return (self.w == other.w and self.x == other.x
                    and self.y == other.y and self.z == other.z)
        return NotImplemented

    def __hash__(self):
        return hash((self.w, self.x, self.y, self.z))

    def __repr__(self):
        parts = []
        if self.w != 0.0:
            parts.append(str(self.w))
        if self.x != 0.0:
            parts.append(f"{-self.x}e2e3")
        if self.y != 0.0:
            parts.append(f"{-self.y}e3e1")
        if self.z != 0.0:
            parts.append(f"{-self.z}e1e2")
        return " + ".join(parts) if parts else "0"


class Odd:
    """Odd-grade element of GA(3,0): vectors + trivector."""

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------
    @classmethod
    def zero(cls):
        return cls(0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __neg__(self):
        return Odd(-self.w, -self.x, -self.y, -self.z)

    def __add__(self, other):
        if isinstance(other, Odd):
            return Odd(self.w + other.w, self.x + other.x,
                       self.y + other.y, self.z + other.z)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Odd):
            return Odd(self.w - other.w, self.x - other.x,
                       self.y - other.y, self.z - other.z)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Even):
            a, b = self, other
            return Odd(
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
                a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
            )
        if isinstance(other, Odd):
            a, b = self, other
            return Even(
                -a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z,
                -a.w * b.x - a.x * b.w - a.y * b.z + a.z * b.y,
                -a.w * b.y - a.y * b.w - a.z * b.x + a.x * b.z,
                -a.w * b.z - a.z * b.w - a.x * b.y + a.y * b.x,
            )
        if isinstance(other, (int, float)):
            return Odd(other * self.w, other * self.x,
                       other * self.y, other * self.z)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Odd(other * self.w, other * self.x,
                       other * self.y, other * self.z)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            inv = 1.0 / other
            return Odd(inv * self.w, inv * self.x, inv * self.y, inv * self.z)
        return NotImplemented

    # ------------------------------------------------------------------
    # Reverse / adjoint  (negate trivector)
    # ------------------------------------------------------------------
    @property
    def adjoint(self):
        return Odd(-self.w, self.x, self.y, self.z)

    # ------------------------------------------------------------------
    # Grade projection
    # ------------------------------------------------------------------
    def project(self, n: int):
        if n == 1:
            return Odd(0.0, self.x, self.y, self.z)
        if n == 3:
            return Odd(self.w, 0.0, 0.0, 0.0)
        return Odd.zero()

    # ------------------------------------------------------------------
    # Trace / dot / norm
    # ------------------------------------------------------------------
    def tr(self) -> float:
        return 0.0

    def dot(self, other) -> float:
        if isinstance(other, Odd):
            return (-self.w * other.w + self.x * other.x
                    + self.y * other.y + self.z * other.z)
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
        return (math.isclose(self.w, other.w, rel_tol=rtol, abs_tol=atol) and
                math.isclose(self.x, other.x, rel_tol=rtol, abs_tol=atol) and
                math.isclose(self.y, other.y, rel_tol=rtol, abs_tol=atol) and
                math.isclose(self.z, other.z, rel_tol=rtol, abs_tol=atol))

    def __eq__(self, other):
        if isinstance(other, Odd):
            return (self.w == other.w and self.x == other.x
                    and self.y == other.y and self.z == other.z)
        return NotImplemented

    def __hash__(self):
        return hash((self.w, self.x, self.y, self.z))

    def __repr__(self):
        parts = []
        if self.x != 0.0:
            parts.append(f"{self.x}e1")
        if self.y != 0.0:
            parts.append(f"{self.y}e2")
        if self.z != 0.0:
            parts.append(f"{self.z}e3")
        if self.w != 0.0:
            parts.append(f"{self.w}e123")
        return " + ".join(parts) if parts else "0"


# ------------------------------------------------------------------
# Basis elements for GA(3,0)
# ------------------------------------------------------------------
e1 = Odd(0.0, 1.0, 0.0, 0.0)   # grade-1 basis vector e1
e2 = Odd(0.0, 0.0, 1.0, 0.0)   # grade-1 basis vector e2
e3 = Odd(0.0, 0.0, 0.0, 1.0)   # grade-1 basis vector e3
I3 = Odd(1.0, 0.0, 0.0, 0.0)   # pseudoscalar e1e2e3

basis = [e1, e2, e3]
