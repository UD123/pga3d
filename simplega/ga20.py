"""
GA(2,0) implementation.
Python port of SimpleGA.jl/src/core20.jl and src/ga20.jl

Underlying representation: complex numbers.
    Even(c1)  ->  scalar + pseudoscalar  (grades 0 and 2)
    Odd(c1)   ->  vectors                (grade 1)

Mapping:
    Even(a + bi)  represents  a + b*e1e2
    Odd(a + bi)   represents  a*e1 + b*e2
"""

import cmath
import math


class Even:
    """
    Even-grade element of GA(2,0).
    Stored as a single complex number c1.
        real(c1) = scalar (grade-0) coefficient
        imag(c1) = I2 = e1e2 (grade-2) coefficient
    """

    __slots__ = ("c1",)

    def __init__(self, c1=0 + 0j):
        self.c1 = complex(c1)

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------
    @classmethod
    def zero(cls):
        return cls(0 + 0j)

    @classmethod
    def one(cls):
        return cls(1 + 0j)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __neg__(self):
        return Even(-self.c1)

    def __add__(self, other):
        if isinstance(other, Even):
            return Even(self.c1 + other.c1)
        if isinstance(other, (int, float)):
            return Even(self.c1 + other)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return Even(self.c1 + other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Even):
            return Even(self.c1 - other.c1)
        if isinstance(other, (int, float)):
            return Even(self.c1 - other)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return Even(-self.c1 + other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Even):
            # Even * Even -> Even: plain complex product
            return Even(self.c1 * other.c1)
        if isinstance(other, Odd):
            # Even * Odd -> Odd: conj(a) * b
            return Odd(self.c1.conjugate() * other.c1)
        if isinstance(other, (int, float)):
            return Even(other * self.c1)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Even(other * self.c1)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Even):
            return Even(self.c1 / other.c1)
        if isinstance(other, (int, float)):
            return Even(self.c1 / other)
        return NotImplemented

    # ------------------------------------------------------------------
    # Reverse / adjoint
    # ------------------------------------------------------------------
    @property
    def adjoint(self):
        """Reverse: conjugate c1."""
        return Even(self.c1.conjugate())

    # ------------------------------------------------------------------
    # Grade projection
    # ------------------------------------------------------------------
    def project(self, n: int):
        if n == 0:
            return Even(self.c1.real + 0j)
        if n == 2:
            return Even(self.c1.imag * 1j)
        return Even.zero()

    # ------------------------------------------------------------------
    # Trace / dot / norm
    # ------------------------------------------------------------------
    def tr(self) -> float:
        return self.c1.real

    def dot(self, other) -> float:
        if isinstance(other, Even):
            return (self.c1 * other.c1).real
        if isinstance(other, Odd):
            return 0.0
        return NotImplemented

    def norm(self) -> float:
        return math.sqrt(abs(self.dot(self)))

    # ------------------------------------------------------------------
    # Exponential
    # ------------------------------------------------------------------
    def exp(self):
        """Full exponential: exp(c1)."""
        return Even(cmath.exp(self.c1))

    def bivector_exp(self):
        """Exponential of grade-2 part only: exp(i * imag(c1))."""
        return Even(cmath.exp(1j * self.c1.imag))

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    def isapprox(self, other, rtol: float = 1e-9, atol: float = 0.0) -> bool:
        if not isinstance(other, Even):
            return False
        return cmath.isclose(self.c1, other.c1, rel_tol=rtol, abs_tol=atol)

    def __eq__(self, other):
        if isinstance(other, Even):
            return self.c1 == other.c1
        return NotImplemented

    def __hash__(self):
        return hash(self.c1)

    def __repr__(self):
        parts = []
        if self.c1.real != 0.0:
            parts.append(str(self.c1.real))
        if self.c1.imag != 0.0:
            parts.append(f"{self.c1.imag}I2")
        return " + ".join(parts) if parts else "0"


class Odd:
    """
    Odd-grade element of GA(2,0).
    Stored as a single complex number c1.
        real(c1) = e1 coefficient
        imag(c1) = e2 coefficient
    """

    __slots__ = ("c1",)

    def __init__(self, c1=0 + 0j):
        self.c1 = complex(c1)

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------
    @classmethod
    def zero(cls):
        return cls(0 + 0j)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __neg__(self):
        return Odd(-self.c1)

    def __add__(self, other):
        if isinstance(other, Odd):
            return Odd(self.c1 + other.c1)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Odd):
            return Odd(self.c1 - other.c1)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Even):
            # Odd * Even -> Odd: a * b
            return Odd(self.c1 * other.c1)
        if isinstance(other, Odd):
            # Odd * Odd -> Even: conj(a) * b
            return Even(self.c1.conjugate() * other.c1)
        if isinstance(other, (int, float)):
            return Odd(other * self.c1)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Odd(other * self.c1)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Odd(self.c1 / other)
        return NotImplemented

    # ------------------------------------------------------------------
    # Reverse / adjoint
    # ------------------------------------------------------------------
    @property
    def adjoint(self):
        """Reverse of Odd in GA(2,0) is itself."""
        return Odd(self.c1)

    # ------------------------------------------------------------------
    # Grade projection
    # ------------------------------------------------------------------
    def project(self, n: int):
        return self if n == 1 else Odd.zero()

    # ------------------------------------------------------------------
    # Trace / dot / norm
    # ------------------------------------------------------------------
    def tr(self) -> float:
        return 0.0

    def dot(self, other) -> float:
        if isinstance(other, Odd):
            return (self.c1.conjugate() * other.c1).real
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
        return cmath.isclose(self.c1, other.c1, rel_tol=rtol, abs_tol=atol)

    def __eq__(self, other):
        if isinstance(other, Odd):
            return self.c1 == other.c1
        return NotImplemented

    def __hash__(self):
        return hash(self.c1)

    def __repr__(self):
        parts = []
        if self.c1.real != 0.0:
            parts.append(f"{self.c1.real}e1")
        if self.c1.imag != 0.0:
            parts.append(f"{self.c1.imag}e2")
        return " + ".join(parts) if parts else "0"


# ------------------------------------------------------------------
# Basis elements for GA(2,0)
# ------------------------------------------------------------------
e1 = Odd(1 + 0j)   # grade-1 basis vector e1
e2 = Odd(0 + 1j)   # grade-1 basis vector e2
I2 = Even(0 + 1j)  # pseudoscalar e1e2

basis = [e1, e2]
