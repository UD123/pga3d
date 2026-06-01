"""
Generic Clifford / geometric algebra over signature (p, q, r).

Inspired by the API of ganja.js (https://github.com/enkimute/ganja.js) —
in particular the `Algebra(p, q, r)` factory pattern — but implemented
from scratch in Python.

The implementation is dense: a multivector is stored as a length-2**n
numpy array indexed by basis-blade bitmasks. This is the simplest
correct representation; it is not optimised for large n.

Conventions
-----------
* `n = p + q + r` total dimensions.
* Bit `i` of a blade index corresponds to basis vector `e_{i+1}`
  (1-based names, 0-based bits).
* Signature: bits `[0, p)`  square to +1,
              bits `[p, p+q)` square to −1,
              bits `[p+q, n)` square to  0.
* Canonical blade order is increasing bit index: e1 e2 e3 …
* Grade of blade `b` is `popcount(b)`.
"""

from __future__ import annotations

from typing import List, Sequence

import math
import numpy as np


def _popcount(x: int) -> int:
    return bin(x).count("1")


def _canonical_sign(a: int, b: int) -> int:
    """
    Sign incurred by reordering the concatenation of the basis vectors of
    blades `a` and `b` into canonical (increasing) order, ignoring metric.

    Counts the number of swaps needed: for each bit set in `b`, count
    how many bits of `a` lie strictly above it.
    """
    swaps = 0
    a_shift = a >> 1
    while a_shift:
        swaps += _popcount(a_shift & b)
        a_shift >>= 1
    return -1 if (swaps & 1) else 1


def _grade_indices(n: int) -> List[List[int]]:
    """Return a list `grades` where `grades[k]` is the list of blade
    bitmasks of grade `k`."""
    out: List[List[int]] = [[] for _ in range(n + 1)]
    for b in range(1 << n):
        out[_popcount(b)].append(b)
    return out


def _blade_name(b: int, n: int) -> str:
    """Human-readable basis-blade name, e.g. 5 -> 'e13' (bits 0 and 2)."""
    if b == 0:
        return "1"
    parts = []
    for i in range(n):
        if (b >> i) & 1:
            parts.append(str(i + 1))
    return "e" + "".join(parts)


# ---------------------------------------------------------------------------
# Algebra factory
# ---------------------------------------------------------------------------

class _AlgebraSpec:
    """Internal: cached per-algebra tables (signature, sign matrix, etc.)."""

    __slots__ = (
        "p", "q", "r", "n", "dim",
        "metric", "signs", "grades", "grade_of", "names",
    )

    def __init__(self, p: int, q: int, r: int):
        self.p = int(p)
        self.q = int(q)
        self.r = int(r)
        self.n = self.p + self.q + self.r
        self.dim = 1 << self.n

        # Per-bit metric: +1 for first p, -1 for next q, 0 for next r.
        metric = np.empty(self.n, dtype=np.int8)
        metric[: self.p] = 1
        metric[self.p : self.p + self.q] = -1
        metric[self.p + self.q :] = 0
        self.metric = metric

        # Sign matrix: signs[a, b] in {-1, 0, +1} such that
        #     e_a * e_b  =  signs[a, b] * e_{a XOR b}
        # signs[a, b] = 0 iff a and b share a null bit.
        dim = self.dim
        signs = np.zeros((dim, dim), dtype=np.int8)
        for a in range(dim):
            for b in range(dim):
                common = a & b
                # metric contribution
                metric_sign = 1
                ok = True
                bit = 0
                while common:
                    if common & 1:
                        m = int(metric[bit])
                        if m == 0:
                            ok = False
                            break
                        metric_sign *= m
                    common >>= 1
                    bit += 1
                if not ok:
                    continue
                signs[a, b] = _canonical_sign(a, b) * metric_sign
        self.signs = signs

        self.grades = _grade_indices(self.n)
        self.grade_of = np.array(
            [_popcount(b) for b in range(dim)], dtype=np.int8
        )
        self.names = [_blade_name(b, self.n) for b in range(dim)]


_SPEC_CACHE: dict[tuple[int, int, int], _AlgebraSpec] = {}


def _get_spec(p: int, q: int, r: int) -> _AlgebraSpec:
    key = (int(p), int(q), int(r))
    spec = _SPEC_CACHE.get(key)
    if spec is None:
        spec = _AlgebraSpec(*key)
        _SPEC_CACHE[key] = spec
    return spec


# ---------------------------------------------------------------------------
# Multivector
# ---------------------------------------------------------------------------

class Multivector:
    """
    Element of a Clifford algebra `Algebra(p, q, r)`.

    Stored densely as a length-`2**n` numpy array of `float64` coefficients,
    indexed by basis-blade bitmask.

    Construct via the `Algebra(...)` factory or by arithmetic on its
    basis vectors. Direct construction is allowed for advanced use:

        Multivector(spec, coeffs)
    """

    __slots__ = ("_spec", "_v")

    def __init__(self, spec: _AlgebraSpec, coeffs):
        self._spec = spec
        v = np.zeros(spec.dim, dtype=np.float64)
        if coeffs is not None:
            arr = np.asarray(coeffs, dtype=np.float64).ravel()
            if arr.size != spec.dim:
                raise ValueError(
                    f"coefficient vector has length {arr.size}, "
                    f"expected {spec.dim} for Algebra({spec.p},{spec.q},{spec.r})"
                )
            v[:] = arr
        self._v = v

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def _from_array(cls, spec: _AlgebraSpec, arr: np.ndarray) -> "Multivector":
        out = cls.__new__(cls)
        out._spec = spec
        out._v = arr
        return out

    @classmethod
    def scalar(cls, spec: _AlgebraSpec, value: float) -> "Multivector":
        v = np.zeros(spec.dim, dtype=np.float64)
        v[0] = float(value)
        return cls._from_array(spec, v)

    @classmethod
    def basis(cls, spec: _AlgebraSpec, blade: int) -> "Multivector":
        v = np.zeros(spec.dim, dtype=np.float64)
        v[blade] = 1.0
        return cls._from_array(spec, v)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def algebra(self):
        return Algebra(self._spec.p, self._spec.q, self._spec.r)

    @property
    def coeffs(self) -> np.ndarray:
        """Return a copy of the underlying coefficient vector."""
        return self._v.copy()

    def __len__(self) -> int:
        return self._spec.dim

    def __getitem__(self, blade: int) -> float:
        return float(self._v[blade])

    # ------------------------------------------------------------------
    # Same-algebra checks
    # ------------------------------------------------------------------
    def _check(self, other: "Multivector") -> None:
        if other._spec is not self._spec:
            s, o = self._spec, other._spec
            if (s.p, s.q, s.r) != (o.p, o.q, o.r):
                raise ValueError(
                    "multivectors live in different algebras: "
                    f"({s.p},{s.q},{s.r}) vs ({o.p},{o.q},{o.r})"
                )

    # ------------------------------------------------------------------
    # Arithmetic — addition / subtraction / negation
    # ------------------------------------------------------------------
    def __pos__(self) -> "Multivector":
        return self

    def __neg__(self) -> "Multivector":
        return Multivector._from_array(self._spec, -self._v)

    def __add__(self, other):
        if isinstance(other, Multivector):
            self._check(other)
            return Multivector._from_array(self._spec, self._v + other._v)
        if isinstance(other, (int, float)):
            out = self._v.copy()
            out[0] += float(other)
            return Multivector._from_array(self._spec, out)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Multivector):
            self._check(other)
            return Multivector._from_array(self._spec, self._v - other._v)
        if isinstance(other, (int, float)):
            out = self._v.copy()
            out[0] -= float(other)
            return Multivector._from_array(self._spec, out)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            out = -self._v.copy()
            out[0] += float(other)
            return Multivector._from_array(self._spec, out)
        return NotImplemented

    # ------------------------------------------------------------------
    # Arithmetic — geometric product
    # ------------------------------------------------------------------
    def __mul__(self, other):
        if isinstance(other, Multivector):
            self._check(other)
            return Multivector._from_array(
                self._spec, _gp(self._spec, self._v, other._v)
            )
        if isinstance(other, (int, float)):
            return Multivector._from_array(self._spec, self._v * float(other))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Multivector._from_array(self._spec, self._v * float(other))
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Multivector._from_array(self._spec, self._v / float(other))
        if isinstance(other, Multivector):
            return self * other.inverse()
        return NotImplemented

    # ------------------------------------------------------------------
    # Wedge (outer) and inner products
    #
    # XOR `^` is used for the wedge product (note: lower precedence than
    # `*` in Python — always parenthesise).
    # `|` is the symmetric inner product.
    # ------------------------------------------------------------------
    def __xor__(self, other):
        if isinstance(other, Multivector):
            self._check(other)
            return Multivector._from_array(
                self._spec, _wedge(self._spec, self._v, other._v)
            )
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, Multivector):
            self._check(other)
            return Multivector._from_array(
                self._spec, _inner(self._spec, self._v, other._v)
            )
        return NotImplemented

    # ------------------------------------------------------------------
    # Involutions
    # ------------------------------------------------------------------
    def reverse(self) -> "Multivector":
        """Reverse `~A`: per blade of grade g, sign = (−1)^(g(g−1)/2)."""
        g = self._spec.grade_of
        signs = np.where(((g * (g - 1) // 2) & 1) == 1, -1.0, 1.0)
        return Multivector._from_array(self._spec, self._v * signs)

    def __invert__(self) -> "Multivector":
        return self.reverse()

    def grade_involution(self) -> "Multivector":
        """Grade involution: sign = (−1)^g per blade."""
        g = self._spec.grade_of
        signs = np.where((g & 1) == 1, -1.0, 1.0)
        return Multivector._from_array(self._spec, self._v * signs)

    def conjugate(self) -> "Multivector":
        """Clifford conjugation: reverse ∘ grade-involution."""
        return self.reverse().grade_involution()

    # ------------------------------------------------------------------
    # Grade tools
    # ------------------------------------------------------------------
    def grade(self, k: int) -> "Multivector":
        """Project onto grade k."""
        out = np.zeros_like(self._v)
        idx = self._spec.grades[k] if 0 <= k <= self._spec.n else []
        for b in idx:
            out[b] = self._v[b]
        return Multivector._from_array(self._spec, out)

    def grades(self) -> List[int]:
        """Return the sorted list of grades present in this multivector."""
        present = set()
        for b in range(self._spec.dim):
            if self._v[b] != 0.0:
                present.add(int(self._spec.grade_of[b]))
        return sorted(present)

    # ------------------------------------------------------------------
    # Scalar / norm / inverse
    # ------------------------------------------------------------------
    def scalar_part(self) -> float:
        return float(self._v[0])

    def norm_squared(self) -> float:
        """`<A * ~A>_0` — may be negative in mixed-signature algebras."""
        return float(_gp(self._spec, self._v, self.reverse()._v)[0])

    def norm(self) -> float:
        return math.sqrt(abs(self.norm_squared()))

    def normalized(self) -> "Multivector":
        n = self.norm()
        if n == 0.0:
            raise ZeroDivisionError("cannot normalise a zero multivector")
        return Multivector._from_array(self._spec, self._v / n)

    def inverse(self) -> "Multivector":
        """
        Inverse via `A^{-1} = ~A / <A ~A>_0` when that scalar is non-zero.
        Falls back to a small dense solve otherwise.
        """
        ns = self.norm_squared()
        if abs(ns) > 1e-14:
            return Multivector._from_array(self._spec, self.reverse()._v / ns)
        # Fallback: solve A * x = 1 in the dense 2^n representation.
        M = _left_mult_matrix(self._spec, self._v)
        rhs = np.zeros(self._spec.dim)
        rhs[0] = 1.0
        try:
            x = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError as exc:
            raise ZeroDivisionError("multivector is not invertible") from exc
        return Multivector._from_array(self._spec, x)

    # ------------------------------------------------------------------
    # Duality
    # ------------------------------------------------------------------
    def dual(self) -> "Multivector":
        """
        Poincaré dual: `A* = A * I^{-1}` where `I` is the pseudoscalar
        of grade `n`. For degenerate (PGA) algebras this still works as
        long as `I` is invertible in the non-degenerate subalgebra; for a
        purely degenerate component we fall back to the basis-blade
        complement (bit-flip), which matches ganja.js's `Dual` for PGA.
        """
        spec = self._spec
        if spec.r == 0:
            # Non-degenerate: use the pseudoscalar directly.
            I = np.zeros(spec.dim)
            I[spec.dim - 1] = 1.0
            I_inv = Multivector._from_array(spec, I).inverse()
            return Multivector._from_array(
                spec, _gp(spec, self._v, I_inv._v)
            )
        # PGA-style: blade complement A_b  ->  A_{(2^n - 1) ^ b}
        full = spec.dim - 1
        out = np.zeros_like(self._v)
        for b in range(spec.dim):
            out[full ^ b] = self._v[b]
        return Multivector._from_array(spec, out)

    def undual(self) -> "Multivector":
        spec = self._spec
        if spec.r == 0:
            I = np.zeros(spec.dim)
            I[spec.dim - 1] = 1.0
            return Multivector._from_array(
                spec, _gp(spec, self._v, I)
            )
        full = spec.dim - 1
        out = np.zeros_like(self._v)
        for b in range(spec.dim):
            out[full ^ b] = self._v[b]
        return Multivector._from_array(spec, out)

    # ------------------------------------------------------------------
    # Exponential (series fallback; exact for pure bivectors)
    # ------------------------------------------------------------------
    def exp(self) -> "Multivector":
        """
        Exponential of a multivector. Optimised when `self` is a pure
        bivector whose square is a scalar (the common rotor / motor
        case); falls back to a truncated power series otherwise.
        """
        spec = self._spec
        # Test for "pure bivector with scalar square".
        only_grade2 = all(
            (g == 2) or (self._v[bidx] == 0.0)
            for g, blades in enumerate(spec.grades) for bidx in blades
        )
        if only_grade2:
            sq = _gp(spec, self._v, self._v)
            sq_is_scalar = np.all(sq[1:] == 0.0) or np.allclose(sq[1:], 0.0)
            if sq_is_scalar:
                s = float(sq[0])
                if s < -1e-15:                       # negative square → rot
                    a = math.sqrt(-s)
                    c, sn = math.cos(a), math.sin(a) / a
                    out = np.zeros_like(self._v)
                    out[0] = c
                    out += sn * self._v
                    return Multivector._from_array(spec, out)
                if s > 1e-15:                        # positive square → boost
                    a = math.sqrt(s)
                    c, sn = math.cosh(a), math.sinh(a) / a
                    out = np.zeros_like(self._v)
                    out[0] = c
                    out += sn * self._v
                    return Multivector._from_array(spec, out)
                # null square → 1 + A
                out = self._v.copy()
                out[0] += 1.0
                return Multivector._from_array(spec, out)

        # Generic series: scale to small norm, then square back.
        nrm = float(np.max(np.abs(self._v)))
        scale = 1
        x = self
        if nrm > 0.5:
            scale = 1
            while nrm > 0.5:
                nrm /= 2.0
                scale *= 2
            x = self * (1.0 / scale)
        out = Multivector.scalar(spec, 1.0)
        term = Multivector.scalar(spec, 1.0)
        for k in range(1, 32):
            term = term * x * (1.0 / k)
            out = out + term
        # Repeated squaring back to original.
        s = scale
        while s > 1:
            out = out * out
            s //= 2
        return out

    # ------------------------------------------------------------------
    # Sandwich product:  R . A  →  R A ~R
    # ------------------------------------------------------------------
    def __matmul__(self, other):
        """`R @ A` computes the sandwich product  R A ~R."""
        if isinstance(other, Multivector):
            self._check(other)
            return self * other * self.reverse()
        return NotImplemented

    # ------------------------------------------------------------------
    # Equality / approximate equality
    # ------------------------------------------------------------------
    def __eq__(self, other) -> bool:
        if isinstance(other, Multivector):
            return (other._spec is self._spec or
                    (self._spec.p, self._spec.q, self._spec.r) ==
                    (other._spec.p, other._spec.q, other._spec.r)) \
                   and np.array_equal(self._v, other._v)
        if isinstance(other, (int, float)):
            return self._v[0] == float(other) and not np.any(self._v[1:])
        return NotImplemented

    def __hash__(self):
        return hash((self._spec.p, self._spec.q, self._spec.r,
                     self._v.tobytes()))

    def isclose(self, other: "Multivector",
                rtol: float = 1e-9, atol: float = 1e-12) -> bool:
        self._check(other)
        return bool(np.allclose(self._v, other._v, rtol=rtol, atol=atol))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        names = self._spec.names
        terms = []
        for b in range(self._spec.dim):
            c = float(self._v[b])
            if c == 0.0:
                continue
            if b == 0:
                terms.append(f"{c:g}")
            else:
                if c == 1.0:
                    terms.append(names[b])
                elif c == -1.0:
                    terms.append(f"-{names[b]}")
                else:
                    terms.append(f"{c:g}*{names[b]}")
        body = " + ".join(terms).replace("+ -", "- ") if terms else "0"
        s = self._spec
        return f"Multivector[{s.p},{s.q},{s.r}]({body})"


# ---------------------------------------------------------------------------
# Product implementations (numpy-vectorised over the sign table)
# ---------------------------------------------------------------------------

def _gp(spec: _AlgebraSpec, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geometric product of dense coefficient vectors `a` and `b`."""
    out = np.zeros(spec.dim, dtype=np.float64)
    # Outer product of coefficient vectors, then scatter into output by
    # blade XOR with the precomputed sign matrix.
    nz_a = np.nonzero(a)[0]
    nz_b = np.nonzero(b)[0]
    signs = spec.signs
    for i in nz_a:
        ai = a[i]
        srow = signs[i]
        for j in nz_b:
            s = srow[j]
            if s:
                out[i ^ j] += s * ai * b[j]
    return out


def _wedge(spec: _AlgebraSpec, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Outer (wedge) product — geometric product restricted to disjoint blades."""
    out = np.zeros(spec.dim, dtype=np.float64)
    nz_a = np.nonzero(a)[0]
    nz_b = np.nonzero(b)[0]
    signs = spec.signs
    for i in nz_a:
        ai = a[i]
        srow = signs[i]
        for j in nz_b:
            if i & j:
                continue
            s = srow[j]
            if s:
                out[i ^ j] += s * ai * b[j]
    return out


def _inner(spec: _AlgebraSpec, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Symmetric inner product (Hestenes "fat dot"):
        (A | B)_{|g_a - g_b|}  =  grade-projection of A * B
    """
    out = np.zeros(spec.dim, dtype=np.float64)
    grade_of = spec.grade_of
    signs = spec.signs
    nz_a = np.nonzero(a)[0]
    nz_b = np.nonzero(b)[0]
    for i in nz_a:
        gi = int(grade_of[i])
        ai = a[i]
        srow = signs[i]
        for j in nz_b:
            s = srow[j]
            if not s:
                continue
            ij = i ^ j
            if int(grade_of[ij]) == abs(gi - int(grade_of[j])):
                out[ij] += s * ai * b[j]
    return out


def _left_mult_matrix(spec: _AlgebraSpec, a: np.ndarray) -> np.ndarray:
    """Dense matrix `L` such that `L @ b == coeffs(a * b)` for any `b`."""
    dim = spec.dim
    L = np.zeros((dim, dim), dtype=np.float64)
    signs = spec.signs
    for i in range(dim):
        ai = a[i]
        if ai == 0.0:
            continue
        srow = signs[i]
        for j in range(dim):
            s = srow[j]
            if s:
                L[i ^ j, j] += s * ai
    return L


# ---------------------------------------------------------------------------
# Algebra factory wrapper
# ---------------------------------------------------------------------------

class Algebra:
    """
    Factory for a Clifford / geometric algebra of signature (p, q, r).

    Usage
    -----
        A = Algebra(3, 0, 1)            # 3-D projective GA
        e0, e1, e2, e3 = A.basis_vectors()
        I = A.pseudoscalar()
        point = e0 + 2 * e1             # any multivector

        # Construct a rotor as exp of a bivector and apply it
        R = (math.pi / 4 * (e1 ^ e2)).exp()
        rotated = R @ point             # sandwich product
    """

    __slots__ = ("_spec",)

    def __init__(self, p: int = 0, q: int = 0, r: int = 0):
        self._spec = _get_spec(p, q, r)

    # ------------------------------------------------------------------
    # Algebra-level info
    # ------------------------------------------------------------------
    @property
    def p(self) -> int: return self._spec.p
    @property
    def q(self) -> int: return self._spec.q
    @property
    def r(self) -> int: return self._spec.r
    @property
    def n(self) -> int: return self._spec.n
    @property
    def dim(self) -> int: return self._spec.dim

    @property
    def signature(self) -> tuple[int, int, int]:
        return (self._spec.p, self._spec.q, self._spec.r)

    @property
    def blade_names(self) -> list[str]:
        return list(self._spec.names)

    # ------------------------------------------------------------------
    # Element constructors
    # ------------------------------------------------------------------
    def scalar(self, value: float = 1.0) -> Multivector:
        return Multivector.scalar(self._spec, value)

    def zero(self) -> Multivector:
        return Multivector(self._spec, None)

    def basis_vectors(self) -> list[Multivector]:
        """Return [e_1, e_2, ..., e_n] as a list of Multivectors."""
        return [Multivector.basis(self._spec, 1 << i) for i in range(self._spec.n)]

    def basis_blades(self) -> list[Multivector]:
        """Return all 2**n basis blades in canonical order."""
        return [Multivector.basis(self._spec, b) for b in range(self._spec.dim)]

    def pseudoscalar(self) -> Multivector:
        return Multivector.basis(self._spec, self._spec.dim - 1)

    def multivector(self, coeffs: Sequence[float]) -> Multivector:
        """Construct a multivector from a dense coefficient list of length 2**n."""
        return Multivector(self._spec, coeffs)

    # ------------------------------------------------------------------
    # Hashable so `Algebra(3,0,1) == Algebra(3,0,1)` works as expected
    # ------------------------------------------------------------------
    def __eq__(self, other) -> bool:
        return isinstance(other, Algebra) and other._spec is self._spec

    def __hash__(self) -> int:
        return hash(("Algebra", self._spec.p, self._spec.q, self._spec.r))

    def __repr__(self) -> str:
        s = self._spec
        return f"Algebra({s.p}, {s.q}, {s.r})"
