# API

This page documents the Python API exposed by [`simplega`](../../simplega).
It is structured to match the
[upstream API reference](https://monumoltd.github.io/SimpleGA.jl/dev/#API),
adapted for Python idioms and limited to the ported subset.

## Bases

Each algebra module exposes a `basis` list of grade-1 basis vectors and
named attributes for each individual basis element:

```python
from simplega import ga20, ga30, ga31

ga20.basis            # [e1, e2]
ga20.e1, ga20.e2

ga30.basis            # [e1, e2, e3]
ga30.e1, ga30.e2, ga30.e3
ga30.I3               # pseudoscalar e1 e2 e3

ga31.basis            # [e1, e2, e3, f3]
ga31.s1, ga31.s2, ga31.s3   # bivectors si = ei f3 (Pauli matrices)
ga31.I4                     # pseudoscalar
```

There is also a small dispatcher in
[`simplega.basis`](../../simplega/basis.py):

```python
from simplega.basis import basis

basis(2)          # -> ga20.basis
basis(3)          # -> ga30.basis
basis(3, 1)       # -> ga31.basis
basis()           # -> prints supported algebras
```

Constructors `Even(...)` and `Odd(...)` are exposed, but in practice it
is simpler to build multivectors from the basis vectors via arithmetic.

## Arithmetic

The standard Python arithmetic operators are overloaded and behave as
expected:

| Operator | Meaning |
| --- | --- |
| `a + b`, `a - b` | addition / subtraction (same kind only) |
| `-a` | negation |
| `a * b` | geometric product |
| `a / x` | division by a real `x` |
| `a.adjoint` | reverse of `a` (a property, not a call) |

A scalar can be added to or subtracted from an `Even` element; the
scalar enters the grade-0 slot.

Restriction inherited from the Even/Odd split: you cannot add an `Even`
to an `Odd`. Multiplying them is fine — `Even * Odd` and `Odd * Even`
both produce an `Odd`, and `Odd * Odd` produces an `Even`.

```python
from simplega import ga30
e1, e2 = ga30.e1, ga30.e2

a = 0.5 * e1 + e2          # Odd
b = 2.0 * e1 - 3.0 * e2    # Odd
g = a * b                   # Even   (scalar + bivector)
g.adjoint                   # reverse
g / 2.0                     # divide by scalar
```

## Scalar extraction: `tr`, `dot`, `norm`

These are exposed at module level and as methods on each multivector.

```python
from simplega import tr, dot, norm

tr(M)          # scalar (grade-0) part of M, as a Python float
dot(A, B)      # scalar part of (A * B), as a Python float
norm(M)        # sqrt(|dot(M, M)|)
```

Notes that carry over from the upstream package:

* `tr(A * B)` and `dot(A, B)` return the same value, but `dot` skips
  the work needed for the non-scalar parts and is preferred.
* `dot(A, B)` is the scalar part of the geometric product; it is *not*
  the same as the geometric-algebra inner product `A · B` (see
  [Outer and Inner products](#outer-and-inner-products) below).
* `dot(Even, Odd)` is identically zero.

## Projection: `project(M, n)`

`project(M, n)` returns the grade-`n` part of `M`. The result has the
same Python type as `M` (e.g. an `Even` grade-0 part is still an
`Even`, with everything except the scalar slot zero).

```python
from simplega import project, ga30
e1, e2 = ga30.e1, ga30.e2

g = (e1 + e2) * (2 * e1 + 3 * e2)
project(g, 0)     # scalar part of e1 e2's geometric product
project(g, 2)     # bivector part
```

Valid grades per algebra:

| Algebra | `Even` grades | `Odd` grades |
| --- | --- | --- |
| `GA(2,0)` | 0, 2 | 1 |
| `GA(3,0)` | 0, 2 | 1, 3 |
| `GA(3,1)` | 0, 2, 4 | 1, 3 |

## Exponentiation: `exp` and `bivector_exp`

Two exponential variants are provided for `Even` multivectors:

```python
M.exp()              # full even-grade exponential
M.bivector_exp()     # exponential of the grade-2 part only
```

The module-level helper `simplega.bivector_exp(M)` calls
`M.bivector_exp()`. Use `bivector_exp` when you know your input is a
pure bivector (i.e. a generator of a rotation); it is faster and
numerically nicer than the general `exp`.

```python
import math
from simplega import ga30, bivector_exp

# −(π/4) · e1 e2  →  90° rotation around z
B = ga30.Even(0.0, 0.0, 0.0, math.pi / 4)
R = bivector_exp(B)              # unit rotor
print(R.norm())                  # ≈ 1.0
```

## Outer and inner products

As in the upstream package, this library does not define separate
operators for the inner and outer products. Both are projections of
the geometric product:

```python
from simplega import project

inner = project(a * b, 0)        # <a, b>     (scalar)
outer = project(a * b, 2)        # a ^ b      (bivector)
```

This keeps the geometric product as the primary operation and lets the
optimiser handle the rest.

## Further helpers

* `inject(coeffs, basis_elems)` — linear combination of basis
  elements:

  ```python
  from simplega import inject, ga30
  v = inject([1.0, 2.0, 3.0], ga30.basis)   # 1·e1 + 2·e2 + 3·e3
  ```

* `adjoint(a)` — same as `a.adjoint`.
* `isapprox(a, b, rtol=1e-9, atol=0.0)` — approximate equality.
  Cross-type comparisons (e.g. `Even` vs `Odd`) always return `False`.

## Namespaces

Each algebra is a self-contained module, so it is fine to have several
algebras imported at once. Conversions between algebras are not
automatic; the right map depends on the geometric setting (projective
split, conformal split, etc.) and is left to user code.

## Accuracy and display

Floating-point arithmetic is not strictly associative, so terms that
should algebraically vanish often pick up small machine-precision
residues. The current port does not filter these on display; if you
need a tidy printout, post-process the components or use the supplied
visualizations.

---

## See also

* [Overview](overview.md)
* [Per-algebra notes](algebras.md)
* [Upstream Julia API](https://monumoltd.github.io/SimpleGA.jl/dev/#API)
