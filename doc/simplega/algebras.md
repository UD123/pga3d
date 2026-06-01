# Algebras

Per-algebra notes for the ported subset. The structure mirrors the
[upstream "Core Algebras" page](https://monumoltd.github.io/SimpleGA.jl/dev/#The-Core-Algebras),
but only covers what the Python port currently implements.

## GA(2,0)

* Module: [`simplega.ga20`](../../simplega/ga20.py)
* Basis: `simplega.ga20.basis  →  [e1, e2]`
* Pseudoscalar: `I2 = e1 * e2`

`GA(2,0)` is the geometric algebra of the Euclidean plane.

**Representation.** Even elements `a + b·e1e2` are stored as a single
Python `complex` (`real` = scalar, `imag` = `I2` coefficient). Odd
elements (vectors `a·e1 + b·e2`) are stored the same way under the map
`v ↦ e1 · v`.

**Extra structure.** Because `Even` is just a complex number it forms
a division algebra; the port exposes:

```python
Even.exp()              # plain cmath.exp on the underlying complex
Even.bivector_exp()     # exp restricted to the grade-2 part
A / B                   # division for two Even elements
```

`Odd / Odd` division is not provided — divide by an `Even` instead.

## GA(3,0)

* Module: [`simplega.ga30`](../../simplega/ga30.py)
* Basis: `simplega.ga30.basis  →  [e1, e2, e3]`
* Pseudoscalar: `I3 = e1 * e2 * e3`

`GA(3,0)` is the geometric algebra of Euclidean 3-space; its even
subalgebra is isomorphic to the quaternions.

**Representation.** Both `Even` and `Odd` carry four real components
`(w, x, y, z)`:

* `Even(w, x, y, z)` = `w  +  x·(−e2e3) + y·(−e3e1) + z·(−e1e2)`
* `Odd(w, x, y, z)`  = `w·I3 + x·e1 + y·e2 + z·e3`

The geometric product on `Even` is the quaternion product; the
odd → even map is multiplication by `I3`.

**Sandwich rotations.** Build a rotor `R = exp(−θ/2 · B)` where `B` is
a unit bivector encoding the rotation plane, then rotate a vector `v`
with the sandwich product:

```python
import math
from simplega import ga30, bivector_exp
e1, e2, e3 = ga30.e1, ga30.e2, ga30.e3

# 90° rotation in the e1-e2 plane (around z)
B = ga30.Even(0.0, 0.0, 0.0, math.pi / 4)
R = bivector_exp(B)
v = 0.8 * e1 + 0.2 * e2 + 0.5 * e3
v_rot = R * v * R.adjoint
```

See the dedicated visualizer
[`simplega/ga30_visualization.py`](../../simplega/ga30_visualization.py)
for an interactive tour.

## GA(3,1)

* Module: [`simplega.ga31`](../../simplega/ga31.py)
* Basis: `simplega.ga31.basis  →  [e1, e2, e3, f3]`
* Signature: `e1² = e2² = e3² = +1`, `f3² = −1`
* Pseudoscalar: `I4 = e1 e2 e3 f3`

`GA(3,1)` shares its representation with the spacetime algebra `STA =
G(1,3)`, with a few sign flips. It is useful as the conformal algebra
for the 2-D plane (where `e1`, `e2` have positive square and a null
basis is built from `e3 ± f3`).

**Representation.** Both `Even` and `Odd` carry four `complex`
components `(c1, c2, c3, c4)` representing a 2×2 complex matrix

```
[[c1, c2],
 [c3, c4]]
```

The exposed Pauli-style bivectors are:

```python
ga31.s1 = ga31.Even(0,  1,  1,  0)    # σ₁  =  e1 * f3
ga31.s2 = ga31.Even(0, -j,  j,  0)    # σ₂  =  e2 * f3
ga31.s3 = ga31.Even(1,  0,  0, -1)    # σ₃  =  e3 * f3
```

The odd → even map is right multiplication by `f3`. The grade-2 part
of an `Even` is obtained from `(A − A†) / 2`; the grade-4
(pseudoscalar) part comes from the imaginary part of the trace.

## Quaternions

* Module: [`simplega.quaternions`](../../simplega/quaternions.py)

A standalone quaternion implementation `Quaternion(w, x, y, z)`,
provided for direct comparison with the even subalgebra of `GA(3,0)`.
Unit quaternions act on 3-vectors `p = Quaternion(0, x, y, z)` by
`q · p · q†`. See
[`simplega/quaternion_visualization.py`](../../simplega/quaternion_visualization.py)
for axis/angle, frame, SLERP, and composition diagrams.

## Algebras in upstream that are not (yet) ported

The Python port intentionally focuses on the smallest algebras that
cover most 3-D and quaternion use cases. The following are documented
upstream but not implemented here:

| Algebra | Upstream link |
| --- | --- |
| `G(4,0)` | <https://monumoltd.github.io/SimpleGA.jl/dev/#G(4,0)> |
| `G(1,3)` (STA) | <https://monumoltd.github.io/SimpleGA.jl/dev/#G(1,3),-the-STA> |
| `G(3,0,1)` (PGA) — see [pga3d/](../../pga3d) for an independent implementation | <https://monumoltd.github.io/SimpleGA.jl/dev/#G(3,0,1),-the-PGA> |
| `G(4,1)` (CGA) | <https://monumoltd.github.io/SimpleGA.jl/dev/#G(4,1),-the-CGA> |
| `G(3,3)` | <https://monumoltd.github.io/SimpleGA.jl/dev/#G(3,3)-and-line-geometry.> |
| `G(2,4)` | <https://monumoltd.github.io/SimpleGA.jl/dev/#G(2,4),-conformal-spacetime.> |
| `G(4,4)`, `G(32,32)` | <https://monumoltd.github.io/SimpleGA.jl/dev/#The-Large-Algebras> |

---

## See also

* [Overview](overview.md)
* [API reference](api.md)
* [Upstream algebra index](https://monumoltd.github.io/SimpleGA.jl/dev/#The-Core-Algebras)
