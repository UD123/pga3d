# Overview

`simplega` is a Python port of the Julia package
[SimpleGA.jl](https://github.com/MonumoLtd/SimpleGA.jl) by Chris Doran
et al. It provides compact, grade-split implementations of several
low-dimensional geometric algebras with a uniform API.

The port follows the spirit of the original: each algebra lives in its
own module, exposes a basis, and represents multivectors using a
small, fixed-size struct chosen to match a familiar algebraic isomorphism
(complex numbers, quaternions, 2×2 complex matrices). This keeps the
geometric product to a handful of multiplications per operation.

See the upstream
[documentation home](https://monumoltd.github.io/SimpleGA.jl/dev/) for
the wider context and additional algebras that are not yet ported.

## Installation

This is not yet packaged on PyPI. Use it directly from a checkout:

```bash
git clone https://github.com/UD123/pga3d.git
cd pga3d
python -c "import simplega; print(simplega.__doc__)"
```

Runtime requirement is just Python 3.10+. The visualization modules
also need `numpy` and `matplotlib`.

## First example

```python
from simplega import ga20

e = ga20.basis                # 2-D Euclidean basis: [e1, e2]
a = e[0] + e[1]               # vector  e1 + e2
b = 2.0 * e[0] + 3.0 * e[1]   # vector  2 e1 + 3 e2
print(a * b)                  # geometric product:  <a,b> + a ^ b
```

A second example, in `GA(3,0)`:

```python
import math
from simplega import ga30, project, bivector_exp

e1, e2, e3 = ga30.e1, ga30.e2, ga30.e3
v = 0.8 * e1 + 0.5 * e2 - 0.6 * e3

# 90-degree rotor around z, built as exp of a bivector
R = bivector_exp(ga30.Even(0.0, 0.0, 0.0, math.pi / 4))
v_rot = R * v * R.adjoint     # sandwich product

print("inner part:", project(v * v, 0))
print("rotated v :", v_rot)
```

## Ported algebras

| Module | Algebra | Representation |
| --- | --- | --- |
| [`simplega.ga20`](../../simplega/ga20.py) | `GA(2,0)` | one complex number |
| [`simplega.ga30`](../../simplega/ga30.py) | `GA(3,0)` | quaternion-like `(w, x, y, z)` |
| [`simplega.ga31`](../../simplega/ga31.py) | `GA(3,1)` | 2×2 complex matrix (4 complex entries) |
| [`simplega.quaternions`](../../simplega/quaternions.py) | quaternions | `(w, x, y, z)` |

The remaining algebras from the upstream package (`G(4,0)`, `STA`,
`PGA`, `CGA`, `G(3,3)`, `G(2,4)`, `G(4,4)`, `G(32,32)`) are not yet
implemented in the port. See the
[upstream algebra index](https://monumoltd.github.io/SimpleGA.jl/dev/#The-Core-Algebras)
for what they look like in Julia.

## The Even / Odd trick

To keep each multivector small and each geometric product cheap, the
upstream library splits a multivector into:

* an `Even` part — grade 0 + grade 2 (+ grade 4 in 4-D algebras),
* an `Odd` part — grade 1 + grade 3 (+ grade 5 in 5-D algebras).

Each algebra picks an algebra-specific element that maps odd → even
(e.g. multiplication by the pseudoscalar in `GA(3,0)`), so both `Even`
and `Odd` can be stored in the same compact representation.

A consequence carried over to the port: **you cannot add an `Even` and
an `Odd` directly**. In practice this is rarely a limitation; when you
genuinely need to mix grades, work in an algebra one dimension higher
where both live inside the same even subalgebra (see the
[upstream notes on this](https://monumoltd.github.io/SimpleGA.jl/dev/#Bases-and-the-Even-/-Odd-trick)).

## Next

* [API reference](api.md) — arithmetic, projection, exponentiation,
  shared helpers.
* [Per-algebra notes](algebras.md) — representation details,
  basis elements, and conventions for each ported algebra.
