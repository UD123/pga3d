# SimpleGA (Python port) — documentation

This is a documentation set for the Python port of
[SimpleGA.jl](https://github.com/MonumoLtd/SimpleGA.jl) that lives in
[../../simplega/](../../simplega). It mirrors the structure of the
[upstream Julia documentation](https://monumoltd.github.io/SimpleGA.jl/dev/)
but is written for the Python API and covers only the algebras that
have been ported so far.

For the original Julia package and the full algebra list (G(4,0),
G(1,3), G(3,0,1), G(4,1), G(3,3), G(2,4), G(4,4), G(32,32)), see the
[upstream documentation](https://monumoltd.github.io/SimpleGA.jl/dev/).

## Contents

1. [Overview](overview.md) — installation, first example, design notes.
2. [API](api.md) — bases, arithmetic, projection, exponentiation, helpers.
3. [Algebras](algebras.md) — per-algebra notes for the ported subset.

## Quick links

* Source: [simplega/](../../simplega/)
* `GA(2,0)`: [simplega/ga20.py](../../simplega/ga20.py)
* `GA(3,0)`: [simplega/ga30.py](../../simplega/ga30.py)
* `GA(3,1)`: [simplega/ga31.py](../../simplega/ga31.py)
* Quaternions: [simplega/quaternions.py](../../simplega/quaternions.py)
* Visualizations: [simplega/ga30_visualization.py](../../simplega/ga30_visualization.py),
  [simplega/quaternion_visualization.py](../../simplega/quaternion_visualization.py),
  [simplega/bivector_visualization.py](../../simplega/bivector_visualization.py)
