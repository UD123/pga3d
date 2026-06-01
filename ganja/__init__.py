"""
ganja — a small Python Clifford / geometric algebra library.

Inspired by ganja.js (https://github.com/enkimute/ganja.js) by Steven
De Keninck — in particular its `Algebra(p, q, r)` factory and its
`graph(...)` rendering API. This package is an original Python
reimplementation, not a translation of the JavaScript source.

Modules
-------
ganja.algebra        — generic Clifford algebra factory and multivectors
ganja.visualization  — matplotlib-based `graph(...)` for 2D / 3D PGA

Quick start
-----------
    from ganja import Algebra
    PGA2 = Algebra(2, 0, 1)        # 2-D projective geometric algebra
    e0, e1, e2 = PGA2.basis_vectors()
    point = e0 + 2 * e1 + 3 * e2   # joins / meets via .dual() and ^
"""

from .algebra import Algebra, Multivector
from .visualization import graph

__all__ = ["Algebra", "Multivector", "graph"]
