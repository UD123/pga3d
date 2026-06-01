# PGA3

This library provides Proper Geometry Algorithms, for doing geometry in 3D as the math-gods intended.

Psst ... if you're a math-person, this library implements Projective
Geometric Algebra in 3D (i.e. `ℝ(3, 0, 1)`), so that intersections and
joins etc. are super easy to implement, and benefit from all the
nicities of this algebra. But I don't want to scare our users, so we provide
an API that normal humans can actually understand.


### Status

Experimental and work in progress. I dug into this to see if it's
something we could use in pygfx. I'm not sure yet. This state matches
my progress (and understanding) so far.


### Benefits of this library (over libs that use linalg)

* Small size (once we have the algebra set up, all conversions, intersections etc. are one-liners).
* Transformations (translation + rotations) can be composed with minimal loss of precision (in contrast to composing multiple 4x4 matrices).
* Transformations are "linear", in that you can multiply them, e.g. to get smooth interpolation.
* Less need for checks and exceptions: intersecting two parallel lines? Fine! The result is a point at infinity (a direction)!
* No implicit design decisions like right-handed vs left-handed.


### Diving into the algebra

This lib attempts to provide a way to use a very cool/advanced/new
mathematical tool (3D PGA) without having to know about it.

If you do want to learn more about (projective) geometric algebra,
here are some resources that I found useful. Prepare for your mind being blown:

* Probably a good start: https://www.youtube.com/watch?v=tX4H_ctggYo
* General resources: https://bivector.net
* Cheat sheet: https://bivector.net/3DPGA.pdf
* PGA explained for devs: https://observablehq.com/@enkimute/understanding-pga-1
* Geometric algebra in JS, with interactive demos: https://github.com/enkimute/ganja.js
* C++ implementation of 3D PGA: https://github.com/jeremyong/Klein


### Example figures

The figures below are produced by [doc/generate_figures.py](doc/generate_figures.py),
which drives the plotting helpers in [pga3d_display.py](pga3d_display.py).
Regenerate them with:

```
python doc/generate_figures.py
```

| | |
| --- | --- |
| ![Single point](doc/fig_point.png) | ![Line from two points](doc/fig_line.png) |
| A single `Point(1, 2, 3)`. | A `Line` joining two points. |
| ![Plane from three points](doc/fig_plane.png) | ![Triangle scene](doc/fig_scene.png) |
| A `Plane` built from three points. | Points, lines, and a plane together. |
| ![Translation and rotation](doc/fig_transformations.png) | ![Projection onto a line](doc/fig_projection.png) |
| A point, its translation, and a composed translation+rotation. | Projecting a point onto a line, with the supporting plane. |
