# grayscott


This is a Gray-Scott reaction-diffusion model based on Python and Vispy.

This work is built using several inspirations found on Github and Internet:
Parts of this is based on the following:

(https://github.com/soilstack/react_diffuse/)
which uses Matplotlib and hence is quite slow.
I reused its systems.py and the setup_grid function.

(https://github.com/pmneila/jsexp)
while it's using Javascript and THREE.js, it was a source of understanding
and inspiration, for the brush tool for example.

(https://github.com/glumpy/glumpy/blob/master/examples/grayscott.py)

which I discovered is very close to this:
(http://vispy.org/examples/demo/gloo/terrain.html)

Primary Sources
1. [Karl Sims Original](http://karlsims.com/rd.html)
1. [Detailed discussion of Gray-Scott Model](http://mrob.com/pub/comp/xmorphia/)
1. [Coding Train Reaction-Diffusion (based on Karl Sims)](https://www.youtube.com/watch?v=BV9ny785UNc&t=2100s)
1. [Pearson canonical labelling of systems](https://arxiv.org/abs/patt-sol/9304003)

I do certainly not claim originality nor paternity of all this; I only intend to
try and make it myself working.

This was developed onto Mac, under Python 3.10.

To launch, simply type python3 gs.py

Pearson's pattern can be switched with keys:
- Keys a, b, d, e, g, i, k, l, m, n, p, x, z replacing greek letters.
- Several colormaps are available via 1 - 7, shifted for reversed version.
- Mouse left click in the grid refills reagent v at 0.5.
- Mouse right click in the grid put reagent v at 0.
- Ctrl + Mouse click and drag to modify feed and kill rates.
- Shift + Mouse click and drag modifies the hillshading parameters.
- Key / switch presentation between u and v.
- Key * toggles hillshading on or off.
- Spacebar reseeds the grid.
