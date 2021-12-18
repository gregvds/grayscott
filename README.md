# grayscott


This is a Gray-Scott reaction-diffusion model based on Python and Vispy.

This work is built using several inspirations found on Github and Internet;
Some were made in Python using Matplotlib, others in Javascript using Three.js,
a third one was closer, in Python and Glumpy.

I do certainly not claim originality nor paternity of all this; I only intend to
try and make it myself working, first in Matplotlib, then for better perfs on
OpenGL. As Glumpy did not install on my Mac OS Big Sur (Triangle dependency not
building), I gave a try at Vispy.

This was developed onto Mac, under Python 3.10.

To launch, simply type python3 gs.py

Pearson's pattern can be switched with keys:
such as a, b, d, g, k, i, e, p, x, replacing greek letters.
Several colormaps are available via 1 - 6, shifted for reversed version.
Mouse left click in the grid refills reagent v at 0.5.
Ctrl + Mouse click and drag to modify feed and kill rates
* switch presentation between u and v.
Spacebar reseeds the grid.
