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
1. [Paper of John E. Pearson](https://arxiv.org/pdf/patt-sol/9304003.pdf)
1. [Karl Sims Original](http://karlsims.com/rd.html)
1. [Detailed discussion of Gray-Scott Model](http://mrob.com/pub/comp/xmorphia/)
1. [Coding Train Reaction-Diffusion (based on Karl Sims)](https://www.youtube.com/watch?v=BV9ny785UNc&t=2100s)
1. [Pearson canonical labelling of systems](https://arxiv.org/abs/patt-sol/9304003)

I do certainly not claim originality nor paternity of all this; I only intend to
try and make it myself working.

This was developed onto Mac, under Python 3.10.
Dependencies are:
- vispy
- numpy
- matplotlib

vispy requires a backend such as pyside2, pyside6, wx, Pyqt5...

To launch, simply type python3 gs.py for a 2D version.
This one has more options to play with the model parameters, such as
spatial variation of feed, kill and dd accross the grid. It displays also
these variations momentarily when modifiying them, against a Pearson phase diagram.
Help available with python3 gs.py -h

A second version is currently in development: gs3D.py. It is basically the same
model, but in 3D. The grid of the model is a plane that can be rotated.
Pearson's patterns are still switchable with the same keys. colormaps too.
No local variations of feed, kill or dd implemented yet. This version is more
a sandbox for me to learn OpenGL, shader techniques, shadowing, but still is the
proper Reaction-diffusion, only fancied up graphically.
Help available with python3 gs3D.py -h

Inspirations for the OpenGL shaders techniques and code used:
1. [Learn WebGL by Dr. Wayne Brown](http://learnwebgl.brown37.net/index.html)
1. [OpenGL Tutorials](https://www.opengl-tutorial.org)
1. [Fabien Sanglard's website](https://fabiensanglard.net)
1. [Vispy code and examples](https://github.com/vispy/vispy)
1. [Fresnel factor computation](https://lettier.github.io/3d-game-shaders-for-beginners/fresnel-factor.html)

Commands can be passed with key combos, but these were defined on my keyboard.
Depending on your hardware and os settings, those could not respond as intended.
Please do clone and check the code, and adjust to your liking.

Pearson's pattern can be switched with keys:
-       'a': 'alpha_left',
        'A': 'alpha_right',
        'b': 'beta_left',
        'B': 'beta_right',
        'd': 'delta_left',
        'D': 'delta_right',
        'e': 'epsilon_left',
        'E': 'epsilon_right',
        'g': 'gamma_left',
        'G': 'gamma_right',
        'h': 'eta',
        'i': 'iota',
        'k': 'kappa_left',
        'K': 'kappa_right',
        'l': 'lambda_left',
        'L': 'lambda_right',
        'm': 'mu_left',
        'M': 'mu_right',
        'n': '*nu_left',
        'p': 'pi_left',
        't': 'theta_left',
        'T': 'theta_right',
        'x': 'xi_left',
        'z': 'zeta_left',
        'Z': 'zeta_right'
- Several colormaps are available via 1 - 0, shift for reversed versions.

These are key combos commands for gs.py only:
- Mouse left click in the grid refills reagent v at 0.5.
- Mouse right click in the grid put reagent v at 0.
- Ctrl + Mouse click and drag to modify globaly feed and kill rates.
- Alt + Mouse click and drag to vary sinusoidally feed and kill rates.
- Shift + Mouse click and drag modifies the hillshading parameters.
- key = reset the spatial variation of feed and kill to their Pattern values
- key + toggles guidelines when modifying/modulatin feed and kill rates.
- key / switches presentation between u and v.
- key * toggles hillshading on or off.
- key $ toggles interpolation on or off.
- key Up/Down arrows multiply computation cycles per frame
- key Left/right decreases/increases globally dD of the model
- key Shift + Left/right decreases/increases gaussian variation of dD
        If guidelines are on, the dD general value and modulation curve will be drawn;
        Currently, to make these disappear, one has to switch off guidelines with
        key +, and make another modification to general dD or dD modulation.
- Spacebar reseeds the grid.

keys =, l, spacebar should get you back to something normal ;-).

Keys combo for gs3D.py have still not settled down, but can be easily found by
typing python3 gs3D.py -h in a shell.

If some of these combos would not work for you,
the keyactionDictionnary at the end of Canvas class can be easily adapted:
- (',', ('Control',)): (MainRenderer.modifyLightCharacteristic, ('ambient',))
- defines a tuple with key ',' with modifier key 'Control' and maps it to call
- modifyLightCharacteristic('ambient').
- Just change key and/or modifier key to adapt.

Some new colormaps were designed with a nice online tool:
[CCCTool](https://ccctool.com)

Have fun!

Greg
