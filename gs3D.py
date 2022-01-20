# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
# Author: Gregoire Vandenschrick
# Date:   30/12/2021
# -----------------------------------------------------------------------------
# Parts of this is based on the following:
#
# https://github.com/soilstack/react_diffuse/
# which uses Matplotlib and hence is quite slow.
# I reused its systems.py and the setup_grid function.
#
# https://github.com/pmneila/jsexp
# while it's using Javascript and THREE.js, it was a source of understanding
# and inspiration, for the brush tool for example.
#
# https://github.com/glumpy/glumpy/blob/master/examples/grayscott.py
#
# which I discovered is very close to this:
# http://vispy.org/examples/demo/gloo/terrain.html
#
# Primary Sources
# 1. [Karl Sims Original](http://karlsims.com/rd.html)
# 1. [Detailed discussion of Gray-Scott Model](http://mrob.com/pub/comp/xmorphia/)
# 1. [Coding Train Reaction-Diffusion (based on Karl Sims)](https://www.youtube.com/watch?v=BV9ny785UNc&t=2100s)
# 1. [Pearson canonical labelling of systems](https://arxiv.org/abs/patt-sol/9304003)
#
# This alternative version display the patterns using a 3D plane that can be oriented
# I followed a good part of the lighting explanations found here for the 3D render fragment:
# http://learnwebgl.brown37.net/index.html
# -----------------------------------------------------------------------------
"""
    Gray-Scott reaction-diffusion model
    -----------------------------------

    3D presentation of a model of Reaction-Diffusion.
    For more use help, type python3 gs3D.py -h.
"""

################################################################################
# from math import pi
import argparse
import textwrap

# import numpy as np

from vispy import app
# from vispy.gloo import gl

from gs3D_lib import (GrayScottModel, MainRenderer, Canvas)

# ? Use of this ?
# gl.use_gl('gl+')
# app.use_app('pyside6')

################################################################################
def fun(x):
    c.title = c.title2 +' - FPS: %0.1f' % x


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=textwrap.dedent(Canvas.getCommandsDocs()),
                                     epilog= textwrap.dedent("""Examples:
    python3 gs3D.py
    python3 gs3D.py -c osmort
    python3 gs3D.py -s 512 -p kappa_left -c oslo
    python3 gs3D.py -s 512 -w 800 -p alpha_left -c detroit"""),
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-s",
                        "--Size",
                        type=int,
                        default=512,
                        help="Size of model")
    parser.add_argument("-w",
                        "--Window",
                        type=int,
                        default=1024,
                        help="Size of window")
    parser.add_argument("-p",
                        "--Pattern",
                        choices={**GrayScottModel.speciesDictionnary,
                                 **GrayScottModel.speciesDictionnaryShifted}.values(),
                        default="alpha_left",
                        help="Pearson\' pattern")
    parser.add_argument("-c",
                        "--Colormap",
                        choices={**MainRenderer.colormapDictionnary,
                                 **MainRenderer.colormapDictionnaryShifted}.values(),
                        default="honolulu_r",
                        help="Colormap used")

    args = parser.parse_args()

    c = Canvas(modelSize=(args.Size, args.Size),
               size=(args.Window, args.Window),
               specie=args.Pattern,
               cmap=args.Colormap)
    c.measure_fps2(callback=fun)
    c.show()
    app.run()
