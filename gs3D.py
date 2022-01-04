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

    Pearson's pattern can be switched with keys:
            'a': 'alpha_left',
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
            'n': 'nu_left',
            'p': 'pi_left',
            't': 'theta_left',
            'T': 'theta_right',
            'x': 'xi_left',
            'z': 'zeta_left',
            'Z': 'zeta_right'
    Several colormaps are available via keys 1 - 0, shifted for reversed version.
    Mouse left click in the grid refills reagent v at 0.5.
    Mouse right click in the grid put reagent v at 0.
    key / switch presentation between u and v.
    Spacebar reseeds the grid.
    key Up/Down and Left/right rotates the camera around the plane
    mouse scroll dolly in/out
    Shift key + mouse scroll dolly in/out the light source
    +/- keys to increase/decrease computation cycles per frame
"""

from math import sqrt

import argparse
import textwrap

import numpy as np

from vispy import gloo, app
from vispy.gloo import gl, Program, VertexBuffer, IndexBuffer, FrameBuffer
from vispy.util.transforms import perspective, translate, rotate
from vispy.geometry import create_plane

from gs_lib import (get_colormap, invertCmapName, createAndRegisterCmap,
                    createAndRegisterLinearSegmentedCmap, createColormaps,
                    plot_color_gradients, import_pearsons_types, setup_grid)

from shaders import compute_vertex
from shaders import compute_fragment_2 as compute_fragment
from shaders import render_3D_vertex
from shaders import render_3D_fragment
################################################################################


class Canvas(app.Canvas):

    colormapDictionnary = {
        '1': 'Boston_r',
        '&': 'Boston',
        '2': 'malmo',
        'é': 'malmo_r',
        '3': 'uppsala',
        '"': 'uppsala_r',
        '4': 'oslo_r',
        '\'': 'oslo',
        '5': 'Lochinver',
        '(': 'Lochinver_r',
        '6': 'Rejkjavik',
        '§': 'Rejkjavik_r',
        '7': 'detroit',
        'è': 'antidetroit',
        '8': 'tromso',
        '!': 'osmort',
        '9': 'irkoutsk',
        'ç': 'irkoutsk_r',
        '0': 'krasnoiarsk',
        'à': 'krasnoiarsk_r'
    }

    speciesDictionnary = {
        'a': 'alpha_left',
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
        'n': 'nu_left',
        'p': 'pi_left',
        't': 'theta_left',
        'T': 'theta_right',
        'x': 'xi_left',
        'z': 'zeta_left',
        'Z': 'zeta_right'
    }

    def __init__(self,
                 size=(1024, 1024),
                 modelSize=(512,512),
                 specie='alpha_left',
                 cmap='irkoutsk'):
        app.Canvas.__init__(self,
                            size=size,
                            title='3D Gray-Scott Reaction-Diffusion: - GregVDS',
                            keys='interactive')

        # General constants definition
        # --------------------------------------
        (self.w, self.h)                   = modelSize
        (self.fMin, self.fMax)             = (0.0, 0.08)
        (self.kMin, self.kMax)             = (0.03, 0.07)
        # (self.ddMin, self.ddMax)           = (0.2, 1.3)
        (self.fModAmount, self.kModAmount) = (0, 0)

        # Build plane data
        # --------------------------------------
        # Vertices contains
        # Position being vec3
        # texcoord being vec2
        # normal being vec3
        # color being vec4
        V, F, outline = create_plane(width_segments=self.w, height_segments=self.h)
        vertices = VertexBuffer(V)
        self.faces = IndexBuffer(F)
        self.outline = IndexBuffer(outline)

        # Build texture data
        # the texture contains 4 layers r, g, b, a
        # containing U and V concentrations twice
        # and are used through pingpong alternatively
        # each GPU computation/rendering cycle
        # --------------------------------------
        self.initializeGrid()
        self.pingpong = 1

        # Build view, model, projection
        # --------------------------------------
        self.viewCoordinates = [0, 0, -2.5]
        self.view = translate((self.viewCoordinates[0], self.viewCoordinates[1], self.viewCoordinates[2]))
        print(self.view)
        self.model = np.eye(4, dtype=np.float32)
        self.modelAzimuth = 0
        self.modelDirection = 0

        # light Parameters: direction, shininess exponant, attenuation parameters, ambientLight intensity
        # --------------------------------------
        self.lightDirection = np.array([-.4, .4, -2.1])
        self.shininess = 91.0
        self.c1 = 1.0
        self.c2 = 1.0
        self.c3 = 0.13
        self.ambientLight = 0.5

        # Colormaps related variables
        # --------------------------------------
        self.cmapName = cmap
        createColormaps()

        # Pearson's patterns related variables
        # definition of parameters for du, dv, f, k
        # --------------------------------------
        self.specie = specie
        self.species = import_pearsons_types()
        self.setSpecie(self.specie)

        # Mouse interactions parameters
        # --------------------------------------
        self.mouseDown = False
        self.mousePressControlPos = [0.0, 0.0]
        self.mousePressAltPos     = [0.0, 0.0]
        self.brush     = np.zeros((1, 1, 2), dtype=np.float32)
        self.brushType = 0

        # Dictionnary to map key commands to function
        # All these functions will receive the calling event.
        self.keyactionDictionnary = {
            ' ': self.initializeGrid,
            '/': self.switchReagent,
            '+': self.increaseCycle,
            '-': self.decreaseCycle
        }
        for key in Canvas.colormapDictionnary.keys():
            self.keyactionDictionnary[key] = self.pickColorMap
        for key in Canvas.speciesDictionnary.keys():
            self.keyactionDictionnary[key] = self.pickSpecie

        gl.GL_FRAGMENT_PRECISION_HIGH = 1

        # Build compute program
        # --------------------------------------
        self.computeProgram = Program(compute_vertex, compute_fragment)
        self.computeProgram["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self.computeProgram["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.computeProgram["texture"] = self.texture
        self.computeProgram["texture"].interpolation = gl.GL_NEAREST
        self.computeProgram["texture"].wrapping = gl.GL_REPEAT
        self.computeProgram["params"] = self.params
        self.computeProgram["dx"] = 1./self.w
        self.computeProgram["dy"] = 1./self.h
        self.computeProgram['pingpong'] = self.pingpong
        self.computeProgram['brush'] = self.brush
        self.computeProgram['brushtype'] = self.brushType

        # Build render program
        # --------------------------------------
        self.renderProgram = Program(render_3D_vertex, render_3D_fragment)
        self.renderProgram.bind(vertices)
        self.renderProgram["texture"] = self.computeProgram["texture"]
        self.renderProgram["texture"].interpolation = gl.GL_LINEAR
        self.renderProgram["texture"].wrapping = gl.GL_REPEAT
        self.renderProgram["scalingFactor"] = 30. * (self.w/512)
        self.renderProgram["u_light_position"] = self.lightDirection
        self.renderProgram["u_light_intensity"] = 1, 1, 1
        self.renderProgram["u_Ambient_color"] = self.ambientLight, self.ambientLight, self.ambientLight
        self.renderProgram["u_model"] = self.model
        self.renderProgram["dx"] = 1./self.w
        self.renderProgram["dy"] = 1./self.h
        self.renderProgram["u_view"] = self.view
        self.renderProgram['u_Shininess'] = self.shininess
        self.renderProgram['c1'] = self.c1
        self.renderProgram['c2'] = self.c2
        self.renderProgram['c3'] = self.c3
        self.renderProgram['pingpong'] = self.pingpong
        self.renderProgram["reagent"] = 1
        self.setColormap(self.cmapName)

        self.framebuffer = FrameBuffer(color=self.computeProgram["texture"],
                                       depth=gloo.RenderBuffer((self.h, self.w), format='depth'))
        # cycle of computation per frame
        self.cycle                  = 0

        self.activate_zoom()

        # OpenGL initialization
        # --------------------------------------
        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True,
                       # polygon_offset=(1, 1),
                       # blend_func=('src_alpha', 'one_minus_src_alpha'),
                       line_width=0.75)

        self.show()

    def on_draw(self, event):
        with self.framebuffer:
            gloo.set_viewport(0, 0, self.h, self.w)
            gloo.set_state(depth_test=False, clear_color='black', polygon_offset=(0, 0))
            # gloo.set_viewport(0, 0, self.h, self.w)
            self.computeProgram.draw('triangle_strip')
            for cycle in range(self.cycle):
                self.pingpong = 1 - self.pingpong
                self.computeProgram["pingpong"] = self.pingpong
                self.computeProgram.draw('triangle_strip')
                self.pingpong = 1 - self.pingpong
                self.computeProgram["pingpong"] = self.pingpong
                self.computeProgram.draw('triangle_strip')
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        gloo.set_state(blend=False, depth_test=True,
                       clear_color=(0.30, 0.30, 0.35, 1.00),
                       blend_func=('src_alpha', 'one_minus_src_alpha'),
                       polygon_offset=(1, 1),
                       polygon_offset_fill=True)
        gloo.clear(color=True, depth=True)
        self.renderProgram.draw('triangles', self.faces)
        # exchange between rg and ba sets of texture
        self.pingpong = 1 - self.pingpong
        self.computeProgram["pingpong"] = self.pingpong
        self.renderProgram["pingpong"] = self.pingpong
        # and loop
        self.update()

    def on_resize(self, event):
        self.activate_zoom()

    def activate_zoom(self):
        gloo.set_viewport(0, 0, *self.physical_size)
        projection = perspective(24, self.size[0] / float(self.size[1]),
                                 0.5, 100.0)
        # print(projection)
        self.renderProgram['u_projection'] = projection

    def on_mouse_wheel(self, event):
        # Move the plane in z
        # no shift modifier key
        self.viewCoordinates[2] += event.delta[1]
        self.lightDirection[2] += event.delta[1]
        self.view = translate((self.viewCoordinates[0], self.viewCoordinates[1], self.viewCoordinates[2]))
        self.renderProgram["u_view"] = self.view
        self.renderProgram["u_light_position"] = self.lightDirection
        # Move the light in z
        # shift modifier key
        self.lightDirection[2] += event.delta[0]
        self.renderProgram["u_light_position"] = self.lightDirection

        print('view coordinates: %2.1f, %2.1f, %2.1f' % (self.viewCoordinates[0], self.viewCoordinates[1], self.viewCoordinates[2]))
        print('light coordinates: %2.1f, %2.1f, %2.1f' % (self.lightDirection[0], self.lightDirection[1], self.lightDirection[2]))

    def on_mouse_press(self, event):
        self.mouseDown = True
        (x, y) = event.pos
        (sx, sy) = self.size
        xpos = x/sx
        ypos = 1 - y/sy
        if len(event.modifiers) == 0:
            self.computeProgram['brush'] = [xpos, ypos]
            self.computeProgram['brushtype'] = event.button

    def on_mouse_move(self, event):
        if(self.mouseDown):
            (x, y) = event.pos
            (sx, sy) = self.size
            xpos = x/sx
            ypos = 1 - y/sy
            if len(event.modifiers) == 0:
                # update brush coords here
                self.computeProgram['brush'] = [xpos, ypos]
                self.computeProgram['brushtype'] = event.button

    def on_mouse_release(self, event):
        self.mouseDown = False
        self.computeProgram['brush'] = [0, 0]
        self.computeProgram['brushtype'] = 0

    def on_key_press(self, event):
        if len(event.text) > 0:
            key = event.text
        else:
            key = event.key.name
        action = self.keyactionDictionnary.get(key)
        if action is not None:
            action(event)

        if event.key.name == "Up":
            self.modelAzimuth += 2
            # self.lightDirection[1] += .1
            # self.renderProgram["u_light_position"] = self.lightDirection
        elif event.key.name == "Down":
            self.modelAzimuth -= 2
            # self.lightDirection[1] -= .1
            # self.renderProgram["u_light_position"] = self.lightDirection
        elif event.key.name == "Right":
            self.modelDirection -= 2
            # self.lightDirection[0] += .1
            # self.renderProgram["u_light_position"] = self.lightDirection
        elif event.key.name == "Left":
            self.modelDirection += 2
            # self.lightDirection[0] -= .1
            # self.renderProgram["u_light_position"] = self.lightDirection
        self.modelAzimuth = np.clip(self.modelAzimuth, -90, 0)
        self.modelDirection = np.clip(self.modelDirection, -90, 90)
        azRotationMatrix = rotate(self.modelAzimuth, (1, 0, 0))
        diRotationMatrix = rotate(self.modelDirection, (0, 0, 1))
        self.renderProgram["u_model"] = np.matmul(np.matmul(self.model, diRotationMatrix), azRotationMatrix)
        # elif event.text == "z":
        #     self.shininess *= sqrt(2)
        #     self.shininess = np.clip(self.shininess, 0.1, 8192)
        #     self.renderProgram['u_Shininess'] = self.shininess
        # elif event.text == "a":
        #     self.shininess /= sqrt(2)
        #     self.shininess = np.clip(self.shininess, 0.1, 8192)
        #     self.renderProgram['u_Shininess'] = self.shininess
        # elif event.text == "q":
        #     self.ambientLight = np.clip(self.ambientLight - .01, 0, 1)
        #     self.renderProgram["u_Ambient_color"] = self.ambientLight, self.ambientLight, self.ambientLight
        # elif event.text == "s":
        #     self.ambientLight = np.clip(self.ambientLight + .01, 0, 1)
        #     self.renderProgram["u_Ambient_color"] = self.ambientLight, self.ambientLight, self.ambientLight
        # elif event.text == "e":
        #     self.c1 -= .1
        # elif event.text == "r":
        #     self.c1 += .1
        # elif event.text == "d":
        #     self.c2 -= .01
        # elif event.text == "f":
        #     self.c2 += .01
        # elif event.text == "c":
        #     self.c3 -= .01
        # elif event.text == "v":
        #     self.c3 += .01
        # self.renderProgram['c1'] = np.clip(self.c1, 1, 4)
        # self.renderProgram['c2'] = np.clip(self.c2, 0, .2)
        # self.renderProgram['c3'] = np.clip(self.c3, 0, .2)

        # print('Specularity intensity: %3.0f' % self.shininess)
        # print('Ambient light intensity: %1.1f' % self.ambientLight)
        # print('Light source position: %2.1f, %2.1f, %2.1f' % (self.lightDirection[0], self.lightDirection[1], self.lightDirection[2]))
        # print("Light attenuation parameters: 1/(%1.2f +%1.2f*d +%1.2f*d^2)" % (self.c1, self.c2, self.c3))

    def initializeGrid(self, event=None):
        print('Initialization of the grid.')
        self.UV = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.UV[:, :, 0:2] = setup_grid(self.h, self.w)
        self.UV += np.random.uniform(-0.02, 0.1, (self.h, self.w, 4))
        self.UV[:, :, 2] = self.UV[:, :, 0]
        self.UV[:, :, 3] = self.UV[:, :, 1]
        if not hasattr(self, 'texture'):
            self.texture = gloo.texture.Texture2D(data=self.UV, format=gl.GL_RGBA, internalformat='rgba32f')
        else:
            self.texture.set_data(self.UV)
        if hasattr(self, 'computeProgram'):
            self.computeProgram["texture"] = self.texture
            self.computeProgram["texture"].interpolation = gl.GL_NEAREST
            self.computeProgram["texture"].wrapping = gl.GL_REPEAT
        if hasattr(self, 'renderProgram'):
            self.renderProgram["texture"] = self.computeProgram["texture"]
            self.renderProgram["texture"].interpolation = gl.GL_LINEAR
            self.renderProgram["texture"].wrapping = gl.GL_REPEAT

    def setSpecie(self, specie):
        self.specie = specie
        self.printPearsonPatternDescription()
        self.P = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.P[:, :] = self.species[self.specie][0:4]
        self.modulateFK()
        self.updateComputeParams()

    def pickSpecie(self, event):
        specieName = Canvas.speciesDictionnary.get(event.text)
        if specieName is not None:
            self.setSpecie(specieName)

    def modulateFK(self, pos=None):
        f = self.P[0, 0, 2]
        k = self.P[0, 0, 3]
        if pos:
            self.fModAmount += 0.002 * (pos[1] - self.mousePressAltPos[1])
            self.kModAmount += 0.001 * (pos[0] - self.mousePressAltPos[0])
        rows, cols = self.h, self.w
        sinsF = np.sin(np.linspace(0.0, 2*np.pi, cols))
        sinsK = np.sin(np.linspace(0.0, 2*np.pi, rows))
        for i in range(rows):
            self.P[i, :, 2] = np.clip(f + self.fModAmount*sinsF, self.fMin, self.fMax)
        for i in range(cols):
            self.P[:, i, 3] = np.clip(k + self.kModAmount*sinsK, self.kMin, self.kMax)
        self.updateComputeParams()

    def updateComputeParams(self):
        self.params = gloo.texture.Texture2D(data=self.P, format=gl.GL_RGBA, internalformat='rgba32f')
        if hasattr(self, 'computeProgram'):
            self.computeProgram["params"] = self.params

    def increaseCycle(self, event=None):
        if not self.cycle:
            self.cycle = 1
        else:
            self.cycle *= 2
        if self.cycle > 64:
            self.cycle = 64
        print('Number of cycles: %3.0f' % (1 + 2 * self.cycle), end='\r')

    def decreaseCycle(self, event=None):
        self.cycle = int(self.cycle/2)
        if self.cycle < 1:
            self.cycle = 0
        print('Number of cycles: %3.0f' % (1 + 2 * self.cycle), end='\r')

    def setColormap(self, name):
        print('Using colormap %s.' % name)
        self.cmapName = name
        self.renderProgram["cmap"] = get_colormap(self.cmapName).map(np.linspace(0.0, 1.0, 1024)).astype('float32')

    def pickColorMap(self, event):
        colorMapName = Canvas.colormapDictionnary.get(event.text)
        if colorMapName is not None:
            self.setColormap(colorMapName)

    def switchReagent(self, event=None):
        self.renderProgram["reagent"] = 1 - self.renderProgram["reagent"]
        reagents = ('U', 'V')
        print('Displaying %s reagent concentration.' % reagents[int(self.renderProgram["reagent"])])

    def printPearsonPatternDescription(self):
        self.title = '3D Gray-Scott Reaction-Diffusion: Pattern %s - GregVDS' % self.specie
        print('Pearson\'s Pattern %s' % self.specie)
        print(self.species[self.specie][4])
        print('        dU  dV  f     k \n        %s %s %s %s' % (self.species[self.specie][0],
                                                                 self.species[self.specie][1],
                                                                 self.species[self.specie][2],
                                                                 self.species[self.specie][3]))

################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=textwrap.dedent(__doc__),
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
                        choices=Canvas.speciesDictionnary.values(),
                        default="lambda_left",
                        help="Pearson\' pattern")
    parser.add_argument("-c",
                        "--Colormap",
                        choices=Canvas.colormapDictionnary.values(),
                        default="irkoutsk",
                        help="Colormap used")
    args = parser.parse_args()


    c = Canvas(modelSize=(args.Size, args.Size),
               size=(args.Window, args.Window),
               specie=args.Pattern,
               cmap=args.Colormap)
    app.run()
