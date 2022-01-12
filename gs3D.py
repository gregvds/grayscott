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
    shift + Mouse left click in the grid refills reagent v at 0.5.
    shift + Mouse right click in the grid put reagent v at 0.
    Mouse click and drag rotate the plane.
    key / switch presentation between u and v.
    Spacebar reseeds the grid.
    key Up/Down and Left/right move the light in X and Y
    mouse scroll dolly in/out
    Shift + mouse scroll dolly in/out the light source
    +/- keys to increase/decrease computation cycles per frame
"""

################################################################################
from math import tan, atan, pi, sqrt, asin

import argparse
import textwrap

import numpy as np

from vispy import gloo, app
from vispy.geometry import create_plane
from vispy.gloo import gl, Program, VertexBuffer, IndexBuffer, FrameBuffer
from vispy.util.transforms import perspective, translate, rotate

from gs_lib import (get_colormap, createColormaps, import_pearsons_types, setup_grid)

from shaders import compute_vertex
from shaders import compute_fragment_2 as compute_fragment
from shaders import render_3D_vertex
from shaders import render_3D_fragment
from shaders import shadow_vertex
from shaders import shadow_fragment


# ? Use of this ?
# gl.use_gl('gl+')

################################################################################


class Canvas(app.Canvas):

    colormapDictionnary = {
        '&': 'Boston',
        'é': 'malmo_r',
        '"': 'uppsala_r',
        '\'': 'oslo',
        '(': 'Lochinver_r',
        '§': 'Rejkjavik_r',
        'è': 'antidetroit',
        '!': 'osmort',
        'ç': 'irkoutsk_r',
        'à': 'krasnoiarsk_r'
    }

    colormapDictionnaryShifted = {
        '1': 'Boston_r',
        '2': 'malmo',
        '3': 'uppsala',
        '4': 'oslo_r',
        '5': 'Lochinver',
        '6': 'Rejkjavik',
        '7': 'detroit',
        '8': 'tromso',
        '9': 'irkoutsk',
        '0': 'krasnoiarsk'
    }

    speciesDictionnary = {
        'a': 'alpha_left',
        'b': 'beta_left',
        'd': 'delta_left',
        'e': 'epsilon_left',
        'g': 'gamma_left',
        'h': 'eta',
        'i': 'iota',
        'k': 'kappa_left',
        'l': 'lambda_left',
        'm': 'mu_left',
        'n': 'nu_left',
        'p': 'pi_left',
        't': 'theta_left',
        'x': 'xi_left',
        'z': 'zeta_left'
    }

    speciesDictionnaryShifted = {
        'A': 'alpha_right',
        'B': 'beta_right',
        'D': 'delta_right',
        'E': 'epsilon_right',
        'G': 'gamma_right',
        'K': 'kappa_right',
        'L': 'lambda_right',
        'M': 'mu_right',
        'T': 'theta_right',
        'Z': 'zeta_right'
    }

    def __init__(self,
                 size=(1024, 1024),
                 modelSize=(512,512),
                 specie='alpha_left',
                 cmap='irkoutsk',
                 options=False):
        app.Canvas.__init__(self,
                            size=size,
                            title='3D Gray-Scott Reaction-Diffusion: - GregVDS',
                            keys='interactive')

        # General constants definition
        # --------------------------------------
        (self.w, self.h)                   = modelSize
        (self.fMin, self.fMax)             = (0.0, 0.08)
        (self.kMin, self.kMax)             = (0.03, 0.07)
        (self.fModAmount, self.kModAmount) = (0, 0)
        # (self.ddMin, self.ddMax)           = (0.2, 1.3)

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
        self.model = np.eye(4, dtype=np.float32)
        self.modelAzimuth = 0
        self.modelDirection = 0

        # light Parameters: direction, shininess exponant, attenuation parameters,
        # ambientLight intensity, ambientColor, diffuseColor and specularColor
        # --------------------------------------
        self.lightCoordinates = np.array([-.4, .4, -2.1])
        self.lightIntensity = (1., 1., 1.)
        self.c1 = .8
        self.c2 = .8
        self.c3 = 0.02
        self.ambientIntensity = 0.5
        self.ambientColor = np.array((0.5, 0.5, 0.5, 1))
        self.diffuseColor = (1., 1., 1., 1.)
        self.specularColor = (1., 1., 1., 1.)
        self.shininess = 91.0
        self.useMaterial = False

        # Build view, projection for shadowCam
        # --------------------------------------
        self.shadowCamAt = [0, 0, 0]
        self.updateLightCam()

        # Currently, the shadowmap is a simple image for I cannot set properly
        # the FrameBuffer depth part (RenderBuffer)...
        # Finally, this is useful to pass moments1 and 2 for VSF mapping :-)
        self.shadowMapSize = 2048
        self.shadowGrid = np.ones((self.shadowMapSize, self.shadowMapSize, 4), dtype=np.float32) * .1
        self.shadowTexture = gloo.texture.Texture2D(data=self.shadowGrid, format=gl.GL_RGBA, internalformat='rgba32f')
        # self.shadowGrid = np.ones((self.shadowMapSize, self.shadowMapSize), dtype=np.float32) * .1
        # self.shadowTexture = gloo.texture.Texture2D(data=self.shadowGrid, format=gl.GL_LUMINANCE, internalformat='r32f')

        self.displaySwitch = 0

        # DEBUG, toggles to switch on and off different parts of lighting
        # --------------------------------------
        self.ambient     = True
        self.attenuation = True
        self.diffuse     = True
        self.specular    = True
        self.shadow      = 3

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
        self.mousePressShiftPos   = [0.0, 0.0]
        self.brush     = np.zeros((1, 1, 2), dtype=np.float32)
        self.brushType = 0

        self.holdModel = False

        # Dictionnary to map key commands to function
        # All these functions will receive the calling event even if not used.
        # --------------------------------------
        self.keyactionDictionnary = {
            (' ', ()): (self.initializeGrid, ()),
            ('/', ('Shift',)): (self.switchReagent, ()),
            ('+', ('Shift',)): (self.increaseCycle, ()),
            ('-', ()): (self.decreaseCycle, ()),
            ('=', ()): (self.resetModel, ()),
            ('Up', ()): (self.updateLight, ((0, 0.05, 0),)),
            ('Down', ()): (self.updateLight, ((0, -0.05, 0),)),
            ('Left', ()): (self.updateLight, ((-0.05, 0, 0),)),
            ('Right', ()): (self.updateLight, ((0.05, 0, 0),)),
            (',', ('Control',)): (self.modifyLightCharacteristic, ('ambient',)),
            (';', ('Control',)): (self.modifyLightCharacteristic, ('diffuse',)),
            (':', ('Control',)): (self.modifyLightCharacteristic, ('specular',)),
            ('=', ('Control',)): (self.modifyLightCharacteristic, ('shadow',)),
            ('@', ('Control',)): (self.modifyLightCharacteristic, ('attenuation',)),
            ('A', ('Control',)): (self.modifyLightCharacteristic, ('shininess', '-')),
            ('Z', ('Control',)): (self.modifyLightCharacteristic, ('shininess', '+')),
            ('#', ('Shift',)): (self.switchDisplay, ()),
            ('π', ('Alt',)): (self.pause, ())
        }
        for key in Canvas.colormapDictionnary.keys():
            self.keyactionDictionnary[(key,())] = (self.pickColorMap, ())
        for key in Canvas.colormapDictionnaryShifted.keys():
            self.keyactionDictionnary[(key,('Shift',))] = (self.pickColorMap, ())
        for key in Canvas.speciesDictionnary.keys():
            self.keyactionDictionnary[(key,())] = (self.pickSpecie, ())
        for key in Canvas.speciesDictionnaryShifted.keys():
            self.keyactionDictionnary[(key,('Shift',))] = (self.pickSpecie, ())

        # ? better computation ?
        # --------------------------------------
        gl.GL_FRAGMENT_PRECISION_HIGH = 1

        # Build compute program
        # --------------------------------------
        self.computeProgram = Program(compute_vertex, compute_fragment, count=4)
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
        self.renderProgram["dx"] = 1./self.w
        self.renderProgram["dy"] = 1./self.h
        self.renderProgram['pingpong'] = self.pingpong
        self.renderProgram["reagent"] = 1
        self.renderProgram["shadowMap"] = self.shadowTexture
        self.renderProgram["shadowMap"].interpolation = gl.GL_LINEAR
        self.renderProgram["shadowMap"].wrapping = gl.GL_CLAMP_TO_EDGE
        self.renderProgram["u_Shadowmap_projection"] = self.shadowProjection
        self.renderProgram["u_Shadowmap_view"] = self.shadowView
        self.renderProgram["u_Tolerance_constant"] = 0.005
        self.renderProgram["scalingFactor"] = 30. * (self.w/512)
        self.renderProgram["u_view"] = self.view
        self.renderProgram["u_model"] = self.model
        self.renderProgram["ambient"] = self.ambient
        self.renderProgram["attenuation"] = self.attenuation
        self.renderProgram["diffuse"] = self.diffuse
        self.renderProgram["specular"] = self.specular
        self.renderProgram["shadow"] = self.shadow
        self.renderProgram["u_light_position"] = self.lightCoordinates
        self.renderProgram["u_light_intensity"] = self.lightIntensity
        self.renderProgram["u_Ambient_color"] = self.ambientColor
        self.renderProgram["u_ambient_intensity"] = self.ambientIntensity
        self.renderProgram["u_diffuse_color"] = self.diffuseColor
        self.renderProgram["u_specular_color"] = self.specularColor
        self.renderProgram['u_Shininess'] = self.shininess
        self.renderProgram['c1'] = self.c1
        self.renderProgram['c2'] = self.c2
        self.renderProgram['c3'] = self.c3
        self.setColormap(self.cmapName)

        # Build shadowmap render program
        # --------------------------------------
        self.shadowProgram = Program(shadow_vertex, shadow_fragment, count=self.w*self.h)
        self.shadowProgram.bind(vertices)
        self.shadowProgram["texture"] = self.computeProgram["texture"]
        self.shadowProgram["texture"].interpolation = gl.GL_LINEAR
        self.shadowProgram["texture"].wrapping = gl.GL_REPEAT
        self.shadowProgram['pingpong'] = self.pingpong
        self.shadowProgram["reagent"] = 1
        self.shadowProgram["scalingFactor"] = 30. * (self.w/512)
        self.shadowProgram["u_model"] = self.model
        self.shadowProgram["u_view"] = self.shadowView
        self.shadowProgram['u_projection'] = self.shadowProjection

        # Define a FrameBuffer to update model state in texture
        # --------------------------------------
        self.modelbuffer = FrameBuffer(color=self.computeProgram["texture"],
                                       depth=gloo.RenderBuffer((self.h, self.w), format='depth'))

        # Define a 'DepthBuffer' to render the shadowmap from the light
        # --------------------------------------
        self.shadowBuffer = FrameBuffer(color=self.renderProgram["shadowMap"],
                                       depth=gloo.RenderBuffer((self.shadowMapSize, self.shadowMapSize), format='depth'))

        # cycle of computation per frame
        self.cycle = 0

        self.activate_zoom()

        # OpenGL initialization
        # --------------------------------------
        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True,
                       line_width=0.75)

        self.show()

    ############################################################################
    #

    def on_draw(self, event):
        # Render next model state into buffer
        if self.holdModel is not True:
            with self.modelbuffer:
                gloo.set_viewport(0, 0, self.h, self.w)
                gloo.set_state(depth_test=False,
                               clear_color='black',
                               polygon_offset=(0, 0))
                self.computeProgram.draw('triangle_strip')
                # repeat model state computation several time to speed up slow patterns
                for cycle in range(self.cycle):
                    self.pingpong = 1 - self.pingpong
                    self.computeProgram["pingpong"] = self.pingpong
                    self.computeProgram.draw('triangle_strip')
                    self.pingpong = 1 - self.pingpong
                    self.computeProgram["pingpong"] = self.pingpong
                    self.computeProgram.draw('triangle_strip')

        # Render the shadowmap into buffer
        if self.shadow > 0:
            with self.shadowBuffer:
                gloo.set_viewport(0, 0, self.shadowMapSize, self.shadowMapSize)
                gloo.set_state(depth_test=True,
                               polygon_offset=(1, 1),
                               polygon_offset_fill=True)
                gloo.clear(color=True, depth=True)
                self.shadowProgram.draw('triangles', self.faces)
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        gloo.set_state(blend=False, depth_test=True,
                       clear_color=(0.30, 0.30, 0.35, 1.00),
                       blend_func=('src_alpha', 'one_minus_src_alpha'),
                       polygon_offset=(1, 1),
                       polygon_offset_fill=True)
        gloo.clear(color=True, depth=True)
        self.renderProgram.draw('triangles', self.faces)

        # exchange between rg and ba sets in texture
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
                                 0.1, 20.0)
        if hasattr(self, 'renderProgram'):
            self.renderProgram['u_projection'] = projection
        if hasattr(self, 'coordinatesProgram'):
            self.coordinatesProgram['u_projection'] = projection

    ############################################################################
    # Mouse and keys interactions

    def on_mouse_wheel(self, event):
        # no shift modifier key
        # Move the plane AND light in z
        self.viewCoordinates[2] += event.delta[1]/2
        self.viewCoordinates[2] = np.clip(self.viewCoordinates[2], -5.0, -0.7)
        self.updateView()
        self.lightCoordinates[2] += event.delta[1]/2
        self.lightCoordinates[2] = np.clip(self.lightCoordinates[2], self.viewCoordinates[2]+.1, -0.8)
        # shift modifier key
        # Move only the light in z
        self.lightCoordinates[2] += event.delta[0]/2
        self.lightCoordinates[2] = np.clip(self.lightCoordinates[2], self.viewCoordinates[2]+.1, -0.8)
        self.updateLight()

    def on_mouse_press(self, event):
        self.mouseDown = True
        (x, y) = event.pos
        (sx, sy) = self.size
        xpos = x/sx
        ypos = 1 - y/sy
        if len(event.modifiers) == 0:
            self.mousePressShiftPos = [xpos, ypos]
        elif len(event.modifiers) == 1 and event.modifiers[0] == 'Shift':
            self.computeProgram['brush'] = [xpos, ypos]
            self.computeProgram['brushtype'] = event.button

    def on_mouse_move(self, event):
        if(self.mouseDown):
            (x, y) = event.pos
            (sx, sy) = self.size
            xpos = x/sx
            ypos = 1 - y/sy
            if len(event.modifiers) == 0:
                self.modelAzimuth += 40. * (self.mousePressShiftPos[1] - ypos)
                self.modelDirection -= 40. * (self.mousePressShiftPos[0] - xpos)
                self.modelAzimuth = np.clip(self.modelAzimuth, -90, 0)
                self.modelDirection = np.clip(self.modelDirection, -90, 90)
                self.mousePressShiftPos = [xpos, ypos]
                self.updateModel()
            elif len(event.modifiers) == 1 and event.modifiers[0] == 'Shift':
                """
                print("x, y: %s, %s" % (x, y))
                print("sx, sy: %s, %s" % (sx, sy))
                print("texCoordx, texcoordy: %s, %s" % (int(x/sx*self.w), int(y/sy*self.h)))
                print('CoordinatesBuffer: %s' % self.coordinatesBuffer.read())
                texCoords = self.coordinatesBuffer.read()[int(x/sx*self.w), int(y/sy*self.h)]
                print("texCoords: %s" % texCoords)
                """
                # update brush coords here
                self.computeProgram['brush'] = [xpos, ypos]
                self.computeProgram['brushtype'] = event.button

    def on_mouse_release(self, event):
        self.mouseDown = False
        self.computeProgram['brush'] = [0, 0]
        self.computeProgram['brushtype'] = 0

    NO_ACTION = (None, None)

    def on_key_press(self, event):
        # treats all key event that are defined in keyactionDictionnary
        eventKey = ''
        eventModifiers = []
        if len(event.text) > 0:
            eventKey = event.text
        else:
            eventKey = event.key.name
        for each in event.modifiers:
            eventModifiers.append(each)
        eventModifiers = tuple(eventModifiers)
        (func, args) = self.keyactionDictionnary.get((eventKey, eventModifiers), self.NO_ACTION)
        if func is not None:
            func(event, *args)

    ############################################################################
    # functions related to the Gray-Scott model parameters

    def initializeGrid(self, event=None):
        print('Initialization of the grid.', end="\r")
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
        specieName = Canvas.speciesDictionnary.get(event.text) or Canvas.speciesDictionnaryShifted.get(event.text)
        if specieName is not None:
            self.setSpecie(specieName)

    def modulateFK(self, pos=None):
        f = self.P[0, 0, 2]
        k = self.P[0, 0, 3]
        if pos is not None:
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
        if self.cycle == 0:
            self.cycle = 1
        else:
            self.cycle *= 2
        if self.cycle > 64:
            self.cycle = 64
        print(' Number of cycles: %3.0f' % (1 + 2 * self.cycle), end='\r')

    def decreaseCycle(self, event=None):
        self.cycle = int(self.cycle/2)
        if self.cycle < 1:
            self.cycle = 0
        print(' Number of cycles: %3.0f' % (1 + 2 * self.cycle), end='\r')

    def pause(self, event=None):
        self.holdModel = not self.holdModel

    ############################################################################
    # functions related to the Gray-Scott model appearances/representation

    def switchReagent(self, event=None):
        self.renderProgram["reagent"] = 1 - self.renderProgram["reagent"]
        self.shadowProgram["reagent"] = 1 - self.shadowProgram["reagent"]
        reagents = ('U', 'V')
        print(' Displaying %s reagent concentration.' % reagents[int(self.renderProgram["reagent"])], end="\r")

    def pickColorMap(self, event):
        colorMapName = Canvas.colormapDictionnary.get(event.text) or Canvas.colormapDictionnaryShifted.get(event.text)
        if colorMapName is not None:
            self.setColormap(colorMapName)

    def setColormap(self, name):
        self.cmapName = name
        self.renderProgram["cmap"] = get_colormap(self.cmapName).map(np.linspace(0.0, 1.0, 1024)).astype('float32')
        print(' Using colormap %s.' % name, end="\r")

    SHADOW_TYPE = ("None                     ",
                   "Simple                   ",
                   "Percent 4 samplings      ",
                   "Variance shadow mapping  ")

    def modifyLightCharacteristic(self, event, lightType=None, modification=None):
        if lightType == 'ambient':
            self.ambient = not self.ambient
            self.renderProgram[lightType] = self.ambient
            print(' Ambient light: %s        ' % self.ambient, end="\r")
        elif lightType == 'diffuse':
            self.diffuse = not self.diffuse
            self.renderProgram[lightType] = self.diffuse
            print(' Diffuse light: %s        ' % self.diffuse, end="\r")
        elif lightType == 'specular':
            self.specular = not self.specular
            self.renderProgram[lightType] = self.specular
            print(' Specular light: %s       ' % self.specular, end="\r")
        elif lightType == 'shadow':
            self.shadow = (self.shadow + 1) % 4
            self.renderProgram[lightType] = self.shadow
            print(' Shadows: %s              ' % self.SHADOW_TYPE[self.shadow], end="\r")
        elif lightType == 'attenuation':
            self.attenuation = not self.attenuation
            self.renderProgram[lightType] = self.attenuation
            print(' Attenuation: %s          ' % self.attenuation, end="\r")
        elif lightType == 'shininess' and modification == '-':
            self.shininess *= sqrt(2)
            self.shininess = np.clip(self.shininess, 0.1, 8192)
            self.renderProgram['u_Shininess'] = self.shininess
            print(' Shininess exponant: %3.0f' % self.shininess, end="\r")
        elif lightType == 'shininess' and modification == '+':
            self.shininess /= sqrt(2)
            self.shininess = np.clip(self.shininess, 0.1, 8192)
            self.renderProgram['u_Shininess'] = self.shininess
            print(' Shininess exponant: %3.0f' % self.shininess, end="\r")

    ############################################################################
    # functions to manipulate orientations and positions of model, view, light

    def resetModel(self, event=None):
        self.modelAzimuth = 0
        self.modelDirection = 0
        self.updateModel()

    def updateModel(self):
        model = self.rotateModel()
        if hasattr(self, 'renderProgram'):
            self.renderProgram["u_model"] = model
        if hasattr(self, 'shadowProgram'):
            self.shadowProgram["u_model"] = model
        if hasattr(self, 'coordinatesProgram'):
            self.coordinatesProgram["u_model"] = model

    def rotateModel(self):
        # model is able to rotate but does not move
        # around x axis (vertical to horizontal plane)
        azRotationMatrix = rotate(self.modelAzimuth, (1, 0, 0))
        # print('azRotationMatrix:\n %s' % azRotationMatrix)
        # around y axis (like a turntable)
        diRotationMatrix = rotate(self.modelDirection, (0, 0, 1))
        # print('diRotationMatrix:\n %s' % diRotationMatrix)
        return np.matmul(np.matmul(self.model, diRotationMatrix), azRotationMatrix)

    def updateLight(self, event=None, coordinatesDelta=(0,0,0)):
        # light is able to move along x,y,z axis
        self.lightCoordinates[0] += coordinatesDelta[0]
        self.lightCoordinates[1] += coordinatesDelta[1]
        self.lightCoordinates[2] += coordinatesDelta[2]
        self.lightCoordinates[0] = np.clip(self.lightCoordinates[0], -1, 1)
        self.lightCoordinates[1] = np.clip(self.lightCoordinates[1], -1, 1)
        if hasattr(self, 'renderProgram'):
            self.renderProgram["u_light_position"] = self.lightCoordinates
        self.updateLightCam()

    def updateView(self):
        # view is able to move along z axis only (in/out)
        self.view = translate((self.viewCoordinates[0],
                               self.viewCoordinates[1],
                               self.viewCoordinates[2]))
        if hasattr(self, 'renderProgram'):
            self.renderProgram["u_view"] = self.view
        if hasattr(self, 'coordinatesProgram'):
            self.coordinatesProgram["u_view"] = self.view
        self.updateLightCam()

    def updateLightCam(self):
        # lightcam is placed at the light coordinates
        self.shadowViewCoordinates = self.lightCoordinates
        self.shadowView = translate((-self.shadowViewCoordinates[0],
                                     -self.shadowViewCoordinates[1],
                                     self.shadowViewCoordinates[2]))
        # point camera at center of the model, and compute projection parameters
        self.shadowView, (self.shadowCamFoV, self.shadowCamNear, self.shadowCamFar) = self.rotateShadowView(self.shadowView,
                                                    self.shadowViewCoordinates,
                                                    self.shadowCamAt,
                                                    self.viewCoordinates)
        self.shadowProjection = perspective(self.shadowCamFoV,
                                            self.size[0] / float(self.size[1]),
                                            self.shadowCamNear,
                                            self.shadowCamFar)
        if hasattr(self, 'shadowProgram'):
            self.shadowProgram["u_view"] = self.shadowView
            self.shadowProgram["u_projection"] = self.shadowProjection
        if hasattr(self, 'renderProgram'):
            self.renderProgram["u_Shadowmap_view"] = self.shadowView
            self.renderProgram["u_Shadowmap_projection"] = self.shadowProjection

    def rotateShadowView(self, shadowView, eye, at, cam):
        azimuth = 180. / pi * atan((eye[0]-at[0])/(eye[2]-at[2]))
        declination = -180. / pi * atan((eye[1]-at[1])/(eye[2]-at[2]))
        azRotationMatrix = rotate(azimuth, (0, 1, 0))
        deRotationMatrix = rotate(declination, (1, 0, 0))
        rotatedView = np.matmul(np.matmul(shadowView, deRotationMatrix), azRotationMatrix)
        # Attempt at computing a fov adequate to encompass the model
        # The object being a simple square plane of 1 x 1, but this one being
        # orientable, let's just consider a sphere with radius = half of the diagonal
        # of the square, with 2% more
        # one could also modulate the radius following the closeness of the main camera?
        length = np.linalg.norm(np.subtract(at, eye))
        radius = 1.0 / sqrt(2) * 1.02
        radiusForFOV = radius * np.clip((cam[2] / -1.25), 0.0, 1.0)
        radiusForNearFar = radius * np.clip((cam[2] / -0.9), 0.0, 1.0)
        fov = 2 * asin(radius / length) * 180. / pi
        near = length - radius
        far = length + radius
        return rotatedView, (fov, near, far)

    ############################################################################
    # Output functions

    def printPearsonPatternDescription(self):
        self.title2 = '3D Gray-Scott Reaction-Diffusion: Pattern %s - GregVDS' % self.specie
        print('Pearson\'s Pattern %s' % self.specie)
        print(self.species[self.specie][4])
        print('        dU  dV  f     k \n        %s %s %s %s' % (self.species[self.specie][0],
                                                                 self.species[self.specie][1],
                                                                 self.species[self.specie][2],
                                                                 self.species[self.specie][3]))

    ############################################################################
    # Debug functions

    def switchDisplay(self, event=None):
        self.displaySwitch = (self.displaySwitch + 1) % 3

    def measure_fps2(self, window=1, callback=' %1.1f FPS                   '):
        """Measure the current FPS

        Sets the update window, connects the draw event to update_fps
        and sets the callback function.

        Parameters
        ----------
        window : float
            The time-window (in seconds) to calculate FPS. Default 1.0.
        callback : function | str
            The function to call with the float FPS value, or the string
            to be formatted with the fps value and then printed. The
            default is ``'%1.1f FPS'``. If callback evaluates to False, the
            FPS measurement is stopped.
        """
        # Connect update_fps function to draw
        self.events.draw.disconnect(self._update_fps)
        if callback:
            if isinstance(callback, str):
                callback_str = callback  # because callback gets overwritten

                def callback(x):
                    print(callback_str % x, end="\r")

            self._fps_window = window
            self.events.draw.connect(self._update_fps)
            self._fps_callback = callback
        else:
            self._fps_callback = None


################################################################################
def fun(x):
    c.title = c.title2 +' - FPS: %0.1f' % x


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
    c.measure_fps2(callback=fun)
    app.run()
